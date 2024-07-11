from datetime import datetime
import functools
import logging

import multiprocessing as mp
import pickle
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import tqdm

_ALL_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
_CLOSURES = {
    "height": (lambda h: 12 * int(h[:h.index("\'")]) + int(
        h.rstrip("\"").split("\'")[1])),
    "weight":
        lambda w: float(w.split()[0]),
    "reach":
        lambda r: float(r.replace("\"", "").replace("-", "")),
    "stance":
        lambda o: (1.0 if (1.0 if o.strip().lower() == "orthodox" else 0.0) else
                   (np.nan if not o.replace("-", "").strip() else 0.0)),
    "age":
        lambda a: float((datetime.now() - datetime.strptime(
            a.strip(), "%b %d, %Y")).days / 365),
    "numeric":
        lambda n: float(n),
    "percent":
        lambda p: float(p.rstrip("%")) / 100,
    "n_of_k":
        lambda s: tuple(map(int, s.split(" of "))),
    "time":
        lambda t: 60 * int(t[:t.index(":")]) + int(t.split(":")[1]),
}
_FIGHTER_KEYS = [
    "fighter_id", "name", "wins", "losses", "draws", "height_inches",
    "weight_lbs", "reach_inches", "is_orthodox", "age_years",
    "strikes_landed_per_minute", "strikes_accuracy_percent",
    "strikes_absorbed_per_minute", "strikes_defended_percent",
    "mean_takedowns_per_15_minutes", "takedowns_accuracy_percent",
    "takedowns_defended_percent", "mean_submissions_attempted_per_15_minutes"
]
_FIGHT_KEYS = [
    "fight_id", "datetime", "label", "is_draw", "red_fighter_id", "red_name",
    "red_knockdowns", "red_significant_strikes_landed",
    "red_significant_strikes_attempted",
    "red_significant_strikes_landed_percent", "red_strikes_landed",
    "red_strikes_attempted", "red_strikes_landed_percent",
    "red_takedowns_finished"
    "red_takedowns_attempted", "red_takedowns_finished_percent",
    "red_submissions_attempted", "red_control_time_seconds", "blue_fighter_id",
    "blue_name", "blue_knockdowns", "blue_significant_strikes_landed",
    "blue_significant_strikes_attempted",
    "blue_significant_strikes_landed_percent", "blue_strikes_landed",
    "blue_strikes_attempted", "blue_strikes_landed_percent",
    "blue_takedowns_finished"
    "blue_takedowns_attempted", "blue_takedowns_finished_percent",
    "blue_submissions_attempted", "blue_control_time_seconds"
]
_OMIT = {
    # fights
    "http://ufcstats.com/fight-details/8e03db41687d9132",
    "http://ufcstats.com/fight-details/b80e6a799c95d499",
    "http://ufcstats.com/fight-details/e4fe950846b51bdf",
    "http://ufcstats.com/fight-details/7ffcc3a72e082ace",
    "http://ufcstats.com/fight-details/6449a1a9a69a830c",
    "http://ufcstats.com/fight-details/635fbf57001897c7",
    "http://ufcstats.com/fight-details/a1db4c917777aa79",
    "http://ufcstats.com/fight-details/4b334c9727eee450",
    "http://ufcstats.com/fight-details/b297c3e938e1005e",
    "http://ufcstats.com/fight-details/f59b1215176636f6",
    "http://ufcstats.com/fight-details/c413b0abc04358c3",
    "http://ufcstats.com/fight-details/77bf1e37929b0d59",
    "http://ufcstats.com/fight-details/8b258bbb37f74a66",
    "http://ufcstats.com/fight-details/2f449bd58b3d9a99",
    "http://ufcstats.com/fight-details/565ecefd8a37ad7e",
    "http://ufcstats.com/fight-details/5701dbbbfa4f8313",
    "http://ufcstats.com/fight-details/4bce0ce561a65288",
    "http://ufcstats.com/fight-details/d93c8c77e1091a16",
    "http://ufcstats.com/fight-details/a5c90086fb65f58e",
    "http://ufcstats.com/fight-details/b80872821bc4f6ba",
    "http://ufcstats.com/fight-details/3badedeb2c5533f4"
}

logging.basicConfig(level=logging.INFO, format="%(levelname).1s: %(message)s")
S = requests.Session()


def get_soup(url):
  response = S.get(url, timeout=5)
  html = response.text.encode("ascii", "replace")
  return BeautifulSoup(html, "html.parser")


def write_cache(data, name):
  logging.info(f"writing cache: {name}")
  with open(f"cache/{name}", "wb") as f:
    pickle.dump(data, f)


def open_cache(name):
  with open(f"cache/{name}", "rb") as f:
    return pickle.load(f)


def scrape_links(url, tag, field, regex, limit=None):
  soup = get_soup(url)
  tags = soup(tag, {field: re.compile(regex)}, limit=limit)
  return [t.get(field) for t in tags]


def scrape_event(event):
  soup = get_soup(event)
  fight_links = scrape_links(url=event,
                             tag="tr",
                             field="data-link",
                             regex="http.*fight-details.*")
  fighter_links = scrape_links(url=event,
                               tag="a",
                               field="href",
                               regex="http.*fighter-details.*")
  event_id = event.split("/")[-1].strip()
  event_to_fight_links = {event_id: [f for f in fight_links]}
  event_to_timestamp = {
      event_id:
          datetime.strptime(
              soup.find("ul").find("li").text.split(":")[1].strip(),
              "%B %d, %Y").timestamp()
  }
  return {
      "fights": fight_links,
      "fighters": fighter_links,
      "fight_mapping": event_to_fight_links,
      "time_mapping": event_to_timestamp,
  }


def events_before_now_index(events):
  for i, event in enumerate(events):
    soup = get_soup(event)
    date = datetime.strptime(
        soup.find("ul").find("li").text.split(":")[1].strip(), "%B %d, %Y")
    if date < datetime.now():
      return i
  return -1


def _try_assign(in_value, closure, default_out_value=np.nan):
  try:
    return closure(in_value)
  except:
    return default_out_value


def fighter_name_and_record(fighter):
  name, record = [
      s.text.strip().lower() for s in fighter.find("h2").find_all("span")
  ]
  record = record.split()[1]
  wins, losses, draws = map(int, record.split("-"))
  return name, wins, losses, draws


def fighter_tape(fighter):
  tape = fighter.find("ul").find_all("li")
  h, w, r, o, a = [li.text.split(":")[1].strip() for li in tape]
  return (
      _try_assign(h, _CLOSURES["height"]),
      _try_assign(w, _CLOSURES["weight"]),
      _try_assign(r, _CLOSURES["reach"]),
      _try_assign(o, _CLOSURES["stance"]),
      _try_assign(a, _CLOSURES["age"]),
  )


def fighter_career_stats(fighter):
  strikes = fighter("ul", limit=2)[-1].find_all("li")
  takedowns_and_subs = fighter("ul", limit=3)[-1].find_all("li")
  stats = strikes + takedowns_and_subs
  slpm, sacc, sabs, sdef, tdavg, tdacc, tddef, savg = [
      li.text.split()[-1].strip() for li in stats if li.text.strip()
  ]
  return (
      _try_assign(slpm, _CLOSURES["numeric"]),
      _try_assign(sacc, _CLOSURES["percent"]),
      _try_assign(sabs, _CLOSURES["numeric"]),
      _try_assign(sdef, _CLOSURES["percent"]),
      _try_assign(tdavg, _CLOSURES["numeric"]),
      _try_assign(tdacc, _CLOSURES["percent"]),
      _try_assign(tddef, _CLOSURES["percent"]),
      _try_assign(savg, _CLOSURES["numeric"]),
  )


def get_fighter_details(fighter):
  fighter_id = fighter.split("/")[-1].strip()
  soup = get_soup(fighter)
  info = fighter_name_and_record(soup)
  tape = fighter_tape(soup)
  stats = fighter_career_stats(soup)
  return dict(list(zip(_FIGHTER_KEYS, (fighter_id, *info, *tape, *stats))))


def _get_corner_details(corner):
  n, kd, ss, _, ts, td, _, sub, _, c = corner
  (significant_strikes_landed,
   significant_strikes_attempted) = _try_assign(ss, _CLOSURES["n_of_k"])
  strikes_landed, strikes_attempted = _try_assign(ts, _CLOSURES["n_of_k"])
  takedowns_finished, takedowns_attempted = _try_assign(td, _CLOSURES["n_of_k"])
  return (
      n,  # name
      _try_assign(kd, _CLOSURES["numeric"]),  # knockdowns
      significant_strikes_landed,
      significant_strikes_attempted,
      significant_strikes_landed / max(significant_strikes_attempted, 1),
      strikes_landed,
      strikes_attempted,
      strikes_landed / max(strikes_attempted, 1),
      takedowns_finished,
      takedowns_attempted,
      takedowns_finished / max(takedowns_attempted, 1),
      _try_assign(sub, _CLOSURES["numeric"]),  # submissions attempted
      _try_assign(c, _CLOSURES["time"]),  # control time in seconds
  )


def get_fight_details(fight, timestamps):
  fight_id = fight.split("/")[-1].strip()
  soup = get_soup(fight)
  result = soup.find("i").text.strip().lower()
  table = soup.find("table")
  if table is None:
    logging.info(f"table not found for fight: {fight}")
    return
  data = [p.text.strip().lower() for p in table("p")]
  red_data, blue_data = data[::2], data[1::2]
  red_data = _get_corner_details(red_data)
  blue_data = _get_corner_details(blue_data)
  red_id, blue_id = scrape_links(url=fight,
                                 tag="a",
                                 field="href",
                                 regex="http.*fighter-details.*",
                                 limit=2)
  red_id, blue_id = red_id.split("/")[-1], blue_id.split("/")[-1]
  return dict(
      list(
          zip(
              _FIGHT_KEYS,
              (
                  fight_id,
                  int(timestamps[fight_id]),
                  int(result == "w"),  # label
                  int(result == "d"),  # is_draw
                  red_id,
                  *red_data,
                  blue_id,
                  *blue_data,
              ))))


def scrape_events_fighters_and_fights():
  logging.info("scraping events for fight and fighter links")
  all_events = scrape_links(url=_ALL_EVENTS_URL,
                            tag="a",
                            field="href",
                            regex="http.*event-details.*")
  filter_index = events_before_now_index(all_events)
  all_events = all_events[filter_index:]
  all_fights = set()
  all_fighters = set()
  events_to_fights = dict()
  events_to_timestamps = dict()
  fights_to_timestamps = dict()
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for event_data in tqdm.tqdm(pool.imap_unordered(scrape_event, all_events),
                                total=len(all_events)):
      all_fights.update(event_data["fights"])
      all_fighters.update(event_data["fighters"])
      events_to_fights.update(event_data["fight_mapping"])
      events_to_timestamps.update(event_data["time_mapping"])
  for event_id, fight_list in events_to_fights.items():
    for fight in fight_list:
      fight_id = fight.split("/")[-1]
      fights_to_timestamps[fight_id] = events_to_timestamps[event_id]
  write_cache(all_events, "events.pkl")
  write_cache(all_fights - _OMIT, "fights.pkl")
  write_cache(all_fighters, "fighters.pkl")
  write_cache(fights_to_timestamps, "fights_to_timestamps.pkl")


def scrape_fighter_details():
  logging.info("scraping fighter links for fighter tape info and career stats")
  all_fighters = open_cache("fighters.pkl")
  fighter_details = []
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for fighter in tqdm.tqdm(pool.imap_unordered(get_fighter_details,
                                                 all_fighters),
                             total=len(all_fighters)):
      fighter_details.append(fighter)
  logging.info("writing dataframe: data/fighters.csv")
  pd.DataFrame.from_records(fighter_details).to_csv("data/fighters.csv",
                                                    index=None)


def scrape_fight_details():
  logging.info("scraping fight links for fight-specific stats")
  all_fights = open_cache("fights.pkl")
  fights_to_timestamps = open_cache("fights_to_timestamps.pkl")
  fight_details = []
  worker = functools.partial(get_fight_details, timestamps=fights_to_timestamps)
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for fight in tqdm.tqdm(pool.imap_unordered(worker, all_fights),
                           total=len(all_fights)):
      fight_details.append(fight)
  logging.info("writing dataframe: data/fights.csv")
  pd.DataFrame.from_records(fight_details).to_csv("data/fights.csv", index=None)


if __name__ == "__main__":
  scrape_events_fighters_and_fights()
  scrape_fighter_details()
  scrape_fight_details()
