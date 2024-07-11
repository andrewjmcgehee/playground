import logging

logging.basicConfig(level=logging.INFO, format="%(levelname).1s: %(message)s")

import datetime
import multiprocessing as mp
import os
import pickle
import pprint

from bs4 import BeautifulSoup
import requests
import tqdm

_ALL_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
_NOW = datetime.datetime.now()
_POST_HOC = {
    "juancamilo ronderos": {
        "reach_inches": 64.5,
        "orthodox": 0
    },
    "jesus pinedo": {
        "reach_inches": 74.0,
    },
    "michel batista": {
        "reach_inches": 80.5,
    },
    "jay cucciniello": {
        "reach_inches": 68.0,
    },
    "kalindra faria": {
        "reach_inches": 62.0,
    }
}
_UNKNOWN_REV_COLUMN = 8

# caches
_EVENTS_CACHE = "events.pkl"
_FIGHTS_CACHE = "fights.pkl"
_FIGHTERS_CACHE = "fighters.pkl"
_MISSING_CACHE = "missing.pkl"

# classes
_FIGHT_DETAILS_CLASS = (
    "b-fight-details__table-row b-fight-details__table-row__hover "
    "js-fight-details-click")
_FIGHTER_DETAILS_MEDIUM_CLASS = (
    "b-list__info-box b-list__info-box_style_middle-width js-guide clearfix")
_FIGHTER_DETAILS_MEDIUM_LEFT_CLASS = "b-list__info-box-left"
_FIGHTER_DETAILS_SMALL_CLASS = (
    "b-list__info-box b-list__info-box_style_small-width js-guide")


def _get_fighters_from_row(details_row):
  fighter_1, fighter_2 = None, None
  fighter_1_name, fighter_2_name = None, None
  for cell in details_row.find_all("td",
                                   {"class": "b-fight-details__table-col"}):
    for link in cell.find_all("a"):
      href = link.get("href")
      if href is None or "fighter-details" not in href:
        continue
      name = link.text.strip().lower()
      if fighter_1 is None:
        fighter_1 = href
        fighter_1_name = name
      else:
        fighter_2 = href
        fighter_2_name = name
  return fighter_1, fighter_2, fighter_1_name, fighter_2_name


def _process_event_link(link):
  soup = _get_soup(link)
  fights = []
  for row in soup.find_all("tr", {"class": _FIGHT_DETAILS_CLASS}):
    fighter_1, fighter_2, name_1, name_2 = _get_fighters_from_row(row)
    href = row.get("data-link")
    fighter_1_details = _process_fighter_details(fighter_1, name_1)
    fighter_2_details = _process_fighter_details(fighter_2, name_2)
    fight_details = _process_fight_details(href)
    fights.append((fighter_1_details, fighter_2_details, fight_details))
  return fights


def _assert_assign_fighter_value(key, value, details):
  if key == "height":
    if "\'" not in value:
      raise ValueError(f"feet marker not found in height, {value}")
    if "\"" not in value:
      raise ValueError(f"inches marker not found in height, {value}")
    ft, inches = map(int, value.rstrip("\"").split("\'"))
    details["height_inches"] = 12*ft + inches
  elif key == "weight":
    if "lbs" not in value:
      raise ValueError(f"lbs not found in weight, {value}")
    weight, _ = value.split()
    details["weight_lbs"] = int(weight)
  elif key == "stance":
    details["orthodox"] = int(value.strip().lower() == "orthodox")
  elif key == "dob":
    dob = datetime.datetime.strptime(value, "%b %d, %Y")
    details["age_years"] = (_NOW - dob).days / 365
  elif key == "slpm":
    details["significant_strikes_landed_per_minute"] = float(value)
  elif key == "str. acc.":
    if "%" not in value:
      raise ValueError(f"% not found in significant strikes accuracy, {value}")
    value = value.strip().rstrip("%")
    details["significant_stikes_landed_percentage"] = float(value) / 100
  elif key == "sapm":
    details["significant_strikes_absorbed_per_minute"] = float(value)
  elif key == "str. def.":
    if "%" not in value:
      raise ValueError(
          f"% not found in significant strikes defended percentage, {value}")
    value = value.strip().rstrip("%")
    details["significant_stikes_defended_percentage"] = float(value) / 100
  elif key == "td avg.":
    details["mean_takedowns_per_15_minutes"] = float(value)
  elif key == "td acc.":
    if "%" not in value:
      raise ValueError(f"% not found in takedown accuracy, {value}")
    value = value.strip().rstrip("%")
    details["takedowns_finished_percentage"] = float(value) / 100
  elif key == "td def.":
    if "%" not in value:
      raise ValueError(f"% not found in takedowns defended percentage, {value}")
    value = value.strip().rstrip("%")
    details["takedowns_defended_percentage"] = float(value) / 100
  elif key == "sub. avg.":
    details["mean_submission_attempts_per_15_minutes"] = float(value)
  return details


def _process_fighter_details(href, name):
  soup = _get_soup(href)
  details = {"name": name}
  tape_box = soup.find("div", {"class": _FIGHTER_DETAILS_SMALL_CLASS})
  for li in tape_box.find_all("li"):
    key, value = map(str.strip, li.text.split(":"))
    try:
      details.update(_assert_assign_fighter_value(key.lower(), value, details))
    except ValueError:
      logging.info("attempting to add post hoc information for fighter: "
                   f"{details['name']} {href}")
      try:
        details.update(_POST_HOC[details["name"]])
      except:
        pass
  right_box = soup.find("div", {"class": _FIGHTER_DETAILS_MEDIUM_CLASS})
  stat_box = right_box.find("div",
                            {"class": _FIGHTER_DETAILS_MEDIUM_LEFT_CLASS})
  for li in stat_box.find_all("li"):
    if not li.text.strip():
      continue
    key, value = map(str.strip, li.text.split(":"))
    details.update(_assert_assign_fighter_value(key.lower(), value, details))
  return details


def _process_fight_details(href):
  soup = _get_soup(href)
  keys = [
      "red_corner", "red_knockdowns", "red_significant_strikes_landed",
      "red_significant_strikes_attempted", "red_significant_strikes_percentage",
      "red_total_strikes_landed", "red_total_strikes_attempted",
      "red_total_strikes_percentage", "red_takedowns_finished",
      "red_takedowns_attempted", "red_takedowns_percentage",
      "red_submissions_attempted", "red_control_time_seconds", "blue_corner",
      "blue_knockdowns", "blue_significant_strikes_landed",
      "blue_significant_strikes_attempted",
      "blue_significant_strikes_percentage", "blue_total_strikes_landed",
      "blue_total_strikes_attempted", "blue_total_strikes_percentage",
      "blue_takedowns_finished", "blue_takedowns_attempted",
      "blue_takedowns_percentage", "blue_submissions_attempted",
      "blue_control_time_seconds", "label"
  ]
  # red values in index 0, blue values in index 1
  values = [[], []]
  red_corner_result = soup.find("div", {"class": "b-fight-details__person"})
  label = int(red_corner_result.find("i").text.strip().lower() == "w")
  table = soup.find("table")
  data = table.find("tbody").find("tr")
  for j, cell in enumerate(data.find_all("td")):
    if j == _UNKNOWN_REV_COLUMN:
      continue
    for i, p in enumerate(cell.find_all("p")):
      text = p.text.strip().lower()
      if " of " in text:
        completed, attempted = map(int, text.split(" of "))
        percentage = completed / max(attempted, 1)
        values[i].extend([completed, attempted, percentage])
      elif "%" in text or "--" in text:
        continue
      elif ":" in text:
        m, s = map(int, text.split(":"))
        values[i].append(60*m + s)
      else:
        values[i].append(text)
  values = values[0] + values[1]
  values.append(label)
  assert len(keys) == len(values), f"key value pair mismatch {keys}, {values}"
  return dict(zip(keys, values))


def _get_soup(url):
  source = requests.get(url, allow_redirects=False, timeout=3)
  text = source.text.encode("ascii", "replace")
  return BeautifulSoup(text, "html.parser")


def _open_cache(cache, default_struct_fn=None):
  if os.path.exists(f"./{cache}"):
    logging.info(f"restoring cache {cache}")
    with open(f"./{cache}", "rb") as f:
      return pickle.load(f)
  return default_struct_fn()


def _write_cache(data, cache):
  logging.info(f"writing cache {cache}")
  with open(f"./{cache}", "wb") as f:
    pickle.dump(data, f)


def get_new_event_links():
  events = _open_cache(_EVENTS_CACHE, default_struct_fn=dict)
  new_events = []
  soup = _get_soup(_ALL_EVENTS_URL)
  table = soup.find("table", {"class": "b-statistics__table-events"})
  early_stop = False
  for row in table.find_all("tr"):
    date_span = row.find("span", {"class": "b-statistics__date"})
    if date_span is None:
      continue
    date = datetime.datetime.strptime(date_span.text.strip(), "%B %d, %Y")
    if _NOW >= date + datetime.timedelta(days=1):
      for link in row.find_all("a"):
        href = link.get("href")
        if href in events:
          early_stop = True
          break
        logging.debug(f"  adding event: {link.text.strip()}")
        events[href] = {"date": str(date.date()), "title": link.text.strip()}
        new_events.append(href)
      if early_stop:
        break
  logging.info(f"retrieved {len(events)} events")
  _write_cache(data=events, cache=_EVENTS_CACHE)
  return new_events


def get_all_fight_and_fighter_details(event_links):
  q = mp.Queue()
  pbar = tqdm.tqdm(total=len(event_links))
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for i, res in enumerate(
        pool.imap_unordered(_process_event_link, event_links)):
      q.put(res)
      if i % 10 == 0:
        _write_results(q)
      pbar.update(1)
  _write_results(q)
  # logging.info(f"retrieved {len(fighter_dict)} fighters")
  # logging.info(f"retrieved {len(fight_list)} fights")
  # _write_cache(data=fight_cache, cache=_FIGHTS_CACHE)
  # _write_cache(data=fighter_cache, cache=_FIGHTERS_CACHE)
  # _write_cache(data=missing_cache, cache=_MISSING_CACHE)


def _write_results(q):
  fighters_set = set()
  fighter_list = []
  fight_list = []
  while not q.empty():
    event = q.get()
    for fighter_1, fighter_2, fight in event:
      if fighter_1["name"] not in fighters_set:
        fighter_list.append(fighter_1)
      if fighter_2["name"] not in fighters_set:
        fighter_list.append(fighter_2)
      fight_list.append(fight)
  fighter_cache = _open_cache(_FIGHTERS_CACHE, default_struct_fn=list)
  fighter_cache.extend(fighter_list)
  fight_cache = _open_cache(_FIGHTS_CACHE, default_struct_fn=list)
  fight_cache.extend(fight_list)
  _write_cache(fighter_cache, _FIGHTERS_CACHE)
  _write_cache(fight_cache, _FIGHTS_CACHE)


if __name__ == "__main__":
  event_links = get_new_event_links()
  if event_links:
    get_all_fight_and_fighter_details(event_links)
