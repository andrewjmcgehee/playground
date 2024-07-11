import argparse
import concurrent.futures as concurrent
from datetime import datetime
import functools
import itertools
import logging
import multiprocessing as mp
import os
import re

import bs4
import numpy as np
import pandas as pd
import requests
import tqdm

import constants as C
import util

ap = argparse.ArgumentParser()
ap.add_argument("--force_reload", action="store_true")
logging.basicConfig(level=logging.INFO, format="%(levelname).1s: %(message)s")
map = util.nmap
_SESSION = requests.Session()

# lambda functions
L = util.nameddict({
    "id": lambda x: x.split("/").pop(),
    "str": lambda x: x.text.strip().lower(),
    "datestr": lambda x: re.findall(r"[A-Za-z]+ \d+, \d{4}", x).pop(),
    "cast": lambda x: float(x) if '.' in x else int(x),
    "number": lambda x: list(map(L.cast, re.findall(r"(\d+(?:\.\d+)?)", x))),
})


def get_soup(object):
  if isinstance(object, (bs4.BeautifulSoup, bs4.element.Tag)):
    return object
  if object.startswith("http://"):
    return bs4.BeautifulSoup(
        _SESSION.get(object).text.encode("ascii", "replace"), "html.parser")
  with open(object, "rb") as f:
    text = f.read()
  return bs4.BeautifulSoup(text, "html.parser")


def tags(object, tag, field=None, regex=".*", get_field=False, limit=None):
  soup = get_soup(object)
  if limit == 1:
    if field is not None:
      tag = soup.find(tag, {field: re.compile(regex)})
    else:
      tag = soup.find(tag)
    if get_field:
      return tag.get(field)
    return tag
  if field is not None:
    tags = soup(tag, {field: re.compile(regex)}, limit=limit)
  else:
    tags = soup(tag, limit=limit)
  if get_field:
    return [t.get(field) for t in tags]
  return tags


def url_to_file(url):
  return "_".join([s for s in re.split(":|/|\.", url) if s])


def download_raw_html(url, base_dir):
  timestamp = None
  if isinstance(url, (tuple, list)):
    timestamp, url = url
  file = url_to_file(url)
  if timestamp:
    file = f"{file}_{timestamp}"
  response = _SESSION.get(url)
  with open(f"{os.path.join(base_dir, file)}", "wb") as f:
    f.write(response.text.encode("ascii", "replace"))
  return file


def download(links, base_dir):
  download_fn = functools.partial(download_raw_html, base_dir=base_dir)
  with concurrent.ThreadPoolExecutor(max_workers=C.MAX_THREADS) as exec:
    for _ in tqdm.tqdm(exec.map(download_fn, links), total=len(links)):
      continue


def _scrape_event_file(file):
  file = os.path.join(C.RAW_EVENTS_BASE, file)
  date = L.datestr(L.str(tags(tags(file, "ul", limit=1), "li", limit=1)))
  timestamp = int(datetime.strptime(date, "%B %d, %Y").timestamp())
  fights = tags(file, **C.FIGHT_KWARGS)
  fights = [f for f in fights if f not in C.OMIT]
  fighters = tags(file, **C.FIGHTER_KWARGS)
  return timestamp, fights, fighters


def scrape_event_files(event_files):
  fight_links, fighter_links = set(), set()
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for (timestamp, fights, fighters) in tqdm.tqdm(
        pool.imap_unordered(_scrape_event_file, event_files)):
      fight_links.update(zip(itertools.cycle([timestamp]), fights))
      fighter_links.update(fighters)
  return fight_links, fighter_links


def new_event_links(event_links):
  existing = os.listdir(C.RAW_EVENTS_BASE)
  return [l for l in event_links if url_to_file(l) not in existing]


def parse_fight(fight):
  fight_id, timestamp = fight.split("_")[-2:]
  if datetime.now() < datetime.fromtimestamp(int(timestamp) + C.ONE_DAY):
    return
  result = L.str(tags(fight, "i", limit=1))
  label, is_draw = int(result == 'w'), int(result == 'd')
  table = tags(fight, "table", limit=1)
  general = list(map(L.str, L.number, tags(table, "td", limit=10)))
  general[0] = list(map(L.id, tags(table, "a", "href", get_field=True,
                                   limit=2)))
  table = tags(fight, "table", limit=3).pop()
  specific = list(map(L.str, L.number, tags(table, "td", limit=9)))
  red, blue = list(), list()
  for i, d in enumerate(general + specific):
    if i not in C.FIGHT_DATA_IGNORE:
      if len(d) == 0:
        red.extend([np.nan] * (C.FIGHT_DATA_SIZES[i] // 2))
        blue.extend([np.nan] * (C.FIGHT_DATA_SIZES[i] // 2))
      elif len(d) == 2:  # single number for each fighter
        red.append(d[0])
        blue.append(d[1])
      elif len(d) == 4:  # pair of numbers per fighter (e.g., n of k, time)
        red.extend(d[:2])
        blue.extend(d[2:])
  return dict(
      zip(C.FIGHT_RECORD_KEYS,
          (fight_id, int(timestamp), label, is_draw, *red, *blue)))


def download_subroutine(force_reload=False):
  event_links = tags(C.ALL_EVENTS_URL, **C.EVENT_KWARGS)
  if not force_reload:
    event_links = new_event_links(event_links)
  if event_links:
    logging.info("downloading events html")
    download(event_links, C.RAW_EVENTS_BASE)
    logging.info("scraping fight and fighter links from events html")
    event_files = list(map(url_to_file, event_links))
    fight_links, fighter_links = scrape_event_files(event_files)
    logging.info("downloading fights html")
    download(fight_links, C.RAW_FIGHTS_BASE)
    logging.info("downloading fighters html")
    download(fighter_links, C.RAW_FIGHTERS_BASE)


def main():
  args = ap.parse_args()
  download_subroutine(force_reload=args.force_reload)
  fights = [
      os.path.join(C.RAW_FIGHTS_BASE, f) for f in os.listdir(C.RAW_FIGHTS_BASE)
  ]
  fight_df = []
  with mp.Pool(processes=mp.cpu_count()) as pool:
    for _, res in enumerate(
        tqdm.tqdm(pool.imap_unordered(parse_fight, fights), total=len(fights))):
      if res is not None:
        fight_df.append(res)
  pd.DataFrame.from_dict(fight_df).to_csv("fights.csv", index=None)


if __name__ == "__main__":
  main()