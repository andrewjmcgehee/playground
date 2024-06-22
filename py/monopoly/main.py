import functools
import logging
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

A = "A"
B = "B"
GO_MONEY = 200


@dataclass
class Space:
  index: int
  value: int
  name: str
  mortgaged: bool = False


@dataclass
class Character:
  id: int
  space: Space
  roll: int
  money: int = 0
  has_goojf: bool = False

  def __str__(self):
    return f"{self.id} {self.space.name} {self.money} {self.has_goojf}"

  def __repr__(self):
    return self.__str__()


chance_cards = list(range(16))
community_chest_cards = list(range(16))
random.shuffle(chance_cards)
random.shuffle(community_chest_cards)


def or_gen(spaces_map, names):
  mortgaged = [spaces_map[n].mortgaged for n in names]
  return any(mortgaged)


def utility_fn(character, other_character, factor):
  if character.id == B:
    character.money -= factor * character.roll


def check_sets():
  global spaces_map
  sm = spaces_map
  if or_gen(spaces_map, ["baltic", "mediterranean"]):
    sm["baltic"].value /= 2
    sm["mediterranean"].value /= 2
  if or_gen(spaces_map, ["oriental", "vermont", "connecticut"]):
    sm["oriental"].value /= 2
    sm["vermont"].value /= 2
    sm["connecticut"].value /= 2
  if or_gen(spaces_map, ["st_charles", "states", "virginia"]):
    sm["st_charles"].value /= 2
    sm["states"].value /= 2
    sm["virginia"].value /= 2
  if or_gen(spaces_map, ["st_james", "tennessee", "new_york"]):
    sm["st_james"].value /= 2
    sm["tennessee"].value /= 2
    sm["new_york"].value /= 2
  if or_gen(spaces_map, ["kentucky", "indiana", "illinois"]):
    sm["kentucky"].value /= 2
    sm["indiana"].value /= 2
    sm["illinois"].value /= 2
  if or_gen(spaces_map, ["atlantic", "ventnor", "marvin_gardens"]):
    sm["atlantic"].value /= 2
    sm["ventnor"].value /= 2
    sm["marvin_gardens"].value /= 2
  if or_gen(spaces_map, ["pacific", "north_carolina", "pennsylvania"]):
    sm["pacific"].value /= 2
    sm["north_carolina"].value /= 2
    sm["pennsylvania"].value /= 2
  if or_gen(spaces_map, ["park_place", "boardwalk"]):
    sm["park_place"].value /= 2
    sm["boardwalk"].value /= 2
  # utilities
  if or_gen(spaces_map, ["water", "electric"]):
    fn = functools.partial(utility_fn, factor=4)
    sm["electric"].value = fn
    sm["water"].value = fn
  # railroads
  rr = ["reading_rr", "pennsylvania_rr", "bo_rr", "short_line_rr"]
  rr_mortgaged = sum([sm[n].mortgaged for n in rr])
  if rr_mortgaged < 4:
    sm["reading_rr"].value //= 2**rr_mortgaged
    sm["pennsylvania_rr"].value //= 2**rr_mortgaged
    sm["bo_rr"].value //= 2**rr_mortgaged
    sm["short_line_rr"].value //= 2**rr_mortgaged


def community_chest_fn(character, other_character):
  global community_chest_cards
  if not community_chest_cards:
    community_chest_cards = list(range(16))
    random.shuffle(community_chest_cards)
  card = community_chest_cards.pop()
  if card == 0:
    logging.debug("community chest: advance to go")
    character.space = spaces_map["go"]
    logging.debug(f"{character.id} passed GO")
    character.money += GO_MONEY
  elif card == 1:
    logging.debug("community chest: bank error in your favor $200")
    character.money += 200
  elif card == 2:
    logging.debug("community chest: doctor fee $50")
    character.money -= 50
  elif card == 3:
    logging.debug("community chest: sale of stock $50")
    character.money += 50
  elif card == 4:
    logging.debug("community chest: get out of jail free")
    character.has_goojf = True
  elif card == 5:
    logging.debug("community chest: go directly to jail")
    character.space = spaces_map["visiting_jail"]
    if character.has_goojf:
      character.has_goojf = False
    character.money += spaces_map["go_to_jail"].value
  elif card == 6:
    logging.debug("community chest: holidy fund $100")
    character.money += 100
  elif card == 7:
    logging.debug("community chest: income tax refund")
    character.money += 20
  elif card == 8:
    logging.debug("community chest: birthday; collect $10 from other players")
    other_character.money -= 10
    character.money += 10
  elif card == 9:
    logging.debug("community chest: life insurance $100")
    character.money += 100
  elif card == 10:
    logging.debug("community chest: hospital fee $100")
    character.money -= 100
  elif card == 11:
    logging.debug("community chest: school fee $50")
    character.money -= 50
  elif card == 12:
    logging.debug("community chest: consultancy fee $25")
    character.money += 25
  elif card == 13:
    logging.debug("community chest: house and hotel repairs")
  elif card == 14:
    logging.debug("community chest: second place in beauty contest $10")
    character.money += 10
  elif card == 15:
    logging.debug("community chest: inherit $100")
    character.money += 100
  else:
    raise NotImplementedError()


def handle_space(character, other_character, space):
  if not space.mortgaged:
    if type(space.value) == int:
      # a can only lose money by landing on tax spaces which have negative values
      if character.id == A:
        if space.value < 0:
          logging.debug(f"{character.id} paid the bank ${abs(space.value)}")
          character.money += character.space.value
      else:
        # b can lose money by landing on any space other than go/chance/comm chest
        if space != spaces_map["go_to_jail"]:
          character.money -= abs(space.value)
        if space not in [
            spaces_map["luxury_tax"],
            spaces_map["income_tax"],
            spaces_map["go_to_jail"],
        ]:
          other_character.money += space.value
          logging.debug(f"{character.id} paid {other_character.id} ${abs(space.value)}")
        else:
          logging.debug(f"{character.id} paid the bank ${abs(space.value)}")
    else:
      space.value(character, other_character)


def chance_fn(character, other_character):
  global chance_cards
  if not chance_cards:
    chance_cards = list(range(16))
    random.shuffle(chance_cards)
  card = chance_cards.pop()
  # same outcome for both players
  if card == 1:
    character.space = spaces_map["go"]
    logging.debug("chance: advance to go")
    logging.debug(f"{character.id} passed GO")
    character.money += GO_MONEY
  elif card == 7:
    logging.debug("chance: bank dividend $50")
    character.money += 50
  elif card == 8:
    logging.debug("chance: get out of jail free card")
    character.has_goojf = True
  elif card == 10:  # go directly to jail
    logging.debug("chance: go directly to jail")
    character.space = spaces_map["visiting_jail"]
    if character.has_goojf:
      character.has_goojf = False
    character.money += spaces_map["go_to_jail"].value
  elif card == 11:
    logging.debug("chance: house and hotel repairs")
  elif card == 12:
    logging.debug("chance: speeding fine")
    character.money -= 15
  elif card == 14:
    logging.debug("chance: chairman of the board, pay other players $50")
    character.money -= 50
    other_character.money += 50
    logging.debug(f"{character.id} paid {other_character.id} $50")
  elif card == 15:
    logging.debug("chance: building loan matures $150")
    character.money += 150
  # different outcomes, assume A owns all property
  elif card == 0:
    logging.debug("chance: advance to boardwalk")
    character.space = spaces_map["boardwalk"]
  elif card == 2:
    logging.debug("chance: advance to illinois")
    if character.space.index > spaces_map["illinois"].index:
      character.money += GO_MONEY
      logging.debug(f"{character.id} passed GO")
    character.space = spaces_map["illinois"]
  elif card == 3:
    logging.debug("chance: advance to st charles")
    if character.space.index > spaces_map["st_charles"].index:
      character.money += GO_MONEY
      logging.debug(f"{character.id} passed GO")
    character.space = spaces_map["st_charles"]
  elif card in {4, 5}:
    logging.debug("chance: advance to nearest rail road")
    if character.space.index == spaces_map["chance_2"].index:
      character.money += GO_MONEY
      character.space = spaces_map["reading_rr"]
      logging.debug(f"{character.id} passed GO")
    elif character.space.index == spaces_map["chance_0"].index:
      character.space = spaces_map["pennsylvania_rr"]
    else:  # chance_1
      character.space = spaces_map["bo_rr"]
  elif card == 6:
    logging.debug("chance: advance to nearest utility and pay 10x roll")
    if character.space.index == spaces_map["chance_1"].index:
      character.space = spaces_map["water"]
    else:
      if character.space == spaces_map["chance_2"]:
        character.money += GO_MONEY
        logging.debug(f"{character.id} passed GO")
      character.space = spaces_map["electric"]
  elif card == 9:
    logging.debug("chance: go back 3 spaces")
    if character.space == spaces_map["chance_2"]:
      character.space = spaces_map["community_chest_2"]
    elif character.space == spaces_map["chance_1"]:
      character.space = spaces_map["new_york"]
    else:
      character.space = spaces_map["income_tax"]
  elif card == 13:
    logging.debug("chance: advance to reading rail road")
    if character.space.index > spaces_map["reading_rr"].index:
      character.money += GO_MONEY
      logging.debug(f"{character.id} passed GO")
    character.space = spaces_map["reading_rr"]
  if card in {0, 2, 3, 4, 5, 6, 9, 13}:
    handle_space(character, other_character, character.space)


def roll():
  return random.randint(1, 6) + random.randint(1, 6)


spaces = [
    Space(0, 0, "go"),
    Space(1, 4, "mediterranean"),
    Space(2, community_chest_fn, "community_chest_0"),
    Space(3, 8, "baltic"),
    Space(4, -200, "income_tax"),
    Space(5, 200, "reading_rr"),
    Space(6, 12, "oriental"),
    Space(7, chance_fn, "chance_0"),
    Space(8, 12, "vermont"),
    Space(9, 16, "connecticut"),
    Space(10, 0, "visiting_jail"),
    Space(11, 20, "st_charles"),
    Space(12, functools.partial(utility_fn, factor=10), "electric"),
    Space(13, 20, "states"),
    Space(14, 24, "virginia"),
    Space(15, 200, "pennsylvania_rr"),
    Space(16, 28, "st_james"),
    Space(17, community_chest_fn, "community_chest_1"),
    Space(18, 28, "tennessee"),
    Space(19, 32, "new_york"),
    Space(20, 0, "free_parking"),
    Space(21, 36, "kentucky"),
    Space(22, chance_fn, "chance_1"),
    Space(23, 36, "indiana"),
    Space(24, 40, "illinois"),
    Space(25, 200, "bo_rr"),
    Space(26, 44, "atlantic"),
    Space(27, 44, "ventnor"),
    Space(28, functools.partial(utility_fn, factor=10), "water"),
    Space(29, 48, "marvin_gardens"),
    Space(30, -50, "go_to_jail"),
    Space(31, 52, "pacific"),
    Space(32, 52, "north_carolina"),
    Space(33, community_chest_fn, "community_chest_2"),
    Space(34, 56, "pennsylvania"),
    Space(35, 200, "short_line_rr"),
    Space(36, chance_fn, "chance_2"),
    Space(37, 70, "park_place"),
    Space(38, -100, "luxury_tax"),
    Space(39, 100, "boardwalk"),
]
spaces_map = {space.name: space for space in spaces}


def play_game(args):
  _, spaces, spaces_map = args
  a = Character(id=A, space=spaces_map["go"], roll=0, money=1000)
  b = Character(id=B, space=spaces_map["go"], roll=0, money=1000)
  a_ev = defaultdict(float)
  b_ev = defaultdict(float)
  failed = -1
  for i in range(100):
    a.roll = roll()
    a_index = a.space.index + a.roll
    a.space = spaces[a_index % 40]
    logging.debug(f"A rolled {a.roll} and landed on {a.space.name}")
    if a_index >= 40:
      logging.debug("A passed GO")
      a.money += GO_MONEY
    before_money = a.money
    before_space = a.space
    handle_space(a, b, a.space)
    after_money = a.money
    a_ev[before_space.name] += after_money - before_money
    if a.space == spaces_map["go_to_jail"]:
      a_ev["go_to_jail"] += -50
      a.space = spaces_map["visiting_jail"]
    b.roll = roll()
    b_index = b.space.index + b.roll
    b.space = spaces[b_index % 40]
    logging.debug(f"B rolled {b.roll} and landed on {b.space.name}")
    if b_index >= 40:
      logging.debug("B passed GO")
      b.money += GO_MONEY
    before_money = b.money
    before_space = b.space
    handle_space(b, a, b.space)
    after_money = b.money
    b_ev[before_space.name] += after_money - before_money
    if b.space == spaces_map["go_to_jail"]:
      b_ev["go_to_jail"] += -50
      b.space = spaces_map["visiting_jail"]
    logging.debug(f"A: {a.money}  B: {b.money}\n")
    if b.money <= 0:
      failed = i
      break
  if b.money < 0:
    a.money -= b.money
    b.money = 0
  for k in a_ev:
    a_ev[k] /= 100
  for k in b_ev:
    b_ev[k] /= 100
  return (a.money, b.money, failed, a_ev, b_ev)


def dict_sum(a, b):
  for k in a:
    if k in b:
      a[k] += b[k]
  for k in b:
    if k not in a:
      a[k] = b[k]
  return a


def main():
  # spaces_map["reading_rr"].mortgaged = True
  # spaces_map["pennsylvania_rr"].mortgaged = True
  # spaces_map["bo_rr"].mortgaged = True
  # spaces_map["short_line_rr"].mortgaged = True
  # spaces_map["electric"].mortgaged = True
  # spaces_map["water"].mortgaged = True
  # spaces_map["boardwalk"].mortgaged = True
  # spaces_map["park_place"].mortgaged = True
  # spaces_map["pacific"].mortgaged = True
  # spaces_map["north_carolina"].mortgaged = True
  # spaces_map["pennsylvania"].mortgaged = True
  # spaces_map["mediterranean"].mortgaged = True
  # spaces_map["baltic"].mortgaged = True
  # spaces_map["oriental"].mortgaged = True
  # spaces_map["vermont"].mortgaged = True
  # spaces_map["connecticut"].mortgaged = True
  # spaces_map["atlantic"].mortgaged = True
  # spaces_map["ventnor"].mortgaged = True
  # spaces_map["marvin_gardens"].mortgaged = True
  # spaces_map["kentucky"].mortgaged = True
  # spaces_map["indiana"].mortgaged = True
  # spaces_map["illinois"].mortgaged = True
  # spaces_map["kentucky"].mortgaged = True
  # spaces_map["indiana"].mortgaged = True
  # spaces_map["illinois"].mortgaged = True
  check_sets()

  N = 500_000

  success = 0
  true_failure = 0
  a_growth = 0
  b_growth = 0
  a_ev = dict()
  b_ev = dict()
  pool = mp.Pool(processes=mp.cpu_count())
  for i, result in enumerate(
      pool.imap_unordered(
          play_game,
          zip(range(N), [spaces] * N, [spaces_map] * N),
      )
  ):
    a_money, b_money, fail_iter, a_i_ev, b_i_ev = result
    a_growth += (a_money - 1000) / 1000
    b_growth += (b_money - 1000) / 1000
    success += int(b_money >= 1000)
    true_failure += b_money <= 0
    a_ev = dict_sum(a_ev, a_i_ev)
    b_ev = dict_sum(b_ev, b_i_ev)
    if i % 100 == 0:
      print(
          f"b grows: {success / (i+1):.04f}\t"
          f"b survives: {1 - true_failure / (i+1):.04f}\t"
          f"a growth: {a_growth / (i+1):.04f}\t"
          f"b_growth: {b_growth / (i+1):.04f}",
          end="\r",
      )
  print()
  for k, v in sorted(a_ev.items(), key=lambda x: x[1]):
    print(f"{k:<20s}: {v / N:>7.04f}")
  for k, v in sorted(b_ev.items(), key=lambda x: x[1]):
    print(f"{k:<20s}: {v / N:>7.04f}")


if __name__ == "__main__":
  main()
