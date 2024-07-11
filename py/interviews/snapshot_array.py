import bisect
import logging

from typing import Any


class SnapshotArray():

  def __init__(self, size: int):
    self.size = size
    self.state = [0] * size
    self.snapshots = [list() for _ in range(size)]
    self.modified = dict()
    self.id = 0

  def __check_index(self, index: int):
    try:
      self.state[index]
    except IndexError:
      raise

  def get(self, index: int) -> Any:
    self.__check_index(index)
    return self.state[index]

  def set(self, index: int, value: Any):
    self.__check_index(index)
    # untracked, set to new value
    if index not in self.modified and self.state[index] != value:
      self.modified[index] = self.state[index]
    # tracked, set back to original value
    elif index in self.modified and self.modified[index] == value:
      self.modified.pop(index, None)
    # if an index is untracked and set to the original value, we do nothing
    # if an index is already tracked and set to anything other than the original
    # value, we also do nothing
    self.state[index] = value

  def snapshot(self) -> int:
    # first snapshot, capture all values
    if self.id == 0:
      for i, value in enumerate(self.state):
        self.snapshots[i].append((self.id, value))
    # update only modified indices
    elif self.modified:
      for i in self.modified.keys():
        self.snapshots[i].append((self.id, self.state[i]))
    else:
      logging.info("no values have been modified. no snapshot taken.")
      return self.id - 1
    self.modified.clear()
    self.id += 1
    return self.id - 1

  def query(self, snapshot_id: int, index: int) -> Any:
    self.__check_index(index)
    if snapshot_id >= self.id:
      logging.info("requested snaphot does not exist yet; returning the latest "
                   "result at the requested index")
    snapshot_ids, values = zip(*self.snapshots[index])
    snapshot_index = bisect.bisect(snapshot_ids, snapshot_id)
    return values[snapshot_index - 1]
