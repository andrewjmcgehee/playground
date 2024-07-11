from typing import List

import numpy as np


class SelfDrivingCar():

  def __init__(self, array: List[List[int]], rd=True):
    for row in array:
      for i in row:
        assert i >= 0, "array must contain non-negative integers only"
    self.rows = len(array)
    self.cols = len(array[0])
    self.array = array
    self.rd = rd  # car can only traverse rightward and downward
    self.directions = [(1, 0), (0, 1)]
    if not self.rd:
      self.directions.extend([(-1, 0), (0, -1)])

  def _check_bounds(self, row, col):
    row_safe = (row >= 0 and row < self.rows)
    col_safe = (col >= 0 and col < self.cols)
    return row_safe and col_safe

  def _dfs(self, time, row, col, visited):
    time = max(time, self.array[row][col])
    visited.add((row, col))
    if (row, col) == (self.rows - 1, self.cols - 1):
      return time
    neighbors = []
    for row_mod, col_mod in self.directions:
      next_row = row + row_mod
      next_col = col + col_mod
      if self._check_bounds(next_row, next_col):
        if (next_row, next_col) not in visited:
          next_time = self._dfs(time, next_row, next_col, visited.copy())
          neighbors.append(next_time)
    if not neighbors:
      return float('inf')
    return min(neighbors)

  def _dfs_check(self, time, row, col, visited, max_time):
    time = max(time, self.array[row][col])
    visited.add((row, col))
    if time > max_time:
      return False
    if (row, col) == (self.rows - 1, self.cols - 1):
      return True
    neighbors = []
    for row_mod, col_mod in self.directions:
      next_row = row + row_mod
      next_col = col + col_mod
      if self._check_bounds(next_row, next_col):
        if (next_row, next_col) not in visited:
          possible = self._dfs_check(time, next_row, next_col, visited.copy(),
                                     max_time)
          neighbors.append(possible)
    # no neighbors under the time limit found
    if not neighbors:
      return False
    return True in neighbors

  def dfs_solve(self) -> int:
    """ Simple DFS solution """
    visited = set()
    return self._dfs(0, 0, 0, visited)

  def binary_search_dfs_solve(self) -> int:
    """ Limit the number of DFS paths by checking is it possible to solve in
    time T (where T is the max time in the array). If so, check the lower half
    of T/2. If not check the upper half of T/2. """
    hi, lo = 0, 0
    for row in self.array:
      hi = max(hi, max(row))
    # we know that its possible to solve at the max time so we explore the lower
    # half first
    current = hi // 2
    while hi > lo:
      print(f'trying: t={current}, lo={lo}, hi={hi}')
      visited = set()
      if self._dfs_check(0, 0, 0, visited, max_time=current):
        print('possible')
        hi = current
      else:
        print('not possible')
        lo = current + 1
      current = lo + (hi-lo) // 2
    return current

  def dp_2d_solve(self) -> int:
    memo = np.zeros((self.rows, self.cols))
    memo[0, 0] = self.array[0][0]
    # straight rightward
    for j in range(1, self.cols):
      memo[0, j] = max(memo[0, j - 1], self.array[0][j])
    # straight downward
    for i in range(1, self.rows):
      memo[i, 0] = max(memo[i - 1, 0], self.array[i][0])
    # remaining cells
    for i in range(1, self.rows):
      for j in range(1, self.cols):
        memo[i, j] = max(self.array[i][j], min(memo[i - 1, j], memo[i, j - 1]))
    print(memo)
    return memo[-1, -1]
