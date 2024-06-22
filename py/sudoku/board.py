from dataclasses import dataclass

import numpy as np

N = 9
SQRT_N = 3
ZERO = {0}


@dataclass
class Board:
  state: np.ndarray = None
  readonly_mask: np.ndarray = None
  solution: np.ndarray = None
  loaded: bool = False

  def __init__(self):
    self._generate()
    self._mask()
    self._check()
    self.loaded = True

  def _check(self):
    for i in range(N):
      for j in range(N):
        assert set(self.solution[i, :]) == set(range(1, N + 1))
        assert set(self.solution[:, j]) == set(range(1, N + 1))
    for i in range(0, N, 3):
      for j in range(0, N, 3):
        assert set(
            np.reshape(self.solution[i : i + SQRT_N, j : j + SQRT_N], -1)
        ) == set(range(1, N + 1))

  def _square(self, i, j):
    a = (i // SQRT_N) * SQRT_N
    b = a + SQRT_N
    c = (j // SQRT_N) * SQRT_N
    d = c + SQRT_N
    return a, b, c, d

  def _fill(self):
    stack = [(0, self.solution.copy())]
    while stack:
      norm, board = stack.pop()
      i, j = norm // N, norm % N
      if board[i, j]:
        stack.append((norm + 1, board))
        continue
      row = set(board[i, :]) - ZERO
      col = set(board[:, j]) - ZERO
      a, b, c, d = self._square(i, j)
      square = set(np.reshape(board[a:b, c:d], -1)) - ZERO
      possible = set(range(1, N + 1)) - (row | col | square)
      if not possible:
        continue
      if i == N - 1 and j == N - SQRT_N - 1:
        board[i, j] = possible.pop()
        self.solution = board.copy()
        return
      for val in possible:
        tmp = board.copy()
        tmp[i, j] = val
        stack.append((norm + 1, tmp))

  def _generate(self):
    self.solution = np.zeros((9, 9)).astype(np.uint8)
    tl = (np.random.choice(9, size=(3, 3), replace=False) + 1).astype(np.uint8)
    md = (np.random.choice(9, size=(3, 3), replace=False) + 1).astype(np.uint8)
    br = (np.random.choice(9, size=(3, 3), replace=False) + 1).astype(np.uint8)
    self.solution[0:3, 0:3] = tl
    self.solution[3:6, 3:6] = md
    self.solution[6:9, 6:9] = br
    self._fill()
    self.state = self.solution.copy()

  def _mask(self, rate=0.5):
    self.readonly_mask = np.random.random(size=(9, 9)) < rate
    self.state[~self.readonly_mask] = 0

  def _print_state(self):
    rep = ["   A B C D E F G H I", "⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻"]
    for i in range(N):
      rep.append([])
      rep[-1].append(f"{i}⏐")
      for j in range(N):
        if self.state[i, j] != 0:
          rep[-1].append(f" {self.state[i, j]}")
        else:
          rep[-1].append(" .")
      rep[-1] = "".join(rep[-1])
    print("\n".join(rep))

  def run(self):
    while True:
      self._print_state()
      input()


if __name__ == "__main__":
  b = Board()
  b.run()
