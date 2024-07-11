from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point:
  x: int
  y: int

  def __lt__(self, other):
    if self.x == other.x:
      return self.y > other.y
    return self.x < other.x

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __hash__(self):
    return hash(f"{self.x} {self.y}")


def cross(p, q, r):
  return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)


coords = np.random.randn(500, 2) * 50
points = [Point(x, y) for x, y in coords]


def convex_hull(points):
  points = sorted(set(points))
  if len(points) <= 1:
    return points
  lower = []
  for p in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)
  upper = []
  for p in reversed(points):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)
  return lower[:-1] + upper[:-1]


if __name__ == '__main__':
  ch = convex_hull(points)
  plt.scatter([p.x for p in points], [p.y for p in points], color='C0')
  plt.plot([p.x for p in ch], [p.y for p in ch], '-o', markersize=3, color="C1")
  plt.show()
