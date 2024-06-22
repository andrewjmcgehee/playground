import os

import pandas as pd
import pydot

files = sorted([f for f in os.listdir() if f.endswith(".csv")])


class Node:

  def __init__(self, data):
    self.data = data
    self.children = dict()

  def add_child(self, data):
    if data not in self.children:
      self.children[data] = Node(data)

  def __str__(self):
    return self.data + ": " + str(self.children)

  def __repr__(self):
    return self.__str__()


def move_list(s):
  s = s.strip().split()
  res = []
  for i, m in enumerate(s):
    if i % 3 == 0:
      continue
    res.append(m)
  return res


def draw(a, b, graph, edges):
  if f"{a} {b}" not in edges:
    edges.add(f"{a} {b}")
    e = pydot.Edge(a, b)
    graph.add_edge(e)


def populate(root):
  a = pd.read_csv("eco_a_openings.csv")
  b = pd.read_csv("eco_b_openings.csv")
  c = pd.read_csv("eco_c_openings.csv")
  d = pd.read_csv("eco_d_openings.csv")
  e = pd.read_csv("eco_e_openings.csv")
  for df in [a, b, c, d, e]:
    for i, row in df.iterrows():
      moves = move_list(row["pgn"])
      current = root
      for j, move in enumerate(moves):
        name = f"{j+1} {move}"
        current.add_child(name)
        current = current.children[name]
  return root


def visit(root, graph, edges):
  for k, v in root.children.items():
    draw(root.data, k, graph, edges)
    visit(v, graph, edges)


root = Node("start")
graph = pydot.Dot(graph_type="graph", size="160,90!")
edges = set()
populate(root)
visit(root, graph, edges)
graph.write_png("eco.png")
