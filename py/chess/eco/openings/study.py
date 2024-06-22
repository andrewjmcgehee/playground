import json
import os
from collections import defaultdict

import pydot

e4 = json.load(open("e4_opening_lines.json"))
d4 = json.load(open("d4_opening_lines.json"))


def get_fen(name):
  if name is None:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - c0 -; c1 *;"
  moves = name.split("_")
  fen = []
  i = 0
  move = 1
  while i < len(moves):
    fen.append(f"{move}. {' '.join(moves[i:i+2])}")
    i += 2
    move += 1
  fen = " ".join(fen)
  if not fen:
    fen = f"1. {name} *"
  else:
    fen += " *"
  with open("tmp.txt", "w") as f:
    f.write(fen)
  fen = os.popen("pgn-extract --quiet -Wepd tmp.txt").read()
  return fen.strip().split("\n")[-1]


def get_node(fen, positions, moves, k, create=False):
  # if not create:
  #   print("PARENT: ", end="")
  # print(moves)
  if fen in positions:
    # print(f"found: {fen}")
    last_move = moves.split("_")[-1]
    # print(last_move)
    for p in positions[fen]:
      if p.get_label() == last_move:
        # print("last_move matched\n")
        return p
  if create:
    # print(f"creating: {moves}, {k}\n")
    return pydot.Node(moves, label=k)
  return None


def visit(graph, d, positions, edges, parent=None, parent_moves=""):
  for k, v in d.children.items():
    # print("*" * 40)
    if parent is not None:
      fen = get_fen(parent_moves)
      existing_parent = get_node(fen, positions, parent_moves, k)
      if existing_parent:
        parent = existing_parent
      parent_edge = f"{fen}_{k}"
      a_moves = f"{parent_moves}_{k}"
      fen = get_fen(a_moves)
      a = get_node(fen, positions, a_moves, k, create=True)
      graph.add_node(a)
      positions[fen].append(a)
      parent_move = parent_moves.split("_")[-1]
      edge = f"{fen}_{parent_move}_{k}"
      if edge not in edges:
        graph.add_edge(pydot.Edge(parent, a))
        edges.add(edge)
      visit(graph, v, positions, edges, a, a_moves)
    else:
      fen = get_fen(k)
      a = pydot.Node(k, label=k)
      graph.add_node(a)
      positions[fen].append(a)
      visit(graph, v, positions, edges, a, k)


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


graphs = dict()
for tag, *line in e4:
  if tag not in graphs:
    graphs[tag] = Node(tag)
  current = graphs[tag]
  for move in line:
    current.add_child(move)
    current = current.children[move]

# for tag in graphs:
#   positions = defaultdict(list)
#   edges = set()
#   graph = pydot.Dot(tag, graph_type="digraph")
#   print(f"visiting {tag}...")
#   visit(graph, graphs[tag], positions, edges)
#   graph.write_png(f"./images/e4/{tag}.png")

# graphs = dict()
for tag, *line in d4:
  if tag not in graphs:
    graphs[tag] = Node(tag)
  current = graphs[tag]
  for move in line:
    current.add_child(move)
    current = current.children[move]

# for tag in graphs:
#   positions = defaultdict(list)
#   edges = set()
#   graph = pydot.Dot(tag, graph_type="digraph")
#   print(f"visiting {tag}...")
#   visit(graph, graphs[tag], positions, edges)
#   graph.write_png(f"./images/d4/{tag}.png")

positions = defaultdict(list)
edges = set()
graph = pydot.Dot(tag, graph_type="digraph")
for tag in graphs:
  print(f"visiting {tag}...")
  visit(graph, graphs[tag], positions, edges)
graph.write_png("./images/all.png")
