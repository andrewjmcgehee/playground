from pprint import pprint

# adj matrix
# adj list
# edge list

# 10 nodes
# nodes 0 1 and 2 form a triangle
# node 3 parents 4, 4 parents 5, 5 parents 6 and 7, 7 parents 8 (a DAG)
# node 9 is vestigeal

adj_matrix = [[0 for _ in range(10)] for _ in range(10)]

adj_matrix[0][1] = 1
adj_matrix[1][2] = 1
adj_matrix[2][0] = 1
adj_matrix[3][4] = 1
adj_matrix[4][5] = 1
adj_matrix[5][6] = 1
adj_matrix[5][7] = 1
adj_matrix[7][8] = 1


def dfs(start, target, adj_matrix):
  visited = set()
  path = [start]
  stack = [(start, target, path)]
  while stack:
    current, target, path = stack.pop()
    if current == target:
      return path
    visited.add(current)
    for i, has_edge in enumerate(adj_matrix[current]):
      if has_edge and i not in visited:
        stack.append((i, target, path + [i]))
  return []


print(dfs(0, 2, adj_matrix))
print(dfs(1, 0, adj_matrix))
print(dfs(2, 1, adj_matrix))
print(dfs(3, 0, adj_matrix))
print(dfs(3, 5, adj_matrix))
print(dfs(3, 6, adj_matrix))
print(dfs(3, 7, adj_matrix))
print(dfs(3, 8, adj_matrix))
print()

adj_list = [
    [1],
    [2],
    [0],
    [4],
    [5],
    [6, 7],
    [],
    [8],
    [],
    [],
]


def dfs_adj_list(start, target, adj_list):
  visited = set()
  path = [start]
  stack = [(start, target, path)]
  while stack:
    current, target, path = stack.pop()
    if current == target:
      return path
    visited.add(current)
    for neighbor in adj_list[current]:
      if neighbor not in visited:
        stack.append((neighbor, target, path + [neighbor]))
  return []


print(dfs_adj_list(0, 2, adj_list))
print(dfs_adj_list(1, 0, adj_list))
print(dfs_adj_list(2, 1, adj_list))
print(dfs_adj_list(3, 0, adj_list))
print(dfs_adj_list(3, 5, adj_list))
print(dfs_adj_list(3, 6, adj_list))
print(dfs_adj_list(3, 7, adj_list))
print(dfs_adj_list(3, 8, adj_list))
print()

edge_list = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (5, 7), (7, 8)]


def dfs_edge_list(start, target, edges):
  visited = set()
  path = [start]
  stack = [(start, target, path)]
  while stack:
    current, target, path = stack.pop()
    if current == target:
      return path
    visited.add(current)
    for a, b in edges:
      if a != current:
        continue
      if b not in visited:
        stack.append((b, target, path + [b]))
  return []


print(dfs_edge_list(0, 2, edge_list))
print(dfs_edge_list(1, 0, edge_list))
print(dfs_edge_list(2, 1, edge_list))
print(dfs_edge_list(3, 0, edge_list))
print(dfs_edge_list(3, 5, edge_list))
print(dfs_edge_list(3, 6, edge_list))
print(dfs_edge_list(3, 7, edge_list))
print(dfs_edge_list(3, 8, edge_list))
print()
