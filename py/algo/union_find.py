class UF:

  def __init__(self, n):
    self.groups = list(range(n))

  def union(self, a, b):
    if self.groups[self.find(a)] == self.groups[self.find(b)]:
      return
    self.groups[a] = self.groups[b]

  def find(self, a):
    if self.groups[a] == a:
      return self.groups[a]
    self.groups[a] = self.find(self.groups[a])
    return self.groups[a]


# example
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

# 2 -> 0
# 4 -> 2
# 5 -> 7
# 7 -> 4
# 8 -> 9
#

# iters
# 0, 1, 0, 3, 4, 5, 6, 7, 8, 9
# 0, 1, 0, 3, 0, 5, 6, 7, 8, 9
# 0, 1, 0, 3, 0, 7, 6, 7, 8, 9
# 0, 1, 0, 3, 0, 7, 6, 0, 8, 9
# 0, 1, 0, 3, 0, 7, 6, 0, 9, 9

uf = UF(10)
print(uf.groups)
uf.union(2, 0)
print('union 2 0')
print(uf.groups)
uf.union(4, 2)
print('union 4 2')
print(uf.groups)
uf.union(5, 7)
print('union 5 7')
print(uf.groups)
uf.union(7, 4)
print('union 7 4')
print(uf.groups)
uf.union(8, 9)
print('union 8 9')
print(uf.groups)
