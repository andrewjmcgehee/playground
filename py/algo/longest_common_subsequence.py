import pprint


def lcs(a, b):
  if not a or not b:
    return 0
  if len(a) == 1:
    return int(a in b)
  if len(b) == 1:
    return int(b in a)
  memo = [[0 for _ in range(len(b))] for _ in range(len(a))]
  # initial conditions
  for i in range(len(a)):
    if b[0] == a[i]:
      memo[i][0] = 1
    if i != 0 and memo[i - 1][0] == 1:
      memo[i][0] = 1
  for j in range(len(b)):
    if a[0] == b[j]:
      memo[0][j] = 1
    if j != 0 and memo[0][j - 1] == 1:
      memo[0][j] = 1
  for i in range(1, len(a)):
    for j in range(1, len(b)):
      memo[i][j] = max(memo[i - 1][j], memo[i][j - 1] + int(a[i] == b[j]))
  return memo


def display(a, b, memo):
  print(f'  {" ".join(list(b))}')
  for i in range(len(a)):
    print(f'{a[i]} {" ".join([str(x) for x in memo[i]])}')


# example
# a = abcdefg
# b = bcefh
# result = 4, bcef

#   b c e f h
# a 0 0 0 0 0
# b 1 1 1 1 1
# c 1 2 2 2 2
# d 1 2 2 2 2
# e 1 2 3 3 3
# f 1 2 3 4 4
# g 1 2 3 4 4

a = "andrew"
b = "mountain dew"
memo = lcs(a, b)
display(a, b, memo)
