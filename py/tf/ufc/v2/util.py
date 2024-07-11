class nameddict(dict):
  __getattr__ = dict.__getitem__


def nmap(func, *other, iterable=None):
  if iterable is None:
    *other, iterable = other
  if other:
    return nmap(*other, map(func, iterable))
  return map(func, iterable)