import numpy as np

def toCycles( a):
  done = set()

  def find_cycle( idx):
    incycle = [idx] 
    incycle_set = set([idx])
    while a[idx] not in incycle_set:
      idx = a[idx]
      incycle.append( idx)
      incycle_set = incycle_set.union([idx])
    return incycle, incycle_set

  result = []
  for i in range( len(a)):
    if i not in done:
      incycle, incycle_set = find_cycle( i)
      done = done.union( incycle_set)
      result.append( incycle)
  return result

def cycle_parity( a):
  cs = toCycles( a)
  #result = sum( len(c)-1 for c in cs)
  #result = len(a)%2 + len(cs)%2
  result = len(a) - len(cs)
  return 1-2*(result%2)
