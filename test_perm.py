
from perm import toCycles, cycle_parity
import itertools

def testA():
    a = [1,0]
    result = toCycles( a)
    assert len(result) == 1
    print( result)
    assert cycle_parity( a) == -1

def testB():
    a = [0,1]
    result = toCycles( a)
    assert len(result) == 2
    print( result)
    assert cycle_parity( a) == 1    

def testC():
    a = [0,2,1]
    result = toCycles( a)
    assert len(result) == 2
    print( result)
    assert cycle_parity( a) == -1

def testD():
    odds, evens = 0, 0
    for p in itertools.permutations( range(6)):
        a = list(p)
        result = toCycles( a)
        parity = cycle_parity( a)
        print( a, result, cycle_parity( a))
        if parity == 1:
            evens += 1
        elif parity == -1:
            odds += 1
        else:
            assert False
    assert evens == odds
