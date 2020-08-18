
from i import toCycles, cycle_parity

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
