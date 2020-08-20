from i import reduce_poly

def test_A():
    assert reduce_poly( [1, 0, 1]) == [1,0,1]
    assert reduce_poly( [0, 0, 1]) == [0,0,1]
    assert reduce_poly( [0, 1, 0]) == [0,1]
    assert reduce_poly( [1, 0, 0]) == [1]
