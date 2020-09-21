from foster import *

def test_foster_A():
    Z = ReactanceFunction( Poly( 1.0, [QuadraticTerm( 1), QuadraticTerm( 3)]),
                           Poly( 1.0, [LinearTerm(), QuadraticTerm( 2), QuadraticTerm( 4)]))
    
    assert Z.has_zero_at_infinity()
    assert Z.eval( 1j) == 0
    assert Z.eval( 3j) == 0
    compare( Z.foster(), Z)

    Y = Z.reciprocal()
    assert Y.has_pole_at_infinity()
    assert Y.eval( 0j) == 0
    assert Y.eval( 2j) == 0
    assert Y.eval( 4j) == 0
    compare( Y.foster(), Y)

def test_foster_B():
    Z = ReactanceFunction( Poly( 1.0, [LinearTerm(), QuadraticTerm( 2), QuadraticTerm( 4)]),
                           Poly( 1.0, [QuadraticTerm( 1), QuadraticTerm( 3)]))
    
    assert Z.has_pole_at_infinity()
    assert Z.eval( 0j) == 0
    assert Z.eval( 2j) == 0
    assert Z.eval( 4j) == 0
    compare( Z.foster(), Z)

    Y = Z.reciprocal()
    assert Y.has_zero_at_infinity()
    assert Y.eval( 1j) == 0
    assert Y.eval( 3j) == 0
    compare( Y.foster(), Y)

def test_foster_C():
    Z = ReactanceFunction( Poly( 1.0, [QuadraticTerm( 1), QuadraticTerm( 3), QuadraticTerm(5)]),
                           Poly( 1.0, [LinearTerm(), QuadraticTerm( 2), QuadraticTerm( 4)]))
    
    assert Z.has_pole_at_infinity()
    assert Z.eval( 1j) == 0
    assert Z.eval( 3j) == 0
    assert Z.eval( 5j) == 0
    compare( Z.foster(), Z)

    Y = Z.reciprocal()
    assert Y.has_zero_at_infinity()
    assert Y.eval( 0j) == 0
    assert Y.eval( 2j) == 0
    assert Y.eval( 4j) == 0
    compare( Y.foster(), Y)

def test_foster_D():
    Z = ReactanceFunction( Poly( 1.0, [LinearTerm(), QuadraticTerm( 2), QuadraticTerm( 4)]),
                           Poly( 1.0, [QuadraticTerm( 1), QuadraticTerm( 3), QuadraticTerm(5)]))
    
    assert Z.has_zero_at_infinity()
    assert Z.eval( 0j) == 0
    assert Z.eval( 2j) == 0
    assert Z.eval( 4j) == 0
    compare( Z.foster(), Z)

    Y = Z.reciprocal()
    assert Y.has_pole_at_infinity()
    assert Y.eval( 1j) == 0
    assert Y.eval( 3j) == 0
    assert Y.eval( 5j) == 0
    compare( Y.foster(), Y)


    
