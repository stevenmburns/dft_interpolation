from sympy import symbols, Rational
from sympy.simplify.simplify import simplify
from sympy.simplify.ratsimp import ratsimp

from cascade import Cascade

def test_A():
    r0, r1, r = 100, 200, 1

    R0 = Cascade.Series(r0)
    R1 = Cascade.Series(r1)
    assert 301 == R0.hit(R1).terminate(r)

def test_B():
    s, L, C, r = symbols('s L C r')

    # series L, shunt C

    C0 = Cascade.Series(L*s)
    C1 = Cascade.Shunt(C*s)
    print( simplify(C0.hit(C1).terminate(r)))

def test_C():
    s = symbols('s')
    C = Rational(3,2)
    L = Rational(1,3)
    r = Rational(1,2)                 

    # series L, shunt C

    C0 = Cascade.Shunt( 1/( 1/(C*s) + L*s))
    print( ratsimp(C0.terminate(r)))
                        
    X = ratsimp(1/C0.terminate(r))
    print(X)

    Y = ratsimp(1/(X-2))
    print(Y)

    Z = ratsimp(Y-s/3)
    print(Z)


def test_D():
    s = symbols('s')
    C = Rational(3,2)
    L = Rational(1,3)
    r = Rational(1,2)                 

    # series L, shunt C

    C0 = Cascade.Series( L*s)
    C1 = Cascade.Series( 1/(C*s))
    C2 = Cascade.Shunt( 1/r)

    print( "test_D")

    print( C0)
    print( C1)
    print( C2)

    print( C1.hit(C0))
    print( C2.hit(C1.hit(C0)))

    Y = ratsimp(1/C2.hit(C1.hit(C0)).terminate(0))
                        
    print(Y)

    Z = ratsimp(1/(Y-2))
    print(Z)

    Y = ratsimp(Z-s/3)
    print(Y)

def test_E():
    "Chop Chop example"
    s = symbols('s')

    print( "test_E")
    Z = (2*s**2 + 2*s + 1)/(s*(s**2 + s + 1))
    print( f"Z: {Z}")

    Z = ratsimp(Z)
    print( f"Z: {Z}")

    Z = ratsimp(Z-1/s)
    print( f"Z-1/s: {Z}")

    C = Cascade.Series( 1/s)

    Y = ratsimp(1/Z-s)
    print( f"Y: {Y}")
    C = C.hit(Cascade.Shunt( s))

    Z = ratsimp(1/Y)
    print( f"Z: {Z-1-s}")
    C = C.hit(Cascade.Series( 1+s)).terminate(0)
    print(ratsimp(C))

import sympy

def test_F():
    "Hazony example 5.2.2"
    s = symbols('s')

    print( "test_F")
    Z = (s**2 + s + 1)/(s**2 + 2*s + 2)
    print( f"Z: {Z}")

    min_r = (3-sympy.sqrt(2))/4

    Z1 = ratsimp(Z-min_r)
    print( f"Z1: {Z1}")

    Y1 = ratsimp(1/Z-1)
    print( f"Y1: {Y1}")
    C = Cascade.Shunt( 1)

    Z2 = ratsimp(1/Y1-s)
    print( f"Z2: {Z2}")
    C = C.hit(Cascade.Series( s))

    Y3 = ratsimp(1/Z2-s-1)
    print( f"Y3: {Y3}")

    Ytotal = C.hit(Cascade.Shunt( 1).hit(Cascade.Shunt(s))).terminate_with_admittance(0)

    assert sympy.Eq( 0, ratsimp(1/Ytotal-Z))
    assert sympy.Eq( 1, ratsimp(Ytotal*Z))

    Ytotal = C.hit(Cascade.Shunt( s)).terminate_with_admittance(1)

    assert sympy.Eq( 0, ratsimp(1/Ytotal-Z))
    assert sympy.Eq( 1, ratsimp(Ytotal*Z))

    Ytotal = C.terminate_with_admittance(1+s)

    assert sympy.Eq( 0, ratsimp(1/Ytotal-Z))
    assert sympy.Eq( 1, ratsimp(Ytotal*Z))

