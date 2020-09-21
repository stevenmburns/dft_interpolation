from sympy import symbols, Rational
import numpy as np
from chop import *

def test_chop_chop():
    s = symbols('s')

    P = 2*s**2 + 2*s + 1
    Q = s*(s**2+s+1)

    P, Q = chop_1_over_s( P, Q, s)
    print(f"new_P, new_Q: {P} {Q}")

    P, Q = chop_s( Q, P, s)
    print(f"new_P, new_Q: {P} {Q}")

    P, Q = chop_s( Q, P, s)
    print(f"new_P, new_Q: {P} {Q}")

def test_chop_quadratic():
    s = symbols('s')

    P = s**2+2
    Q = 2*s**2+3*s+4

    w0 = np.sqrt(2)

    new_P, new_Q = chop_quadratic( Q, P, s, w0)
    print(f"new_P, new_Q: {new_P} {new_Q}")

def test_chop_linear():
    s = symbols('s')

    P = (s+1)
    Q = (s**2+1)*(s+3)

    s0 = -3

    new_P, new_Q = chop_linear( P, Q, s, s0)
    print(f"new_P, new_Q: {new_P} {new_Q}")


