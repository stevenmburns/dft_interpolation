from sympy import symbols, Rational
from sympy.simplify.simplify import simplify
from sympy.simplify.ratsimp import ratsimp
from sympy import cancel
from sympy.simplify.simplify import radsimp

from cascade import Cascade
from chop import *

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

import numpy as np
import sympy
from cascade import plot, plot_real_part

def test_F():
    "Hazony example 5.2.2"
    s = symbols('s')

    print( "test_F")
    Z = (s**2 + s + 1)/(s**2 + 2*s + 2)
    print( f"Z: {Z}")

    min_r = (3-sympy.sqrt(2))/4

    Z1 = ratsimp(Z-min_r)
    print( f"Z1: {Z1}")

    #plot_real_part( sympy.lambdify(s, Z1, "numpy"))


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

from sympy import S, pprint

def test_G():
    "Hazony example 5.2.2"
    s, k = symbols('s k')

    pprint( "test_G")
    Z = (s**2 + s + 1)/(s**2 + s + 4)
    pprint( f"Z: {Z}")

    #plot_real_part( sympy.lambdify(s, Z, "numpy"))

    w0 = sympy.sqrt(2)

    target = radsimp(Z.subs({s : sympy.I*w0})/(sympy.I*w0))
    print( f"target: {target}")

    eq = sympy.Eq( Z.subs({s:k})/k, target)
    
    roots = sympy.solveset( eq, k)
    if True:
        for k0 in roots:
            Z_k0 = Z.subs({s:k0})
            eta = cancel((k0*Z - s*Z_k0)/(k0*Z_k0 - s*Z))
            print(k0,Z_k0,eta)
            #plot_real_part( sympy.lambdify(s, eta, "numpy"))

    k0 = 1
    Z_k0 = Z.subs({s:k0})
    eta = cancel((k0*Z - s*Z_k0)/(k0*Z_k0 - s*Z))
    print(k0,Z_k0,eta)

    print("normal")
    Z0 = eta*Z_k0
    print( f"Z0: {Z0}")

    Y1 = cancel(1/Z0-4)
    print( f"Y1: {Y1}")
    C = Cascade.Shunt(4)

    Z2 = cancel(1/Y1-s/6-1/(3*s))
    print( f"Z2: {Z2}")
    C = C.hit(Cascade.Series(s/6))
    C = C.hit(Cascade.Series(1/(3*s)))

    eta_Z_k0 = cancel(C.terminate(0))
    print( f"eta_Z_k0: {eta_Z_k0}")
    assert sympy.Eq( cancel(eta_Z_k0 - Z0), 0)

    print("recip")
    Z0 = cancel(Z_k0/eta)
    print( f"Z0: {Z0}")

    Z1 = cancel(Z0 - 1)
    print( f"Z1: {Z1}")
    C = Cascade.Series(1)

    Y2 = cancel(1/Z1 - 2*s/3 - 4/(3*s))
    print( f"Y2: {Y2}")

    C = C.hit(Cascade.Shunt(2*s/3))
    C = C.hit(Cascade.Shunt(4/(3*s)))
    eta_over_Z_k0 = cancel(1/C.terminate_with_admittance(0))
    print( f"eta_over_Z_k0: {eta_over_Z_k0}")
    assert sympy.Eq( cancel(eta_over_Z_k0 - Z0), 0)

    def p(a,b):
        return 1/(1/a + 1/b)

    constructed_Z = cancel( p(eta_Z_k0, (k0*Z_k0)/s) + p(eta_over_Z_k0, (Z_k0*s)/k0))
    print( f"constructed_Z: {constructed_Z}")

    assert sympy.Eq( cancel(constructed_Z - Z), 0)

def test_H():
    "Hazony problem 5.3.a"
    s, k = symbols('s k')
    w = symbols('w', real=True)

    pprint( "test_H")
    Z = (s**3 + 4*s**2 + 5*s + 8)/(2*s**3 + 2*s**2 + 20*s + 9)
    pprint( f"Z: {Z}")

    #plot_real_part( sympy.lambdify(s, Z, "numpy"))

    real_part = cancel(sympy.re(Z.subs({s:sympy.I*w})))
    print( f"real_part: {real_part}")

    roots = sympy.solveset( real_part, w)
    print( f"roots for w: {roots}")
    #plot( sympy.lambdify(w, real_part, "numpy"))

    w0 = sympy.sqrt(6)

    target0 = radsimp(Z.subs({s : sympy.I*w0})/(sympy.I*w0))
    print( f"target: {target0}")

    target1 = radsimp(Z.subs({s : sympy.I*w0})*(sympy.I*w0))
    print( f"target: {target1}")

    assert target0 > 0
    eq = sympy.Eq( Z.subs({s:k})/k, target0)
    #eq = sympy.Eq( Z.subs({s:k})*k, target1)
    
    print( f"eq: {eq}")

    roots = sympy.solveset( eq, k)
    print( f"roots for k: {roots}")

    k0 = Rational(1,4) + sympy.sqrt(33)/4
    Z_k0 = Z.subs({s:k0})
    print(k0,Z_k0)
    print(k0.evalf(),Z_k0.evalf())

    return

    f = s**2 + 6

    den = cancel((k0*Z_k0 - s*Z)/f)
    print( f"den factored: {sympy.factor(den)}")

    num = cancel((k0*Z - s*Z_k0)/f)
    print( f"num factored: {sympy.factor(num).evalf()}")

    print(sympy.factor(cancel(den/num)))

    return

    eta = cancel(((k0*Z - s*Z_k0)/(k0*Z_k0 - s*Z)).evalf())
    print(k0,Z_k0,eta)


    print(k0,Z_k0,eta.evalf())



    print("normal")
    Z0 = eta*Z_k0
    print( f"Z0: {Z0}")

    Y1 = cancel(1/Z0-4)
    print( f"Y1: {Y1}")
    C = Cascade.Shunt(4)

    Z2 = cancel(1/Y1-s/6-1/(3*s))
    print( f"Z2: {Z2}")
    C = C.hit(Cascade.Series(s/6))
    C = C.hit(Cascade.Series(1/(3*s)))

    eta_Z_k0 = cancel(C.terminate(0))
    print( f"eta_Z_k0: {eta_Z_k0}")
    assert sympy.Eq( cancel(eta_Z_k0 - Z0), 0)

    print("recip")
    Z0 = cancel(Z_k0/eta)
    print( f"Z0: {Z0}")

    Z1 = cancel(Z0 - 1)
    print( f"Z1: {Z1}")
    C = Cascade.Series(1)

    Y2 = cancel(1/Z1 - 2*s/3 - 4/(3*s))
    print( f"Y2: {Y2}")

    C = C.hit(Cascade.Shunt(2*s/3))
    C = C.hit(Cascade.Shunt(4/(3*s)))
    eta_over_Z_k0 = cancel(1/C.terminate_with_admittance(0))
    print( f"eta_over_Z_k0: {eta_over_Z_k0}")
    assert sympy.Eq( cancel(eta_over_Z_k0 - Z0), 0)

    def p(a,b):
        return 1/(1/a + 1/b)

    constructed_Z = cancel( p(eta_Z_k0, (k0*Z_k0)/s) + p(eta_over_Z_k0, (Z_k0*s)/k0))
    print( f"constructed_Z: {constructed_Z}")

    assert sympy.Eq( cancel(constructed_Z - Z), 0)

def test_I():
    "Hazony problem 5.3.a"
    s, k = symbols('s k')
    w = symbols('w', real=True)

    pprint( "test_I")
    Z = (s**3 + 3*s**2 + s + 1)/(s**3 + s**2 + 3*s + 1)
    pprint( f"Z: {Z}")



    #plot_real_part( sympy.lambdify(s, Z, "numpy"))


    real_part = cancel(sympy.re(Z.subs({s:sympy.I*w})))
    print( f"real_part: {real_part}")



    roots = sympy.solveset( real_part, w)
    print( f"roots for w: {roots}")
    #plot( sympy.lambdify(w, real_part, "numpy"))

    w0 = 1

    target0 = radsimp(Z.subs({s : sympy.I*w0})/(sympy.I*w0))
    print( f"target: {target0}")

    target1 = radsimp(Z.subs({s : sympy.I*w0})*(sympy.I*w0))
    print( f"target: {target1}")

    assert target0 > 0
    eq = sympy.Eq( Z.subs({s:k})/k, target0)
    #assert target1 > 0
    #eq = sympy.Eq( Z.subs({s:k})*k, target1)

    roots = sympy.solveset( eq, k)
    print( f"roots for k: {roots}")

    k0 = Rational(1,1)
    Z_k0 = Z.subs({s:k0})
    print(k0,Z_k0)
    print(k0.evalf(),Z_k0.evalf())

    den = cancel((k0*Z_k0 - s*Z))
    print( f"den factored: {sympy.factor(den)}")

    num = cancel((k0*Z - s*Z_k0))
    print( f"num factored: {sympy.factor(num)}")

    eta = cancel(num/den)
    print(k0,Z_k0,eta)

    print("normal")
    Z0 = eta*Z_k0
    print( f"Z0: {Z0}")

    Y1 = ratsimp(1/Z0-1)
    print( f"Y1: {Y1}")
    C = Cascade.Shunt(1)

    Z2 = ratsimp(1/Y1 - s/2 - 1/(2*s))
    print( f"Z2: {Z2}")

    C = C.hit(Cascade.Series(s/2))
    C = C.hit(Cascade.Series(1/(2*s)))

    eta_Z_k0 = cancel(C.terminate(0))
    print( f"eta_Z_k0: {eta_Z_k0}")
    assert sympy.Eq( cancel(eta_Z_k0 - Z0), 0)

    print("recip")
    Z0 = cancel(Z_k0/eta)
    print( f"Z0: {Z0}")

    Z1 = ratsimp(Z0-1)
    print( f"Z1: {Z1}")
    C = Cascade.Series(1)

    Y2 = ratsimp(1/Z1 - s/2 - 1/(2*s))
    print( f"Y2: {Y2}")

    C = C.hit(Cascade.Shunt(s/2))
    C = C.hit(Cascade.Shunt(1/(2*s)))
    eta_over_Z_k0 = cancel(1/C.terminate_with_admittance(0))
    print( f"eta_over_Z_k0: {eta_over_Z_k0}")
    assert sympy.Eq( cancel(eta_over_Z_k0 - Z0), 0)

    def p(a,b):
        return a*b/(a+b)

    constructed_Z = cancel( p(eta_Z_k0, (k0*Z_k0)/s) + p(eta_over_Z_k0, (Z_k0*s)/k0))
    print( f"constructed_Z: {constructed_Z}")

    assert sympy.Eq( cancel(constructed_Z - Z), 0)

def test_J():
    "Second problem in Guillemin"
    s, k = symbols('s k')
    w = symbols('w', real=True)

    pprint( "test_I")
    Z = (s**2 + s + 8)/(s**2 + 2*s + 2)
    pprint( f"Z: {Z}")
    Y = 1/Z

    #plot_real_part( sympy.lambdify(s, Y, "numpy"))

    real_part = cancel(sympy.re(Y.subs({s:sympy.I*w})))
    print( f"real_part: {real_part}")

    roots = sympy.solveset( real_part, w)
    print( f"roots for w: {roots}")
    #plot( sympy.lambdify(w, real_part, "numpy"))

    w0 = 2

    target0 = radsimp(Y.subs({s : sympy.I*w0})/(sympy.I*w0))
    print( f"target: {target0.evalf()}")
    target0 = Rational(1,2)

    target1 = radsimp(Y.subs({s : sympy.I*w0})*(sympy.I*w0))
    print( f"target: {target1.evalf()}")
    target1 = Rational(2,1)

    assert target0 > 0
    eq = sympy.Eq( Y.subs({s:k})/k, target0)
    #assert target1 > 0
    #eq = sympy.Eq( Z.subs({s:k})*k, target1)

    roots = sympy.solveset( eq, k)
    print( f"roots for k: {roots}")

    k0 = Rational(1,1)
    Y_k0 = Y.subs({s:k0})
    print(k0,Y_k0)
    print(k0.evalf(),Y_k0.evalf())

    den = cancel((k0*Y_k0 - s*Y))
    print( f"den factored: {sympy.factor(den)}")

    num = cancel((k0*Y - s*Y_k0))
    print( f"num factored: {sympy.factor(num)}")

    eta = cancel(num/den)
    print(k0,Y_k0,eta)

    print("normal")
    Y0 = eta*Y_k0
    print( f"Y0: {Y0}")

    Z1 = ratsimp(1/Y0-4)
    print( f"Z1: {Z1}")
    C = Cascade.Series(4)

    Y2 = ratsimp(1/Z1)
    print( f"Y2: {Y2}")

    C = C.hit(Cascade.Shunt(s/10))
    C = C.hit(Cascade.Shunt(2/(5*s)))

    eta_Y_k0 = cancel(C.terminate_with_admittance(0))
    print( f"eta_Y_k0: {eta_Y_k0}")
    assert sympy.Eq( cancel(eta_Y_k0 - Y0), 0)

    print("recip")
    Y0 = ratsimp(Y_k0/eta)
    print( f"Y0: {Y0}")


    Y1 = ratsimp(Y0 - 1)
    print( f"Y1: {Y1}")
    C = Cascade.Shunt(1)

    Z2 = ratsimp(1/Y1 - 2*s/5 - 8/(5*s))
    print( f"Z2: {Z2}")

    C = C.hit(Cascade.Series(2*s/5))
    C = C.hit(Cascade.Series(8/(5*s)))
    eta_over_Y_k0 = cancel(1/C.terminate(0))

    print( f"eta_over_Y_k0: {eta_over_Y_k0}")
    assert sympy.Eq( cancel(eta_over_Y_k0 - Y0), 0)

    def p(a,b):
        return a*b/(a+b)

    constructed_Y = cancel( p(eta_Y_k0, (k0*Y_k0)/s) + p(eta_over_Y_k0, (Y_k0*s)/k0))
    print( f"constructed_Y: {constructed_Y}")

    assert sympy.Eq( cancel(constructed_Y - Y), 0)

def test_K():
    "Random problem"
    s = symbols('s')

    print( "test_K")
    P = ((s + 1)**2 + 1)*((s + 1)**2 + 9)
    Q = ((s + 1)**2 + 4)*((s + 1)**2 + 16)

    pprint( f"P: {P}")
    pprint( f"Q: {Q}")

    P, Q, w0, r = remove_r( P, Q, s)

    P,Q = bott_duffin( P, Q, s, w0)

    P,Q = chop_quadratic( Q, P, s, w0)
    print(f"new_P, new_Q: {P} {Q}")

    if False:
        plot( rp( P/Q, s))

    P,Q,w0,r = remove_r( P, Q, s, bracket=[3.5,5])
    print(f"w0: {w0} new_P, new_Q: {P} {Q}")

    if False:
        plot( rp( P/Q, s))

    if False:
        P = P.as_poly(s,domain='RR')
        Q = Q.as_poly(s,domain='RR')

        P_roots = P.nroots()
        print( f"roots for P: {P_roots}")

        Q_roots = Q.nroots()
        print( f"roots for Q: {Q_roots}")


    # Switch to make X positive
    P, Q = bott_duffin( Q, P, s, w0)
    print(f"new_P, new_Q: {P} {Q}")

    # Switch to put j-axis zeros in denominator
    P, Q = chop_quadratic( Q, P, s, w0)
    print(f"new_P, new_Q: {P} {Q}")

    R = (P/Q).evalf()
    print(f"Final R: {R}")

def test_L():
    "Random problem"
    s = symbols('s')

    print( "test_K")
    P = ((s + 1)**2 + 1)*((s + 2)**2 + 9)
    Q = ((s + 1)**2 + 4)*((s + 2)**2 + 16)

    pprint( f"P: {P}")
    pprint( f"Q: {Q}")

    P, Q, w0, r = remove_r( P, Q, s, bracket=[.5,2])

    P,Q = bott_duffin( P, Q, s, w0)

    P,Q = chop_quadratic( Q, P, s, w0)
    print(f"Chop #1: new_P, new_Q: {P} {Q}")

    if True:
        plot( rp( P/Q, s))



    P,Q,w0,r = remove_r( P, Q, s, at_infinity=True)
    print(f"w0: {w0} new_P, new_Q: {P} {Q}")
    if True:
        P = P.as_poly(s,domain='RR')
        Q = Q.as_poly(s,domain='RR')

        P_roots = P.nroots()
        print( f"roots for P: {P_roots}")

        Q_roots = Q.nroots()
        print( f"roots for Q: {Q_roots}")


    P,Q = chop_s( -Q, -P, s)
    P = P.as_poly(s,domain='RR')
    Q = Q.as_poly(s,domain='RR')

    print(f"new_P, new_Q: {P} {Q}")

    if True:
        plot( rp( P/Q, s))

    P,Q,w0,r = remove_r( P, Q, s, at_infinity=True)
    print(f"new_P, new_Q: {P} {Q}")

    P,Q = chop_s( Q, P, s)
    print(f"new_P, new_Q: {P} {Q}")

    print(f"Final R/G: {(P/Q).evalf()}")

def test_M():
    "Random problem"
    s = symbols('s')

    print( "test_K")
    P = ((s + 1)**2 + 1)*((s + 2)**2 + 9)
    Q = s*((s + 1)**2 + 4)*((s + 2)**2 + 16)

    P = P.as_poly(s,domain='RR')
    Q = Q.as_poly(s,domain='RR')
    if True:
        plot( rp( P/Q, s))


    print( f"P: {P}")
    print( f"Q: {Q}")

    P,Q = chop_1_over_s( P, Q, s)
    print( f"P: {P}")
    print( f"Q: {Q}")

    if True:
        plot( rp( P/Q, s))

    P,Q = chop_s( Q, P, s)
    print( f"P: {P}")
    print( f"Q: {Q}")

    if True:
        plot( rp( P/Q, s))

    return


    P, Q, w0, r = remove_r( P, Q, s, bracket=[.5,2])



    P,Q = bott_duffin( P, Q, s, w0)

    P,Q = chop_quadratic( Q, P, s, w0)
    print(f"Chop #1: new_P, new_Q: {P} {Q}")

    if True:
        plot( rp( P/Q, s))



    P,Q,w0,r = remove_r( P, Q, s, at_infinity=True)
    print(f"w0: {w0} new_P, new_Q: {P} {Q}")
    if True:
        P = P.as_poly(s,domain='RR')
        Q = Q.as_poly(s,domain='RR')

        P_roots = P.nroots()
        print( f"roots for P: {P_roots}")

        Q_roots = Q.nroots()
        print( f"roots for Q: {Q_roots}")


    P,Q = chop_s( -Q, -P, s)
    P = P.as_poly(s,domain='RR')
    Q = Q.as_poly(s,domain='RR')

    print(f"new_P, new_Q: {P} {Q}")

    if True:
        plot( rp( P/Q, s))

    P,Q,w0,r = remove_r( P, Q, s, at_infinity=True)
    print(f"new_P, new_Q: {P} {Q}")

    P,Q = chop_s( Q, P, s)
    print(f"new_P, new_Q: {P} {Q}")

    print(f"Final R/G: {(P/Q).evalf()}")

