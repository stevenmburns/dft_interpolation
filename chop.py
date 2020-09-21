from sympy import symbols, Rational
from sympy.simplify.simplify import simplify
from sympy.simplify.ratsimp import ratsimp
from sympy import cancel
from sympy.simplify.simplify import radsimp
import scipy.optimize
import sympy

from sympy.polys.polytools import div as polydiv, rem as polyrem

def chop_1_over_s( P, Q, s):
    print( P.subs( { s: 0}))
    print( Q.subs( { s: 0}))

    factor = s

    Q_prime, Q_prime_rem = polydiv( Q.as_poly(s,domain='RR'), factor)
    print( f"Q_prime: {Q_prime}")
    print( f"Q_prime rem: {Q_prime_rem}")

    A = sympy.limit( P/Q_prime, s, 0)
    print(f"A: {A}")

    P_new = P - A*Q_prime
    print( f"P_new: {P_new}")
    P_prime, P_prime_rem = polydiv( P_new.as_poly(s,domain='RR'), factor)
    print( f"P_prime: {P_prime}")
    print( f"P_prime rem: {P_prime_rem}")

    return P_prime, Q_prime

def remove_leading_coeff( P):
    coeff = sympy.LC(P)
    if np.abs(coeff) < 1.e-7:
        print(f"P (before removing leading term): {P} {coeff}")
        P = P-coeff*sympy.LM(P)
        print(f"P (after removing leading term): {P}")

    return P
    

def chop_s( P, Q, s):
    degreeP = sympy.degree(P)
    print(f"P: {P}")
    print(f"Q: {Q}")

    B = sympy.limit( P/(s*Q), s, sympy.oo)
    print(f"B: {B}")

    P_prime = P - B*s*Q
    print( f"degrees: {degreeP} {sympy.degree(P_prime)}")

    if degreeP == sympy.degree(P_prime):
        P_prime = remove_leading_coeff( P_prime)

    return P_prime, Q

def chop_linear( P, Q, s, s0):
    print( P.subs( { s: s0}))
    print( Q.subs( { s: s0}))

    A = sympy.limit( (s-s0)*P/Q, s, s0)
    print(f"A: {A}")

    factor = s - s0

    Q_prime, Q_prime_rem = polydiv( Q.as_poly(s,domain='RR'), factor)
    print( f"Q_prime: {Q_prime}")
    print( f"Q_prime rem: {Q_prime_rem}")

    P_new = P - A*Q_prime
    print( f"P_new: {P_new}")
    P_prime, P_prime_rem = polydiv( P_new.as_poly(s,domain='RR'), factor)
    print( f"P_prime: {P_prime}")
    print( f"P_prime rem: {P_prime_rem}")

    return P_prime, Q_prime

def quadratic_residue( P, Q, s, w0):
    factor = s**2 + w0**2

    Q_prime, Q_prime_rem = polydiv( Q.as_poly(s,domain='RR'), factor)
    print( f"Q_prime: {Q_prime}")
    print( f"Q_prime rem: {Q_prime_rem}")

    Ap = sympy.limit( P, s, sympy.I*w0)
    Aq = sympy.limit( (s-sympy.I*w0)/Q, s, sympy.I*w0)
    Aqq = sympy.limit( 1/(Q_prime*(s+sympy.I*w0)), s, sympy.I*w0)

    A = Ap*Aqq
    print(f"Ap, Aq, Aqq, A: {Ap} {Aq} {Aqq} {A}")
    Ahat = 2*(s*sympy.re(A)-w0*sympy.im(A))
    print(f"Ahat: {Ahat}")

    return Ahat, Q_prime, factor


def chop_quadratic( P, Q, s, w0):
    print( P.subs( { s: sympy.I*w0}))
    print( P.subs( { s: -sympy.I*w0}))
    print( Q.subs( { s: sympy.I*w0}))
    print( Q.subs( { s: -sympy.I*w0}))

    Ahat, Q_prime, factor = quadratic_residue( P, Q, s, w0)

    P_new = P - Ahat*Q_prime
    print( f"P_new: {P_new}")
    P_prime, P_prime_rem = polydiv( P_new.as_poly(s,domain='RR'), factor)
    print( f"P_prime: {P_prime}")
    print( f"P_prime rem: {P_prime_rem}")

    return P_prime, Q_prime

def rp( Z, s):
    w = symbols( "w")
    real_part = sympy.re(Z.subs({s:sympy.I*w}))
    return sympy.lambdify( w, real_part, "numpy")

def remove_r( P, Q, s, bracket=None, at_infinity=False, at_zero=False):
    Z = P/Q

    f = rp(Z, s)

    if True:
        plot( f)

    if at_infinity:
        degreeP = sympy.degree(P)
        R = sympy.limit( Z, s, sympy.oo)
        w0 = sympy.oo
        Z = Z-R
        P, Q = Z.as_numer_denom()
        if degreeP == sympy.degree(P):
            P = remove_leading_coeff(P)
        return P, Q, w0, R

    elif at_zero:
        R = sympy.limit( Z, s, 0)
        w0 = 0
    else:
        result = scipy.optimize.minimize_scalar( f, method="brent", bracket=bracket)
        assert result.success
        R = result.fun
        w0 = result.x

    Z = Z-R

    P, Q = Z.as_numer_denom()
    return P, Q, w0, R
    

def bott_duffin( P, Q, s, w0):
    X = (P/Q).subs({s : sympy.I*w0})

    print( f"X: {X.evalf()}")

    target0 = X/(sympy.I*w0)
    print( f"target: {target0.evalf()}")

    target0 = sympy.re(target0).evalf()

    assert target0 > 0

    k = symbols('k')

    k0_result = scipy.optimize.root_scalar( sympy.lambdify( k, (P/Q).subs({s:k}) - k*target0, "numpy"), method="brentq", bracket=[0,1000])

    assert k0_result.converged
    k0 = k0_result.root

    print( f"k0: {k0}")

    Z_k0 = (P/Q).subs({s:k0})
    print( f"Z_k0: {Z_k0.evalf()}")

    num = k0*P - s*Z_k0*Q
    den = Q*k0*Z_k0 - s*P

    eta_num,eta_den = num.as_poly(s,domain='RR'),den.as_poly(s,domain='RR')
    factor = sympy.poly( s-k0)

    eta_num, eta_num_rem = polydiv( eta_num, factor)
    eta_den, eta_den_rem = polydiv( eta_den, factor)

    print( f"eta_num: {eta_num}")
    print( f"eta_den: {eta_den}")

    print( f"eta_num rem: {eta_num_rem}")
    print( f"eta_den rem: {eta_den_rem}")

    if False:
        num_roots = eta_num.nroots()
        den_roots = eta_den.nroots()
        print( f"roots for eta_num: {num_roots}")
        print( f"roots for eta_den: {den_roots}")

        fuzz = 0.000001
        def find_pure_imaginary( roots):
            result = []
            for r in roots:
                f = sympy.re(r)
                if np.abs(f) < fuzz:
                    result.append(sympy.im(r)*sympy.I)
            return result

        imag_num_roots = find_pure_imaginary(num_roots)
        imag_den_roots = find_pure_imaginary(den_roots)

        print( f"imaginary numer roots: {imag_num_roots}")
        print( f"imaginary denom roots: {imag_den_roots}")

        assert len(imag_num_roots) % 2 == 0
        assert len(imag_den_roots) % 2 == 0

        assert len(imag_num_roots) == 2

        assert np.isclose( np.array(imag_num_roots).astype(np.complex128)[0], 1j*w0) or np.isclose( np.array(imag_num_roots).astype(np.complex128)[1], 1j*w0)

    P,Q = eta_num, eta_den

    return P, Q
