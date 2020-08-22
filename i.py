import numpy as np
from scipy.sparse import csc_matrix, linalg as sla
from perm import cycle_parity
from scipy.fft import fft

import numpy.linalg as la

from roots_by_qz import numerator, denominator

def gen_points( n):
  return [ np.exp( np.pi*2*k/n*1j) for k in range(n)]

def evaluate_point( T, W, d):
  Q = sla.splu( T)
  det = np.prod( Q.U.diagonal()) * cycle_parity( Q.perm_r) * cycle_parity( Q.perm_c)

  if np.abs(det) >= 1e-6:
    X = Q.solve( W)
    WW = T.dot(X)

    assert np.allclose( WW, W)

    F = d.T.dot(X)
    D = det
    N = F*D
    return (D,N)
  else:
    return (None,None)

def interpolate( GG, CC, W, d, n, fS=1.0):
  def poly_scale( a, fS):
    s = np.array( [ fS**i for i in range(len(a))])
    return a*s

  def evaluate_points( points, fS):
    Ds = []
    Ns = []
    for s in gen_points( n):
      T = GG + CC*s/fS
      (D,N) = evaluate_point( T, W, d)
      Ds.append( D)
      Ns.append( N)
    return Ds, Ns

  Ds, Ns = evaluate_points( gen_points( n), fS)

  new_fS = fS
  if any( D is None for D in Ds):
    new_fS *= 1.1
    print( f"TF evalated a pole during interpolation {Ds}. Rescaling {new_fS}...")
    Ds, Ns = evaluate_points( gen_points( n), new_fS)    

  numerator_coeffs = poly_scale( fft(Ns) / n, new_fS)
  denominator_coeffs = poly_scale( fft(Ds) / n, new_fS)
  return [ x.real for x in numerator_coeffs], [ x.real for x in denominator_coeffs]

def companion_matrix( a):
  assert a[-1] == 1
  n = len(a)-1

  G = np.block( [[np.zeros( (n-1,1)), np.eye(n-1)],
                 [-np.array(a[:-1]).reshape((1,n))]])

  C = -np.eye( n)

  W = np.array( [1] + [0]*(n-1))
  d = np.array( [0]*(n-1) + [1])
  return G,C,W,d

def second_order_system( wo=1, Q=10):
  a = [wo**2, wo/Q, 1]
  G, C, W, d = companion_matrix( a)
  return G, C, W, d, 3

def reduce_poly( c):
  first_non_zero = None
  for idx in range( len(c)-1, -1, -1):
    if np.abs(c[idx]) > 1.e-6:
      first_non_zero = idx
      break
  if first_non_zero is not None:
    return c[:first_non_zero+1]
  else:
    return c

def run( G, C, W, d, n=2, fS=1.0):
  GG = csc_matrix( G)
  CC = csc_matrix( C)
  run_sparse( GG, CC, W, d, n, fS)

from plot_transfer_function import plot_transfer_function

def run_sparse( GG, CC, W, d, n=2, fS=1.0):
  print( "="*40)
  print( f"run_sparse: n={n} fS={fS}")

  numerator_coeffs, denominator_coeffs = interpolate( GG, CC, W, d, n, fS)

  def f( x):
    s = np.format_float_positional( x, 5, trim='-')
    if s[-1] == '.':
      s = s[:-1]
    if s == "-0":
      s = "0"
    return s

  def cf( z):
    if type(z) is complex or type(z) is np.complex128:
      sRe = f( z.real)
      sIm = f( z.imag)
      if sIm == "0":
        return sRe
      elif sRe == "0":
        return f"{sIm}j"
      elif z.imag < 0:
        return f"{sRe}{sIm}j"
      else:
        return f"{sRe}+{sIm}j"
    else:
      return f(z)

  def p( a):
    return '(' + (' '.join( cf(z) for z in a)) + ')'
  
  def p_factored( a):
    return ''.join( f'(s-({cf(-z)}))' for z in a)

  def convert_to_poly( k, a):
    c = np.array( [k])
    # (s - p)*c
    for z in a:
      c = z*np.block( [c, np.zeros( (1,))])  + np.block( [ np.zeros( (1,)), c])
    return c

  numerator_coeffs = reduce_poly(numerator_coeffs)
  denominator_coeffs = reduce_poly(denominator_coeffs)
  print( f"DFT Interpolated: {p(numerator_coeffs)} {p(denominator_coeffs)}")
  plot_transfer_function( 1, numerator_coeffs, 1, denominator_coeffs, xunits='Hz')

  n_roots = np.roots( numerator_coeffs[::-1])
  d_roots = np.roots( denominator_coeffs[::-1])

  def pc( k, negative_roots):
    print( cf(k), p_factored(negative_roots))

  pc( numerator_coeffs[-1], (-x for x in n_roots))
  pc( denominator_coeffs[-1], ( -x for x in d_roots))

  print( "="*40)

  n_k, n_a = numerator( GG.A, CC.A, W, d)
  pc( n_k, n_a)
  d_k, d_a = denominator( GG.A, CC.A, W, d)
  pc( d_k, d_a)

  print( f"QZ Regenerated: {p(convert_to_poly( n_k, n_a))} {p(convert_to_poly( d_k, d_a))}")
