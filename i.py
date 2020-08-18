import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix, linalg as sla
import scipy.sparse.linalg as spla

from scipy.fft import ifft, fft
from itertools import accumulate

def gen_points( n):
  return [ np.exp( np.pi*2*k/n*1j) for k in range(n)]

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
  return [1,-1][sum( len(c)-1 for c in cs) % 2]

def interpolate( G, C, W, d, n):
  GG = csc_matrix( G)
  CC = csc_matrix( C)

  Ns = []
  Ds = []
  for s in gen_points( n):
    T = GG + CC*s
    Q = sla.splu( T)
    det = np.prod( Q.U.diagonal()) * cycle_parity( Q.perm_r) * cycle_parity( Q.perm_c)
    X = Q.solve( W)
    if not np.allclose( T.dot(X), W):
      print( f"All close fails: {T.dot(X)} {W}")
    F = d.T.dot(X)
    D = det
    Ds.append(D)  
    N = F*D
    Ns.append(N)

  numerator_coeffs = fft(Ns) / n
  denominator_coeffs = fft(Ds) / n
  return numerator_coeffs, denominator_coeffs

def roots_by_qz( G, C):
  GG,CC,Q,Z = la.qz(G,C,output='complex')

  # Shouldn't need to do this, but I don't know if this is one or minus one
  dQ = la.det(Q)
  dZ = la.det(Z)
  print( f"det of Q, R: {dQ} {dZ}")
  K = dQ * dZ

  roots = []
  for i in range( GG.shape[0]):
    gg = GG[i,i]
    cc = CC[i,i]
    if np.abs(cc) < 1e-10:
      K *= gg
    else:
      K *= cc
      roots.append( gg/cc)

  return K, roots

def numerator( G, C, W, d):
  n = G.shape[0]
  Ghat = np.block([ [G,                    W.reshape( (n,1))],
                    [-d.T.reshape( (1,n)), np.zeros( (1,1))]])
  Chat = np.block([ [C,                    np.zeros( (n,1))],
                    [np.zeros( (1,n)),     np.zeros( (1,1))]])
  return roots_by_qz( Ghat, Chat)

def denominator( G, C, W, d):
  n = G.shape[0]
  Ghat = np.block([ [G,                    np.zeros( (n,1))],
                    [-d.T.reshape( (1,n)), np.ones( (1,1))]])
  Chat = np.block([ [C,                    np.zeros( (n,1))],
                    [np.zeros( (1,n)),     np.zeros( (1,1))]])
  return roots_by_qz( Ghat, Chat)

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
  return companion_matrix( a)

def run( G, C, W, d):
  print( interpolate( G, C, W, d, 2))
  print( interpolate( G, C, W, d, 3))
  print( interpolate( G, C, W, d, 4))

  #print( roots_by_qz( G, C))
  print( numerator( G, C, W, d))
  print( denominator( G, C, W, d))


def ex1():
  G = np.array( [[1,0],
                 [0,2]])

  C = np.array( [[3,-3],
                 [-3,3]])

  W = np.array( [1,0])

  d = np.array( [0,1])
  run( G, C, W, d)

def ex2():
  G = np.array( [[2,-1],
                 [-1,2]])

  C = np.array( [[1,-1],
                 [-1,1]])

  W = np.array( [1,0])

  d = np.array( [0,1])
  run( G, C, W, d)


def ex3():
  "Sallen-Key"
  "Result should be N=1, D=10s^2 + 2s + 1"
  "10 * (s^2 + 0.2s + 0.1)"
  "s = -0.1 +- 0.3j"
  G1 = 1
  G2 = 1
  C1 = 10
  C2 = 1
  K = 1
  G = np.array( [[G1+G2, -G2],
                 [-G2,    G2]])

  C = np.array( [[C1,-C1*K],
                 [0,  C2]])

  W = np.array( [G1,0])

  d = np.array( [0,K])
  run( G, C, W, d)


def ex4():
  run( *second_order_system())

if __name__ == "__main__":
  ex3()
