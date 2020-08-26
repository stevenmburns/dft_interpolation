import numpy as np

from interpolate import run, second_order_system

def test_ex1():
  G = np.array( [[1,0],
                 [0,2]])

  C = np.array( [[3,-3],
                 [-3,3]])

  W = np.array( [1,0])

  d = np.array( [0,1])
  run( G, C, W, d, 2)

def test_ex2():
  G = np.array( [[2,-1],
                 [-1,2]])

  C = np.array( [[1,-1],
                 [-1,1]])

  W = np.array( [1,0])

  d = np.array( [0,1])
  run( G, C, W, d, 2)


def test_ex3():
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
  run( G, C, W, d, 3)


def test_ex4():
  run( *second_order_system())

def test_ex5():
  "Parallel RLC"
  fS = 1.0

  g = 1
  C = 1*fS
  L = 1*fS

  G = np.array( [[g, 1],
                 [1, 0]])

  C = np.array( [[C,0],
                 [0,-L]])

  W = np.array( [1,0])

  d = np.array( [1,0])
  run( G, C, W, d, 3)
