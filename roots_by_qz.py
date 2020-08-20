import numpy as np
import scipy.linalg as la

def roots_by_qz( G, C):
  GG,CC,Q,Z = la.qz(G,C,output='complex')

  # Shouldn't need to do this, but I don't know if this is one or minus one
  dQ = la.det(Q)
  dZ = la.det(Z)
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

