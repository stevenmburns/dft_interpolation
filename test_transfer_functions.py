
import numpy as np
from modified_nodal import *
from interpolate import run_sparse

def test_A():
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 2, 0, 1))

    mn.add( ConductanceElement( 0, 1, 1))
    mn.add( ConductanceElement( 1, 2, 1))

    mn.semantic( 1)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 1)

def test_B():
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 2, 0, 1))

    mn.add( ResistanceElement( 0, 1, 1))
    mn.add( ResistanceElement( 1, 2, 1))

    mn.semantic( 1)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 1)

def test_C():
    "Page 226, Example 7.7.1"
    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 1, 1))
    mn.add( ConductanceElement( 1, 0, 1))
    mn.add( ConductanceElement( 1, 2, 1))
    mn.add( CapacitanceElement( 1, 2, 1))
    mn.add( ConductanceElement( 2, 0, 1))

    mn.semantic( 2)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_D():
    "Page 174, Example 6.1.1"
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 3, 0, 1))
    mn.add( ConductanceElement( 3, 1, 1))
    mn.add( ConductanceElement( 1, 2, 1))
    mn.add( CapacitanceElement( 1, 4, 1))
    mn.add( CapacitanceElement( 2, 0, 1))
    mn.add( VVTElement( 2, 0, 4, 0, 1))

    mn.semantic(4)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_D():
    "Third order Butterworth, Wikipedia"
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.add( InductanceElement( 1, 2, 3/2))
    mn.add( CapacitanceElement( 2, 0, 4/3))
    mn.add( InductanceElement( 2, 3, 1/2))
    mn.add( ConductanceElement( 3, 0, 1))

    mn.semantic(3)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 4)

def test_E():
    "Multiple feedback topology, Wikipedia"
    mn = ModifiedNodal()

    """"
Want wo to be ~1, and Q = 10"
    wo = 1/sqrt(R3*R4*C2*C5)
    Q = sqrt(R3*R4*C2*C5)/((R4+R3+R4/R1*R3)*C5)
    wo = 1/sqrt(0.1)
    Q = sqrt(0.1)/(3*.01)
      = 0.32/0.03 ~= 10
"""

    R1, C2, R3, R4, C5 = 1,10,1,1,.01
    #R1, C2, R3, R4, C5 = 1,1,1,1,1

    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.add( ConductanceElement( 1, 2, 1/R1))
    mn.add( CapacitanceElement( 2, 0, C2))
    mn.add( ConductanceElement( 2, 3, 1/R3))

    mn.add( ConductanceElement( 4, 2, 1/R4))
    mn.add( CapacitanceElement( 4, 3, C5))

    mn.add( OpAmpElement( 0, 3, 4, 0))

    mn.semantic(4)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_F():
    "Cauer filter"
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 1, 0, 1))

    mn.add( InductanceElement( 1, 2, 1.19244))
    mn.add( InductanceElement( 2, 3, 0.96649))
    mn.add( InductanceElement( 3, 4, 0.70095))
    mn.add( InductanceElement( 4, 5, 0.79165))
    mn.add( InductanceElement( 5, 6, 0.26559))

    mn.add( ConductanceElement( 6, 0, 1))

    mn.add( InductanceElement( 2, 7,  0.43702))
    mn.add( CapacitanceElement( 7, 0, 1.20086))

    mn.add( InductanceElement( 3, 8,  1.30606))
    mn.add( CapacitanceElement( 8, 0, 0.63117))

    mn.add( InductanceElement( 4, 9,  1.28066))
    mn.add( CapacitanceElement( 9, 0, 0.57651))

    mn.add( InductanceElement( 5, 10,  0.22423))
    mn.add( CapacitanceElement( 10, 0, 0.86114))

    mn.semantic(6)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 14)

def test_G():
    mn = ModifiedNodal()

    mn.add( VoltageSourceElement( 1, 0, 1))

    N = 20
    for i in range(1, N):
        mn.add( ConductanceElement( i, i+1, 1))
        mn.add( CapacitanceElement( i+1, 0, 1))
    
    mn.semantic(N)

    run_sparse( mn.G, mn.C, mn.W, mn.d, N)

def test_H():
    """Example 6.5.1, page 184"""

    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 1, 1))
    mn.add( ConductanceElement( 1, 0, 4))
    mn.add( CapacitanceElement( 1, 0, 1))
    mn.add( ConductanceElement( 1, 2, 1))
    mn.add( InductanceElement( 2, 0, 1))
    mn.add( VCTElement( 1, 0, 0, 2, 3))

    mn.semantic(2)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_bott_duffin():

    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 4, 1))

    mn.add( ConductanceElement( 4, 2, 4))
    mn.add( CapacitanceElement( 4, 2, 2))
    mn.add( InductanceElement( 4, 3, 1/6))
    mn.add( CapacitanceElement( 3, 2, 3))

    mn.add( InductanceElement( 2, 0, 1/2))
    mn.add( InductanceElement( 2, 1, 3/4))
    mn.add( CapacitanceElement( 2, 1, 2/3))
    mn.add( ConductanceElement( 1, 0, 1))

    mn.semantic(4)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 7, xunits='rads/sec')

def test_bott_duffin_gyrator():

    mn = ModifiedNodal()

    k = 1
    # Z(s) = (s**2 + s + 1)/(s**2 + s + 4)
    z_of_k = 1/2

    zscale = z_of_k

    mn.add( CurrentSourceElement( 0, 3, 1))

    mn.add( GyratorElement( 1, 0, 1, 2, g1=1/zscale, g2=1/zscale))

    mn.add( CapacitanceElement( 3, 1, 1/(zscale*k)))

    mn.add( CapacitanceElement( 3, 4, 3/(zscale*2)))
    mn.add( InductanceElement( 4, 2, zscale/3))
    mn.add( ConductanceElement( 3, 2, 2/zscale))

    mn.semantic(3)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 4, xunits='rads/sec')

def test_bott_duffin_gyrator_unit():

    mn = ModifiedNodal()

    k = 1

    mn.add( CurrentSourceElement( 0, 3, 1))

    mn.add( GyratorElement( 0, 1, 2, 1, g1=1, g2=1))

    mn.add( CapacitanceElement( 3, 1, 1/k))

    mn.add( ConductanceElement( 3, 2, 1))

    mn.semantic(3)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 2, xunits='rads/sec')

def test_function_29():

    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 3, 1))

    mn.add( ConductanceElement( 3, 0, 1))
    mn.add( InductanceElement( 3, 0, 5/6))

    mn.add( InductanceElement( 3, 2, 5/4))

    mn.add( ConductanceElement( 2, 1, 8/25))


    mn.add( ConductanceElement( 1, 0, 24/25))
    mn.add( CapacitanceElement( 1, 0, 16/25))

    mn.semantic(3)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 7)

def test_unit_r():

    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 2, 1))

    k = 3

    mn.add( ConductanceElement( 2, 1, 1))
    mn.add( InductanceElement( 2, 1, 1/k))

    mn.add( ConductanceElement( 1, 0, 1))
    mn.add( CapacitanceElement( 1, 0, 1/k))

    mn.semantic(2)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 7)

def test_brune_rc():

    mn = ModifiedNodal()
    
    L1 = 1/(1+np.sqrt(2))
    L2 = 2/(1+np.sqrt(2))
    M = -np.sqrt(2)/(1+np.sqrt(2))
    C = 2 + np.sqrt(2)
    R = 1

    mn.add( CurrentSourceElement( 0, 1, 1))

    mn.add( TransformerElement( 1, 2, 3, 2, l1=L1, l2=L2, m=M))    
    mn.add( CapacitanceElement( 2, 0, C))

    mn.add( ConductanceElement( 3, 0, 1/R))

    mn.semantic(1)
    run_sparse( mn.G, mn.C, mn.W, mn.d, 7)

def test_brune():

    """
G: [[ 0.  0.  0.  1.  0.]
    [ 0.  0.  0. -1. -1.]
    [ 0.  0.  4.  0.  1.]
    [ 1. -1.  0.  0.  0.]
    [ 0. -1.  1.  0.  0.]]
C: [[ 0.    0.    0.    0.    0.  ]
    [ 0.    1.    0.    0.    0.  ]
    [ 0.    0.    0.    0.    0.  ]
    [ 0.    0.    0.   -1.   -0.5 ]
    [ 0.    0.    0.   -0.5  -0.25]]
W: [1. 0. 0. 0. 0.]
d: [0. 0. 1. 0. 0.]
"""

    mn = ModifiedNodal()

    L1 = 1
    L2 = 1/4
    M = 1/2
    C = 1
    R = 1/4

    mn.add( CurrentSourceElement( 0, 1, 1))

    mn.add( TransformerElement( 1, 2, 3, 2, l1=L1, l2=L2, m=M))    
    mn.add( CapacitanceElement( 2, 0, C))

    mn.add( ConductanceElement( 3, 0, 1/R))

    mn.semantic(1)
    run_sparse( mn.G, mn.C, mn.W, mn.d, 7)

def test_brune_fig26():

    mn = ModifiedNodal()

    R1 = 0.453
    L1 = 0.547
    L2 = 0.047
    M = 0.16
    C = 10.66
    R = 0.047

    mn.add( CurrentSourceElement( 0, 1, 1))

    mn.add( ConductanceElement( 1, 4, 1/R1))

    mn.add( TransformerElement( 4, 2, 3, 2, l1=L1, l2=L2, m=M))    
    mn.add( CapacitanceElement( 2, 0, C))

    mn.add( ConductanceElement( 3, 0, 1/R))

    mn.semantic(1)
    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_brune_fig28():

    mn = ModifiedNodal()

    mn.add( CurrentSourceElement( 0, 1, 1))

    mn.add( ConductanceElement( 1, 3, 2))
    mn.add( InductanceElement( 1, 2, 1/2))
    mn.add( CapacitanceElement( 2, 3, 1))

    mn.add( InductanceElement( 3, 0, 1/2))
    mn.add( CapacitanceElement( 3, 4, 1))
    mn.add( ConductanceElement( 4, 0, 2))

    mn.semantic(1)
    run_sparse( mn.G, mn.C, mn.W, mn.d, 5)

