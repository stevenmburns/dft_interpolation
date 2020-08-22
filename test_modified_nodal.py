
import numpy as np
from modified_nodal import ModifiedNodal
from interpolate import run_sparse

def test_A():
    mn = ModifiedNodal()

    mn.add_voltage_source( 2, 0, 1)

    mn.add_conductance( 0, 1, 1)
    mn.add_conductance( 1, 2, 1)

    mn.semantic( 1)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 1)

def test_B():
    mn = ModifiedNodal()

    mn.add_voltage_source( 2, 0, 1)

    mn.add_resistance( 0, 1, 1)
    mn.add_resistance( 1, 2, 1)

    mn.semantic( 1)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 1)

def test_C():
    "Page 226, Example 7.7.1"
    mn = ModifiedNodal()

    mn.add_current_source( 0, 1, 1)
    mn.add_conductance( 1, 0, 1)
    mn.add_conductance( 1, 2, 1)
    mn.add_capacitance( 1, 2, 1)
    mn.add_conductance( 2, 0, 1)

    mn.semantic( 2)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_D():
    "Page 174, Example 6.1.1"
    mn = ModifiedNodal()

    mn.add_voltage_source( 3, 0, 1)
    mn.add_conductance( 3, 1, 1)
    mn.add_conductance( 1, 2, 1)
    mn.add_capacitance( 1, 4, 1)
    mn.add_capacitance( 2, 0, 1)
    mn.add_vvt_source( 2, 0, 4, 0, 1)

    mn.semantic(4)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_D():
    "Third order Butterworth, Wikipedia"
    mn = ModifiedNodal()

    mn.add_voltage_source( 1, 0, 1)
    mn.add_inductance( 1, 2, 3/2)
    mn.add_capacitance( 2, 0, 4/3)
    mn.add_inductance( 2, 3, 1/2)
    mn.add_conductance( 3, 0, 1)

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

    mn.add_voltage_source( 1, 0, 1)
    mn.add_conductance( 1, 2, 1/R1)
    mn.add_capacitance( 2, 0, C2)
    mn.add_conductance( 2, 3, 1/R3)

    mn.add_conductance( 4, 2, 1/R4)
    mn.add_capacitance( 4, 3, C5)

    mn.add_op_amp( 0, 3, 4, 0)

    mn.semantic(4)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)

def test_F():
    "Cauer filter"
    mn = ModifiedNodal()

    mn.add_voltage_source( 1, 0, 1)

    mn.add_inductance( 1, 2, 1.19244)
    mn.add_inductance( 2, 3, 0.96649)
    mn.add_inductance( 3, 4, 0.70095)
    mn.add_inductance( 4, 5, 0.79165)
    mn.add_inductance( 5, 6, 0.26559)

    mn.add_conductance( 6, 0, 1)

    mn.add_inductance( 2, 7,  0.43702)
    mn.add_capacitance( 7, 0, 1.20086)

    mn.add_inductance( 3, 8,  1.30606)
    mn.add_capacitance( 8, 0, 0.63117)

    mn.add_inductance( 4, 9,  1.28066)
    mn.add_capacitance( 9, 0, 0.57651)

    mn.add_inductance( 5, 10,  0.22423)
    mn.add_capacitance( 10, 0, 0.86114)

    mn.semantic(6)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 14)

def test_G():
    mn = ModifiedNodal()

    mn.add_voltage_source( 1, 0, 1)

    N = 20
    for i in range(1, N):
        mn.add_conductance( i, i+1, 1)
        mn.add_capacitance( i+1, 0, 1)
    
    mn.semantic(N)

    run_sparse( mn.G, mn.C, mn.W, mn.d, N)

def test_H():
    """Example 6.5.1, page 184"""

    mn = ModifiedNodal()

    mn.add_current_source( 0, 1, 1)
    mn.add_conductance( 1, 0, 4)
    mn.add_capacitance( 1, 0, 1)
    mn.add_conductance( 1, 2, 1)
    mn.add_inductance( 2, 0, 1)
    mn.add_vct_source( 1, 0, 0, 2, 3)

    mn.semantic(2)

    run_sparse( mn.G, mn.C, mn.W, mn.d, 3)
