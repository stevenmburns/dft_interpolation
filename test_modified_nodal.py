
import numpy as np
from modified_nodal import ModifiedNodal
from i import run_sparse

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

    d = np.array( [1,0,0,0,0])

    run_sparse( mn.G, mn.C, mn.W, mn.d, 1)
