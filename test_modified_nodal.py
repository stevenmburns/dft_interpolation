
import numpy as np
from modified_nodal import *

def test_Conductance():
    mn = ModifiedNodal()
    mn.add( ConductanceElement( 1, 0, 2))
    mn.semantic( 1)
    mn.W[0] = 1
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 0.5)

def test_Resistance():
    mn = ModifiedNodal()
    mn.add( ResistanceElement( 1, 0, 2))
    mn.semantic( 1)
    mn.W[0] = 1
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 2.0)

def test_Capacitance():
    mn = ModifiedNodal()
    mn.add( CapacitanceElement( 1, 0, 2))
    mn.semantic( 1)
    mn.W[0] = 1
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 0.5)

def test_Inductance():
    mn = ModifiedNodal()
    mn.add( InductanceElement( 1, 0, 2))
    mn.semantic( 1)
    mn.W[0] = 1
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 2.0)

def test_Impedance():
    mn = ModifiedNodal()
    mn.add( ImpedanceElement( 1, 0, r=2, l=3))
    mn.semantic( 1)
    mn.W[0] = 1
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 5.0)

def test_CurrentSource():
    mn = ModifiedNodal()
    mn.add( ConductanceElement( 1, 0, 2))
    mn.add( CurrentSourceElement( 0, 1, 1))
    mn.semantic( 1)
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 0.5)

def test_VoltageSource():
    mn = ModifiedNodal()
    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.semantic( 1)
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), 1.0)

def test_VVT():
    mn = ModifiedNodal()
    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.add( VVTElement( 1, 0, 2, 0, -3))
    mn.semantic( 2)
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), -3.0)

def test_VCT():
    mn = ModifiedNodal()
    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.add( VCTElement( 1, 0, 0, 2, -3))
    mn.add( ConductanceElement( 2, 0, 1))
    mn.semantic( 2)
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), -3.0)

def test_OpAmp():
    "Inverting Amp"
    mn = ModifiedNodal()
    mn.add( VoltageSourceElement( 1, 0, 1))
    mn.add( OpAmpElement( 0, 2, 3, 0))
    G1 = 10
    G2 = 2.5
    mn.add( ConductanceElement( 1, 2, G1))
    mn.add( ConductanceElement( 2, 3, G2))
    mn.semantic( 3)
    mn.factor( s=1)
    mn.solve()
    assert np.isclose( mn.phi(), -G1/G2)
