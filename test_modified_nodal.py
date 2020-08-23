
import numpy as np
from modified_nodal import *


def test_Conductance():
    mn = ModifiedNodal()
    mn.add(ConductanceElement(1, 0, 2))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 0.5)
    # I=GV; V=I/G; V=IG^-1; V'=-IG^-2=-0.25
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), -0.25)


def test_Resistance():
    mn = ModifiedNodal()
    mn.add(ResistanceElement(1, 0, 2))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 2.0)
    #  V=IR; V'=I=1
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), 1)


def test_Capacitance():
    mn = ModifiedNodal()
    mn.add(CapacitanceElement(1, 0, 2))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 0.5)
    # I=CsV; V=I/(Cs); V=(I/s) C^-1; V'=-(I/s) C^-2=-0.25
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), -0.25)


def test_Inductance():
    mn = ModifiedNodal()
    mn.add(InductanceElement(1, 0, 2))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 2.0)
    # V=IsL; V'=(Is)=1
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), 1)


def test_Impedance():
    mn = ModifiedNodal()
    mn.add(ImpedanceElement(1, 0, r=2, l=3))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 5.0)
    # V=IZ; V'=I=1
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), 1)


def test_CurrentSource():
    mn = ModifiedNodal()
    G = 2
    mn.add(ConductanceElement(1, 0, G))
    mn.add(CurrentSourceElement(0, 1, 1))
    mn.semantic(1)
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 0.5)
    # V=I/G; V'=1/G
    mn.solve_adjoint()
    assert np.isclose(mn.elements[1].sens(mn), 1/G)


def test_VoltageSource():
    mn = ModifiedNodal()
    mn.add(VoltageSourceElement(1, 0, 1))
    mn.semantic(1)
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), 1.0)
    # V=E; V'=1
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), 1)


def test_VVT():
    mn = ModifiedNodal()
    E = 1
    mu = -3
    mn.add(VoltageSourceElement(1, 0, E))
    mn.add(VVTElement(1, 0, 2, 0, mu))
    mn.semantic(2)
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), E*mu)
    # V=mu E; V'=E
    mn.solve_adjoint()
    assert np.isclose(mn.elements[1].sens(mn), E)


def test_VCT():
    mn = ModifiedNodal()
    E = 1
    g = -3
    g2 = 2
    mn.add(VoltageSourceElement(1, 0, E))
    mn.add(VCTElement(1, 0, 0, 2, g))
    mn.add(ConductanceElement(2, 0, g2))
    mn.semantic(2)
    mn.factor(s=1)
    mn.solve()
    # V=E*g/g2
    assert np.isclose(mn.phi(), E*g/g2)
    # V'=E/g2
    mn.solve_adjoint()
    assert np.isclose(mn.elements[1].sens(mn), E/g2)


def test_OpAmp():
    "Inverting Amp"
    mn = ModifiedNodal()
    mn.add(VoltageSourceElement(1, 0, 1))
    mn.add(OpAmpElement(0, 2, 3, 0))
    G1 = 10
    G2 = 2.5
    mn.add(ConductanceElement(1, 2, G1))
    mn.add(ConductanceElement(2, 3, G2))
    mn.semantic(3)
    mn.factor(s=1)
    mn.solve()
    assert np.isclose(mn.phi(), -G1/G2)


def test_ex_6_1_1():
    "page 174"
    mn = ModifiedNodal()
    k = 1
    G1 = 1
    G2 = 1
    C1 = 1
    C2 = 1

    mn.add(VoltageSourceElement(3, 0, 1))
    mn.add(ConductanceElement(3, 1, G1))
    mn.add(ConductanceElement(1, 2, G2))
    mn.add(CapacitanceElement(2, 0, C2))
    mn.add(CapacitanceElement(4, 1, C1))
    mn.add(VVTElement(2, 0, 4, 0, k))

    mn.semantic(2)
    mn.factor(s=2)
    mn.solve()
    assert np.isclose(mn.phi(), 1/9)
    mn.solve_adjoint()
    assert np.isclose(mn.elements[1].sens(mn), 2/27)


def test_ex_6_4_1():
    "page 180"
    mn = ModifiedNodal()
    J = 1
    G1 = 1
    C1 = 0
    G2 = 0
    C2 = 1
    G3 = 1
    C3 = 0

    mn.add(CurrentSourceElement(0, 1, J))
    mn.add(ConductanceElement(1, 0, G1))
    mn.add(CapacitanceElement(1, 0, C1))
    mn.add(ConductanceElement(1, 2, G2))
    mn.add(CapacitanceElement(1, 2, C2))
    mn.add(ConductanceElement(2, 0, G3))
    mn.add(CapacitanceElement(2, 0, C3))

    mn.semantic(2)
    mn.factor(s=1j)
    mn.solve()
    assert np.isclose(mn.phi(), 0.4+0.2j)
    mn.solve_adjoint()
    print(mn.X, mn.Xa)
    assert np.isclose(mn.elements[2].sens(mn), 1/25-7j/25)
    assert np.isclose(mn.elements[3].sens(mn), -3/25-4j/25)
    assert np.isclose(mn.elements[6].sens(mn), 1/25-7j/25)


def test_ex_6_5_1():
    "page 184"
    mn = ModifiedNodal()
    J = 1
    G1 = 4
    C = 1
    G2 = 1
    L = 1
    g = 3

    mn.add(CurrentSourceElement(0, 1, J))
    mn.add(ConductanceElement(1, 0, G1))
    mn.add(CapacitanceElement(1, 0, C))
    mn.add(ConductanceElement(1, 2, G2))
    mn.add(InductanceElement(2, 0, L))
    mn.add(VCTElement(1, 0, 0, 2, g))

    mn.semantic(2)
    mn.factor(s=1j)
    mn.solve()

    mn.solve_adjoint()
    #print(mn.G, mn.C, mn.W, mn.d, mn.X, mn.Xa)
    assert np.isclose(mn.phi(), (8+16j)/20)
    d_V2_by_d_L = mn.elements[4].sens(mn)
    assert np.isclose(d_V2_by_d_L, (272+304j)/400)
    assert np.isclose(mn.abs_sens(d_V2_by_d_L), 22*np.sqrt(320)/400)
    assert np.isclose(mn.abs_sens_dbs(d_V2_by_d_L), 20*np.log10(np.e)*22/20)
    assert np.isclose(mn.phase_sens(d_V2_by_d_L), -6/20)

def test_ex_6_5_2():
    "page 186"
    mn = ModifiedNodal()
    J = 1
    G1 = 4
    C = 1
    G2 = 1
    L = 1
    g = 3
    omega = 1

    mn.add(CurrentSourceElement(0, 1, J))
    mn.add(ConductanceElement(1, 0, G1))
    mn.add(CapacitanceElement(1, 0, C))
    mn.add(ConductanceElement(1, 2, G2))
    mn.add(InductanceElement(2, 0, L))
    mn.add(VCTElement(1, 0, 0, 2, g))

    mn.semantic(2)
    mn.factor(s=omega*1j)
    mn.solve()

    mn.solve_adjoint()
    #print(mn.G, mn.C, mn.W, mn.d, mn.X, mn.Xa)
    assert np.isclose(mn.phi(), (8+16j)/20)
    d_V2_by_d_C = mn.elements[2].sens(mn)
    d_V2_by_d_L = mn.elements[4].sens(mn)

    d_V2_by_d_omega = 1/omega * ( C*d_V2_by_d_C + L*d_V2_by_d_L)
    assert np.isclose(d_V2_by_d_omega, (384+288j)/400)

    d_V2_by_d_omega2 = mn.sens_to_omega()
    print( f"d_V2_by_d_omega{{,2}}: {d_V2_by_d_omega} {d_V2_by_d_omega2}")
    assert np.isclose(d_V2_by_d_omega, (384+288j)/400)

    assert np.isclose(mn.abs_sens(d_V2_by_d_omega), 24*np.sqrt(320)/400)
    assert np.isclose(mn.abs_sens_dbs(d_V2_by_d_omega), 20*np.log10(np.e)*24/20)
    assert np.isclose(mn.phase_sens(d_V2_by_d_omega), -12/20)
