
import numpy as np
from modified_nodal import *
import pytest

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
    omega = 1
    C = 2
    mn.add(CapacitanceElement(1, 0, C))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1j*omega)
    mn.solve()
    assert np.isclose(mn.phi(), 1/(C*mn.s))
    # I=CsV; V=I/(Cs); V=(I/s) C^-1; V'=-(I/s) C^-2=-0.25
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), -1/(C**2*mn.s))


def test_Inductance():
    mn = ModifiedNodal()
    omega = 3
    L = 2
    mn.add(InductanceElement(1, 0, L))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1j*omega)
    mn.solve()
    assert np.isclose(mn.phi(), L*mn.s)
    # V=IsL; V'=(Is)=s
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn), mn.s)


def test_Impedance():
    mn = ModifiedNodal()
    omega = 2
    R = 2
    L = 3
    mn.add(ImpedanceElement(1, 0, r=R, l=L))
    mn.semantic(1)
    mn.W[0] = 1
    mn.factor(s=1j*omega)
    mn.solve()
    assert np.isclose(mn.phi(), R + L*mn.s)
    # V=IZ; V'=I=1
    mn.solve_adjoint()
    assert np.isclose(mn.elements[0].sens(mn,'r'), 1)
    assert np.isclose(mn.elements[0].sens(mn,'l'), mn.s)


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


def test_IdealTransformer():
    mn = ModifiedNodal()
    n = 4.5
    mn.add(VoltageSourceElement(1, 0, 1))
    mn.add(IdealTransformerElement(2, 0, 3, 0, n))
    mn.add(ConductanceElement(1, 2, 1))
    mn.add(ConductanceElement(3, 0, 1))
    mn.semantic(3)
    mn.factor(s=1)
    mn.solve()
    mn.solve_adjoint()

    print(mn.phi())
    # n * (n*n+1)^-1
    # (n*n+1)^-1 - n * (n*n+1)^-2 * 2 * n
    assert np.isclose( mn.phi(), n/(n*n+1))
    assert np.isclose( mn.elements[1].sens(mn), mn.phi()/n - 2*mn.phi()**2)

def test_Transformer_ex16_3():
    mn = ModifiedNodal()
    omega = 10
    R1 = 2
    R2 = 3
    L1 = 1
    L2 = 2
    M = 1
    mn.add(VoltageSourceElement(1, 0, 1))
    mn.add(TransformerElement(2, 0, 0, 3, l1=L1, l2=L2, m=M))
    mn.add(ConductanceElement(1, 2, 1/R1))
    mn.add(ConductanceElement(3, 0, 1/R2))
    mn.semantic(3)
    mn.factor(s=1j*omega)
    mn.solve()
    mn.solve_adjoint()

    N = -mn.s*M*R2
    D = (mn.s*L1+R1)*(mn.s*L2+R2)-(mn.s*M)**2
    phi = N/D
    d_phi_d_L1 = -N*mn.s*(mn.s*L2+R2)/(D**2)
    d_phi_d_L2 = -N*mn.s*(mn.s*L1+R1)/(D**2)
    d_phi_d_M = (-mn.s*D*R2-2*mn.s**3*M**2*R2)/(D**2)

    assert np.isclose( mn.phi(), phi)
    assert np.isclose(mn.elements[1].sens(mn,'l1'), d_phi_d_L1)
    assert np.isclose(mn.elements[1].sens(mn,'l2'), d_phi_d_L2)
    

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

    d_V2_by_d_omega = mn.sens_to_omega()
    assert np.isclose(d_V2_by_d_omega, (384+288j)/400)

    d_V2_by_d_omega2 = 1/omega * ( C*d_V2_by_d_C + L*d_V2_by_d_L)
    assert np.isclose(d_V2_by_d_omega2, (384+288j)/400)
    print( f"d_V2_by_d_omega{{,2}}: {d_V2_by_d_omega} {d_V2_by_d_omega2}")


    assert np.isclose(mn.abs_sens(d_V2_by_d_omega), 24*np.sqrt(320)/400)
    assert np.isclose(mn.abs_sens_dbs(d_V2_by_d_omega), mn.factor_napiers_to_dbs*24/20)
    assert np.isclose(mn.phase_sens(d_V2_by_d_omega), -12/20)

def test_one_pole():
    mn = ModifiedNodal()
    E = 1
    G = 1
    C = 1
    omega = 8

    mn.add(VoltageSourceElement(1, 0, E))
    mn.add(ConductanceElement(1, 2, G))
    mn.add(CapacitanceElement(2, 0, C))

    mn.semantic(2)
    mn.factor(s=omega*1j)
    mn.solve()

    print( mn.X)

    # 1/sqrt(1+omega^2)
    assert np.isclose(np.abs(mn.phi()), 1/np.sqrt(1+omega**2))

    mn.solve_adjoint()
    d_V2_by_d_omega = mn.sens_to_omega()

    assert np.isclose(mn.abs_sens(d_V2_by_d_omega), -omega*(1+omega**2)**(-3/2))
    # -1/2*log(1+omega^2)
    # -1/2*1/(1+omega^2)*2*omega
    assert np.isclose(mn.abs_sens_napiers(d_V2_by_d_omega), -omega/(1+omega**2))
    assert np.isclose(mn.abs_sens_dbs(d_V2_by_d_omega), -mn.factor_napiers_to_dbs*omega/(1+omega**2))

    assert np.isclose(mn.sens_to_log10(mn.abs_sens_dbs(d_V2_by_d_omega), omega), -20, atol=0.5)



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

    for log_omega in np.arange(-1, 1.05, 0.1):
        omega = 10**log_omega
        mn.factor(s=omega*1j)
        mn.solve()
        mn.solve_adjoint()

        abs_phi = mn.phi()
        abs_phi_dbs = np.abs(abs_phi)

        #print(mn.sensitivities())

        d_phi_by_d_omega = mn.sens_to_omega()

        abs_sens = mn.abs_sens(d_phi_by_d_omega)
        abs_sens_napiers = mn.abs_sens_napiers(d_phi_by_d_omega)
        abs_sens_dbs = mn.abs_sens_dbs(d_phi_by_d_omega)
        abs_sens_dbs_log10_omega = mn.sens_to_log10(abs_sens_dbs, omega)

        print( omega, abs_phi_dbs, abs_sens, abs_sens_napiers, abs_sens_dbs, abs_sens_dbs_log10_omega)
