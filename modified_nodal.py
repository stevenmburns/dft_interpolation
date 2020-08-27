
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix, linalg as sla


class Element:
    def __init__(self):
        pass


class TwoTerminalElement(Element):
    def __init__(self, i, j, extra_row_cols=0):
        self.i = i
        self.j = j
        self.extra_row_cols = extra_row_cols
        super().__init__()

    def max_node(self):
        return max(self.i, self.j)


class FourTerminalElement(Element):
    def __init__(self, j, jp, k, kp, extra_row_cols=0):
        self.j = j
        self.jp = jp
        self.k = k
        self.kp = kp
        self.extra_row_cols = extra_row_cols
        super().__init__()

    def max_node(self):
        return max(self.j, self.jp, self.k, self.kp)


class ResistanceElement(TwoTerminalElement):
    def __init__(self, i, j, r):
        self.r = r
        super().__init__(i, j, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(self.i, None, 1)
        mn.assign_G(None, self.i, 1)
        mn.assign_G(self.j, None, -1)
        mn.assign_G(None, self.j, -1)
        mn.assign_G(None, None, -self.r)
        mn.incr_cursor()

    def sens(self, mn):
        return -mn.Xa[self.cursor]*mn.X[self.cursor]

    def rel_sens(self, mn):
        # f(x)
        # f'(x) --- absolute sens
        # df/dx
        # (df/f) / (dx/x) --- rel sens
        # x/f*df/dx --- rel sens

        return self.sens(mn)*self.r/mn.phi()


class InductanceElement(TwoTerminalElement):
    def __init__(self, i, j, l):
        self.l = l
        super().__init__(i, j, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(self.i, None, 1)
        mn.assign_G(None, self.i, 1)
        mn.assign_G(self.j, None, -1)
        mn.assign_G(None, self.j, -1)
        mn.assign_C(None, None, -self.l)
        mn.incr_cursor()

    def sens(self, mn):
        return -mn.s*mn.Xa[self.cursor]*mn.X[self.cursor]

    def rel_sens(self, mn):
        return self.sens(mn)*self.l/mn.phi()

class ImpedanceElement(TwoTerminalElement):
    def __init__(self, i, j, *, r, l):
        self.r = r
        self.l = l
        super().__init__(i, j, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(self.i, None, 1)
        mn.assign_G(None, self.i, 1)
        mn.assign_G(self.j, None, -1)
        mn.assign_G(None, self.j, -1)
        mn.assign_G(None, None, -self.r)
        mn.assign_C(None, None, -self.l)
        mn.incr_cursor()

    def sens(self, mn, key='r'):
        if key == 'r':
            return -mn.Xa[self.cursor]*mn.X[self.cursor]
        elif key == 'l':
            return -mn.s*mn.Xa[self.cursor]*mn.X[self.cursor]
        else:
            return None

    def rel_sens(self, mn, key='r'):
        if key == 'r':
            return self.sens(mn,key)*self.r/mn.phi()
        elif key == 'l':
            return self.sens(mn,key)*self.l/mn.phi()
        else:
            return None

class ConductanceElement(TwoTerminalElement):
    def __init__(self, i, j, g):
        self.g = g
        super().__init__(i, j)

    def update(self, mn):
        mn.update_G(self.i, self.i, self.g)
        mn.update_G(self.j, self.j, self.g)
        mn.update_G(self.i, self.j, -self.g)
        mn.update_G(self.j, self.i, -self.g)

    def sens(self, mn):
        result = 0
        if self.i > 0:
            result += mn.Xa[self.i-1]*mn.X[self.i-1]
        if self.j > 0:
            result += mn.Xa[self.j-1]*mn.X[self.j-1]
        if self.i > 0 and self.j > 0:
            result -= mn.Xa[self.i-1]*mn.X[self.j-1]
            result -= mn.Xa[self.j-1]*mn.X[self.i-1]
        return result

    def rel_sens(self, mn):
        return self.sens(mn)*self.g/mn.phi()


class CapacitanceElement(TwoTerminalElement):
    def __init__(self, i, j, c):
        self.c = c
        super().__init__(i, j)

    def update(self, mn):
        mn.update_C(self.i, self.i, self.c)
        mn.update_C(self.j, self.j, self.c)
        mn.update_C(self.i, self.j, -self.c)
        mn.update_C(self.j, self.i, -self.c)

    def sens(self, mn):
        result = 0
        if self.i > 0:
            result += mn.Xa[self.i-1]*mn.X[self.i-1]
        if self.j > 0:
            result += mn.Xa[self.j-1]*mn.X[self.j-1]
        if self.i > 0 and self.j > 0:
            result -= mn.Xa[self.i-1]*mn.X[self.j-1]
            result -= mn.Xa[self.j-1]*mn.X[self.i-1]
        return mn.s*result

    def rel_sens(self, mn):
        return self.sens(mn)*self.c/mn.phi()


class CurrentSourceElement(TwoTerminalElement):
    def __init__(self, i, j, J):
        self.J = J
        super().__init__(i, j)

    def update(self, mn):
        mn.update_W(self.i, -self.J)
        mn.update_W(self.j,  self.J)

    def sens(self, mn):
        result = 0
        if self.i > 0:
            result -= mn.Xa[self.i-1]*mn.W[self.i-1]
        if self.j > 0:
            result -= mn.Xa[self.j-1]*mn.W[self.j-1]
        return result

    def rel_sens(self, mn):
        return self.sens(mn)*self.J/mn.phi()


class VoltageSourceElement(TwoTerminalElement):
    def __init__(self, i, j, E):
        self.E = E
        super().__init__(i, j, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(self.i, None, 1)
        mn.assign_G(None, self.i, 1)
        mn.assign_G(self.j, None, -1)
        mn.assign_G(None, self.j, -1)
        mn.assign_W(None, self.E)
        mn.incr_cursor()

    def sens(self, mn):
        return -mn.Xa[self.cursor]*mn.W[self.cursor]

    def rel_sens(self, mn):
        return self.sens(mn)*self.E/mn.phi()


class VVTElement(FourTerminalElement):
    def __init__(self, j, jp, k, kp, mu):
        self.mu = mu
        super().__init__(j, jp, k, kp, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(None, self.j, -self.mu)
        mn.assign_G(None, self.jp, self.mu)
        mn.assign_G(self.k, None, 1)
        mn.assign_G(None, self.k, 1)
        mn.assign_G(self.kp, None, -1)
        mn.assign_G(None, self.kp, -1)
        mn.incr_cursor()

    def sens(self, mn):
        result = 0
        if self.j > 0:
            result -= mn.Xa[self.cursor]*mn.X[self.j-1]
        if self.jp > 0:
            result += mn.Xa[self.cursor]*mn.X[self.jp-1]
        return result

    def rel_sens(self, mn):
        return self.sens(mn)*self.mu/mn.phi()


class VCTElement(FourTerminalElement):
    def __init__(self, j, jp, k, kp, g):
        self.g = g
        super().__init__(j, jp, k, kp)

    def update(self, mn):
        mn.update_G(self.k, self.j, self.g)
        mn.update_G(self.k, self.jp, -self.g)
        mn.update_G(self.kp, self.j, -self.g)
        mn.update_G(self.kp, self.jp, self.g)

    def sens(self, mn):
        result = 0
        if self.k > 0 and self.j > 0:
            result += mn.Xa[self.k-1]*mn.X[self.j-1]
        if self.k > 0 and self.jp > 0:
            result -= mn.Xa[self.k-1]*mn.X[self.jp-1]
        if self.kp > 0 and self.j > 0:
            result -= mn.Xa[self.kp-1]*mn.X[self.j-1]
        if self.kp > 0 and self.jp > 0:
            result += mn.Xa[self.kp-1]*mn.X[self.jp-1]

        return result

    def rel_sens(self, mn):
        return self.sens( mn)/self.g


class OpAmpElement(FourTerminalElement):
    def __init__(self, j, jp, k, kp):
        super().__init__(j, jp, k, kp, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(None, self.j, 1)
        mn.assign_G(None, self.jp, -1)
        mn.assign_G(self.k, None, 1)
        mn.assign_G(self.kp, None, -1)
        mn.incr_cursor()

    def sens(self, mn):
        return None  # No parameters (we could use inverse gain (B))

    def rel_sens(self, mn):
        return None


class IdealTransformerElement(FourTerminalElement):
    def __init__(self, j, jp, k, kp, n):
        self.n = n
        super().__init__(j, jp, k, kp, 1)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(None, self.j, 1)
        mn.assign_G(None, self.jp, -1)
        mn.assign_G(None, self.k, -self.n)
        mn.assign_G(None, self.kp, self.n)
        mn.assign_G(self.j, None, 1)
        mn.assign_G(self.jp, None, -1)
        mn.assign_G(self.k, None, -self.n)
        mn.assign_G(self.kp, None, self.n)
        mn.incr_cursor()

    def sens(self, mn):
        result = 0
        if self.k > 0:
            result -= mn.Xa[self.k-1]*mn.X[self.cursor]
            result -= mn.Xa[self.cursor]*mn.X[self.k-1]
        if self.kp > 0:
            result += mn.Xa[self.kp-1]*mn.X[self.cursor]
            result += mn.Xa[self.cursor]*mn.X[self.kp-1]
        return result

    def rel_sens(self, mn):
        return self.sens(mn)*self.n/mn.phi()


class TransformerElement(FourTerminalElement):
    def __init__(self, j, jp, k, kp, *, l1, l2, m):
        self.l1 = l1
        self.l2 = l2
        self.m = m
        super().__init__(j, jp, k, kp, 2)

    def update(self, mn):
        self.cursor = mn.cursor
        mn.assign_G(None, self.j, 1)
        mn.assign_G(None, self.jp, -1)
        mn.assign_G(self.j, None, 1)
        mn.assign_G(self.jp, None, -1)
        mn.incr_cursor()

        mn.assign_G(None, self.k, 1)
        mn.assign_G(None, self.kp, -1)
        mn.assign_G(self.k, None, 1)
        mn.assign_G(self.kp, None, -1)

        # None trick can't be used because there are two cursors
        mn.c_dok[self.cursor+0,self.cursor+0] = -self.l1
        mn.c_dok[self.cursor+0,self.cursor+1] = -self.m
        mn.c_dok[self.cursor+1,self.cursor+0] = -self.m
        mn.c_dok[self.cursor+1,self.cursor+1] = -self.l2

        mn.incr_cursor()

    def sens(self, mn, key='l1'):
        if key == 'l1':
            return -mn.s*mn.Xa[self.cursor+0]*mn.X[self.cursor+0]
        if key == 'l2':
            return -mn.s*mn.Xa[self.cursor+1]*mn.X[self.cursor+1]
        if key == 'm':
            result = 0
            result += -mn.s*mn.Xa[self.cursor+0]*mn.X[self.cursor+1]
            result += -mn.s*mn.Xa[self.cursor+1]*mn.X[self.cursor+0]
            return result

    def rel_sens(self, mn, key='l1'):
        if key == 'l1':
            return self.sens(mn,key)*self.l1/mn.phi()
        if key == 'l2':
            return self.sens(mn,key)*self.l2/mn.phi()
        if key == 'm':
            return self.sens(mn,key)*self.m/mn.phi()

class ModifiedNodal:
    #factor_napiers_to_dbs = 20*np.log10(np.e)
    factor_napiers_to_dbs = 20/np.log(10)

    def __init__(self):
        self.elements = []

    def add(self, el):
        self.elements.append(el)

    def incr_cursor(self):
        self.cursor += 1
        assert self.cursor <= self.n

    def update_G(self, i, j, val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i > 0 and j > 0:
            self.g_dok[i-1, j-1] = self.g_dok[i-1, j-1] + val

    def assign_G(self, i, j, val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i > 0 and j > 0:
            self.g_dok[i-1, j-1] = val

    def update_C(self, i, j, val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i > 0 and j > 0:
            self.c_dok[i-1, j-1] = self.c_dok[i-1, j-1] + val

    def assign_C(self, i, j, val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i > 0 and j > 0:
            self.c_dok[i-1, j-1] = val

    def update_W(self, i, val):
        if i is None:
            i = self.cursor+1
        if i > 0:
            self.W[i-1] += val

    def assign_W(self, i, val):
        if i is None:
            i = self.cursor+1
        if i > 0:
            self.W[i-1] = val

    def semantic(self, output_node):
        max_node = 0
        for el in self.elements:
            max_node = max(max_node, el.max_node())

        self.n = max_node
        for el in self.elements:
            self.n += el.extra_row_cols

        self.cursor = max_node

        self.g_dok = dok_matrix((self.n, self.n))
        self.c_dok = dok_matrix((self.n, self.n))
        self.W = np.zeros((self.n,))
        self.d = np.zeros((self.n,))
        if output_node > 0:
            self.d[output_node-1] = 1

        for el in self.elements:
            el.update(self)

        assert self.cursor == self.n

        self.G = csc_matrix(self.g_dok)
        self.C = csc_matrix(self.c_dok)

        del self.g_dok
        del self.c_dok

        print(f"G: {self.G.A}")
        print(f"C: {self.C.A}")
        print(f"W: {self.W}")
        print(f"d: {self.d}")

    def factor(self, *, s=0):
        self.s = s
        self.T = self.G + self.s*self.C
        self.lu = sla.splu(self.T)

    def solve(self):
        self.X = self.lu.solve(self.W)
        return self.X

    def solve_adjoint(self):
        self.Xa = self.lu.solve(-self.d, trans='T')
        return self.Xa

    def phi(self):
        return self.d.T.dot(self.X)

    def phase_sens(self, sens):
        return (sens/self.phi()).imag

    def abs_sens(self, sens):
        f = self.phi()
        return np.abs(f)*(sens/f).real

    def abs_sens_napiers(self, sens):
        return (sens/self.phi()).real

    def abs_sens_dbs(self, sens):
        return self.factor_napiers_to_dbs*self.abs_sens_napiers(sens)

    def sens_to_omega(self):
        return 1j*self.Xa.T.dot(self.C.dot(self.X))

    def sens_to_log(self, sens, x):
        return sens*x

    def sens_to_log10(self, sens, x):
        return sens*x*np.log(10)

    def sensitivities(self):
        return [el.sens(self) for el in self.elements]

    def rel_sensitivities(self):
        return [el.rel_sens(self) for el in self.elements]
