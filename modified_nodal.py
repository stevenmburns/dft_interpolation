
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
import itertools

class Element:
    def __init__(self):
        pass

class TwoTerminalElement(Element):
    def __init__(self,i,j,extra_row_cols=0):
        self.i = i
        self.j = j
        self.extra_row_cols = extra_row_cols
        super().__init__()

    def max_node(self):
        return max(self.i, self.j)

class ResistanceElement(TwoTerminalElement):
    def __init__(self,i,j,r):
        self.r = r
        super().__init__(i,j,1)

    def update(self, mn):
        self.extra = mn.cursor
        mn.assign_g_dok(self.i,None, 1)
        mn.assign_g_dok(None,self.i, 1)
        mn.assign_g_dok(self.j,None,-1)
        mn.assign_g_dok(None,self.j,-1)
        mn.assign_g_dok(None,None, -self.r)
        mn.incr_cursor()   

class InductanceElement(TwoTerminalElement):
    def __init__(self,i,j,l):
        self.l = l
        super().__init__(i,j,1)

    def update(self, mn):
        self.extra = mn.cursor
        mn.assign_c_dok(self.i,None, 1)
        mn.assign_c_dok(None,self.i, 1)
        mn.assign_c_dok(self.j,None,-1)
        mn.assign_c_dok(None,self.j,-1)
        mn.assign_c_dok(None,None, -self.l)
        mn.incr_cursor()   

class ConductanceElement(TwoTerminalElement):
    def __init__(self,i,j,g):
        self.g = g
        super().__init__(i,j)

    def update(self, mn):
        mn.update_g_dok(self.i,self.i, self.g)
        mn.update_g_dok(self.j,self.j, self.g)
        mn.update_g_dok(self.i,self.j,-self.g)
        mn.update_g_dok(self.j,self.i,-self.g)

class CapacitanceElement(TwoTerminalElement):
    def __init__(self,i,j,c):
        self.c = c
        super().__init__(i,j)

    def update(self, mn):
        mn.update_c_dok(self.i,self.i, self.c)
        mn.update_c_dok(self.j,self.j, self.c)
        mn.update_c_dok(self.i,self.j,-self.c)
        mn.update_c_dok(self.j,self.i,-self.c)

class ModifiedNodal:
    def __init__( self):
        self.elements = []

        self.conductances = []
        self.resistances = []
        self.capacitances = []
        self.inductances = []
        self.current_sources = []
        self.voltage_sources = []
        self.vvt_sources = []
        self.vct_sources = []
        self.op_amps = []

    def add(self, el):
        self.elements.append( el)

    def incr_cursor(self):
        self.cursor += 1
        assert self.cursor <= self.n

    # i,j numbered from 0 on up (0 is reference node [gnd])
    def add_conductance( self, i, j, g):
        self.add( ConductanceElement( i, j, g))
        #self.conductances.append( (i,j,g))

    def add_resistance( self, i, j, r):
        self.add( ResistanceElement( i, j, r))
        #self.resistances.append( (i,j,r))

    def add_capacitance( self, i, j, c):
        self.add( CapacitanceElement( i, j, c))
        #self.capacitances.append( (i,j,c))

    def add_inductance( self, i, j, l):
        self.add( InductanceElement( i, j, l))
        #self.inductances.append( (i,j,l))

    def add_current_source( self, i, j, J):
        self.current_sources.append( (i,j,J))

    def add_voltage_source( self, i, j, E):
        self.voltage_sources.append( (i,j,E))

    def add_vvt_source( self, j, jp, k, kp, mu):
        self.vvt_sources.append( (j,jp,k,kp,mu))

    def add_vct_source( self, j, jp, k, kp, g):
        self.vct_sources.append( (j,jp,k,kp,g))

    def add_op_amp( self, j, jp, k, kp):
        self.op_amps.append( (j,jp,k,kp))

    def update_g_dok(self,i,j,val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i>0 and j>0:
            self.g_dok[i-1,j-1] = self.g_dok[i-1,j-1] + val

    def assign_g_dok(self,i,j,val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i>0 and j>0:
            self.g_dok[i-1,j-1] = val

    def update_c_dok(self,i,j,val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i>0 and j>0:
            self.c_dok[i-1,j-1] = self.c_dok[i-1,j-1] + val

    def assign_c_dok(self,i,j,val):
        if i is None:
            i = self.cursor+1
        if j is None:
            j = self.cursor+1
        if i>0 and j>0:
            self.c_dok[i-1,j-1] = val

    def semantic( self, output_node):
        max_node = 0
        for el in self.elements:
            max_node = max(max_node,el.max_node())

        for tup in itertools.chain( self.current_sources, self.voltage_sources):
            for x in tup[:2]:
                if max_node < x:
                    max_node = x

        for tup in itertools.chain( self.vvt_sources, self.vct_sources, self.op_amps):
            for x in tup[:4]:
                if max_node < x:
                    max_node = x

        self.n = max_node
        for el in self.elements:
            self.n += el.extra_row_cols

        # nodes go from 0 to max_node-1
        # need number of impedances more rows and columns
        self.n += len(self.voltage_sources) + len(self.vvt_sources) + len(self.op_amps)

        self.cursor = max_node

        self.g_dok = dok_matrix( (self.n, self.n))
        self.c_dok = dok_matrix( (self.n, self.n))
        w = np.zeros( (self.n,))
        d = np.zeros( (self.n,))
        if output_node > 0:
            d[output_node-1] = 1

        for el in self.elements:
            el.update( self)

        for i, j, g in self.conductances:
            self.update_g_dok(i,i, g)
            self.update_g_dok(j,j, g)
            self.update_g_dok(i,j,-g)
            self.update_g_dok(j,i,-g)

        for i, j, c in self.capacitances:
            if i > 0:
                self.c_dok[i-1,i-1] = self.c_dok[i-1,i-1] + c
            if j > 0:
                self.c_dok[j-1,j-1] = self.c_dok[j-1,j-1] + c
            if i > 0 and j > 0:
                self.c_dok[i-1,j-1] = self.c_dok[i-1,j-1] - c
                self.c_dok[j-1,i-1] = self.c_dok[j-1,i-1] - c

        for i, j, r in self.resistances:
            if i > 0:
                self.g_dok[i-1,self.cursor] = 1
                self.g_dok[self.cursor,i-1] = 1
            if j > 0:
                self.g_dok[j-1,self.cursor] = -1
                self.g_dok[self.cursor,j-1] = -1
            self.g_dok[self.cursor,self.cursor] = -r
            self.incr_cursor()
        
        for i, j, l in self.inductances:
            if i > 0:
                self.g_dok[i-1,self.cursor] = 1
                self.g_dok[self.cursor,i-1] = 1
            if j > 0:
                self.g_dok[j-1,self.cursor] = -1
                self.g_dok[self.cursor,j-1] = -1
            self.c_dok[self.cursor,self.cursor] = -l
            self.incr_cursor()

        for i, j, J in self.current_sources:
            if i > 0:
                w[i-1] = w[i-1] - J
            if j > 0:
                w[j-1] = w[j-1] + J

        for i, j, E in self.voltage_sources:
            if i > 0:
                self.g_dok[i-1,self.cursor] = 1
                self.g_dok[self.cursor,i-1] = 1
            if j > 0:
                self.g_dok[j-1,self.cursor] = -1
                self.g_dok[self.cursor,j-1] = -1
            w[self.cursor] = E
            self.incr_cursor()

        for j, jp, k, kp, g in self.vct_sources:
            if k > 0 and j > 0:
                self.g_dok[k-1,j-1] = self.g_dok[k-1,j-1] + g
            if k > 0 and jp > 0:
                self.g_dok[k-1,jp-1] = self.g_dok[k-1,jp-1] - g
            if kp > 0 and j > 0:
                self.g_dok[kp-1,j-1] = self.g_dok[kp-1,j-1] - g
            if kp > 0 and jp > 0:
                self.g_dok[kp-1,jp-1] = self.g_dok[kp-1,jp-1] + g

        for j, jp, k, kp, mu in self.vvt_sources:
            if j > 0:
                self.g_dok[self.cursor,j-1] = -mu
            if jp > 0:
                self.g_dok[self.cursor,jp-1] = mu
            if k > 0:
                self.g_dok[k-1,self.cursor] = 1
                self.g_dok[self.cursor,k-1] = 1
            if kp > 0:
                self.g_dok[kp-1,self.cursor] = -1
                self.g_dok[self.cursor,kp-1] = -1
            self.incr_cursor()

        for j, jp, k, kp in self.op_amps:
            if j > 0:
                self.g_dok[self.cursor,j-1] = 1
            if jp > 0:
                self.g_dok[self.cursor,jp-1] = -1
            if k > 0:
                self.g_dok[k-1,self.cursor] = 1
            if kp > 0:
                self.g_dok[kp-1,self.cursor] = -1
            self.incr_cursor()

        assert self.cursor == self.n

        self.G = csc_matrix(self.g_dok)
        self.C = csc_matrix(self.c_dok)
        self.W = w
        self.d = d

        print( f"G: {self.G.A}")
        print( f"C: {self.C.A}")
        print( f"W: {self.W}")
        print( f"d: {self.d}")
