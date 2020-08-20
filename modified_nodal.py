
import numpy as np
from scipy.sparse import dok_matrix, csc_matrix
import itertools

class ModifiedNodal:
    def __init__( self):
        self.conductances = []
        self.resistances = []
        self.capacitances = []
        self.inductances = []
        self.current_sources = []
        self.voltage_sources = []

    # i,j numbered from 0 on up (0 is reference node [gnd])
    def add_conductance( self, i, j, g):
        self.conductances.append( (i,j,g))

    def add_resistance( self, i, j, r):
        self.resistances.append( (i,j,r))

    def add_capacitance( self, i, j, c):
        self.capacitances.append( (i,j,c))

    def add_inductance( self, i, j, l):
        self.inductances.append( (i,j,l))

    def add_current_source( self, i, j, J):
        self.current_sources.append( (i,j,J))

    def add_voltage_source( self, i, j, E):
        self.voltage_sources.append( (i,j,E))

    def semantic( self, output_node):
        max_node = 0
        for tup in itertools.chain( self.conductances, self.resistances, self.capacitances, self.inductances):
            i, j, _ = tup 
            for x in [i, j]:
                if max_node < x:
                    max_node = x

        # nodes go from 0 to max_node-1
        # need number of impedances more rows and columns
        n = max_node + len(self.resistances) + len(self.inductances) + len(self.voltage_sources)

        cursor = max_node

        g_dok = dok_matrix( (n, n))
        c_dok = dok_matrix( (n, n))
        w = np.zeros( (n,))
        d = np.zeros( (n,))
        if output_node > 0:
            d[output_node-1] = 1

        for i, j, g in self.conductances:
            if i > 0:
                g_dok[i-1,i-1] = g_dok[i-1,i-1] + g
            if j > 0:
                g_dok[j-1,j-1] = g_dok[j-1,j-1] + g
            if i > 0 and j > 0:
                g_dok[i-1,j-1] = g_dok[i-1,j-1] - g
                g_dok[j-1,i-1] = g_dok[j-1,i-1] - g

        for i, j, c in self.capacitances:
            if i > 0:
                c_dok[i-1,i-1] = c_dok[i-1,i-1] + c
            if j > 0:
                c_dok[j-1,j-1] = c_dok[j-1,j-1] + c
            if i > 0 and j > 0:
                c_dok[i-1,j-1] = c_dok[i-1,j-1] - c
                c_dok[j-1,i-1] = c_dok[j-1,i-1] - c

        for i, j, r in self.resistances:
            if i > 0:
                g_dok[i-1,cursor] = 1
                g_dok[cursor,i-1] = 1
            if j > 0:
                g_dok[j-1,cursor] = -1
                g_dok[cursor,j-1] = -1
            g_dok[cursor,cursor] = -r
            cursor += 1
            assert cursor <= n
        
        for i, j, l in self.inductances:
            if i > 0:
                g_dok[i-1,cursor] = 1
                g_dok[cursor,i-1] = 1
            if j > 0:
                g_dok[j-1,cursor] = -1
                g_dok[cursor,j-1] = -1
            c_dok[cursor,cursor] = -l
            cursor += 1
            assert cursor <= n

        for i, j, J in self.current_sources:
            if i > 0:
                w[i-1] = w[i-1] - J
            if j > 0:
                w[j-1] = w[j-1] + J

        for i, j, E in self.voltage_sources:
            if i > 0:
                g_dok[i-1,cursor] = 1
                g_dok[cursor,i-1] = 1
            if j > 0:
                g_dok[j-1,cursor] = -1
                g_dok[cursor,j-1] = -1
            w[cursor] = E
            cursor += 1
            assert cursor <= n

        self.G = csc_matrix(g_dok)
        self.C = csc_matrix(c_dok)
        self.W = w
        self.d = d

        print( f"G: {self.G.A}")
        print( f"C: {self.C.A}")
        print( f"W: {self.W}")
        print( f"d: {self.d}")
