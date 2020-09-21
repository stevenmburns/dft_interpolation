import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class Term:
    def __init__(self):
        pass

class LinearTerm(Term):
    def __init__(self, c=0):
        """s+c"""
        self.c = c
        super().__init__()

    def roots(self):
        return [-self.c]

    def eval( self, s):
        return s+self.c

    def eval_squared( self, s2):
        assert False
        return 1

    def toTuple( self):
        return (self.c, )

class QuadraticTerm(Term):
    def __init__(self, w0, c=0):
        """(s+c)^2 + w0^2"""
        self.w0 = w0
        self.c = c
        super().__init__()

    def roots(self):
        return [-self.c + 1j*w0, -self.c - 1j*w0]

    def eval( self, s):
        return (s+self.c)**2 + self.w0**2

    def toTuple( self):
        return (self.c, self.w0)

    def eval_squared( self, s2):
        assert self.c == 0
        return s2 + self.w0**2


class Poly:
    def __init__(self, k=1, terms=None):
        self.k = k
        if terms is None:
            self.terms = []
        else:
            self.terms = terms

    def add_term(self, term):
        self.terms.append( term)

    def eval( self, s):
        prod = self.k
        for term in self.terms:
            prod *= term.eval(s)
        return prod

    def eval_squared( self, s):
        prod = self.k
        for term in self.terms:
            prod *= term.eval_squared(s)
        return prod

    def zeros( self):
        zeros = []
        for term in self.terms:
            if type(term) == LinearTerm:
                assert term.c == 0
                zeros.append( 0)
            elif type(term) == QuadraticTerm:
                assert term.c == 0
                zeros.append( term.w0)
        zeros.sort()
        return zeros

    def has_matched_terms( self, term): 
        return any([ True for t in self.terms if t.toTuple() == term.toTuple()])

    def remove_term( self, term): 
        return Poly( self.k, [ t for t in self.terms if t.toTuple() != term.toTuple()])

def zcount( zeros):
    if len(zeros) == 0:
        return 0
    elif zeros[0] == 0:
        return 2*len(zeros)-1
    else:
        return 2*len(zeros)
        
class ReactanceFunction:
    def __init__(self, num, den):
        self.num = num
        self.den = den
        self.check()

    def has_zero_at_infinity(self):
        nz = self.num.zeros()
        dz = self.den.zeros()
        return zcount(nz) < zcount(dz)

    def has_pole_at_infinity(self):
        nz = self.num.zeros()
        dz = self.den.zeros()
        return zcount(dz) < zcount(nz)

    def eval( self, s):
        return self.num.eval(s) / self.den.eval(s)

    def check(self):

        z0 = self.num.zeros()
        z1 = self.den.zeros()

        assert abs( zcount(z0) - zcount(z1)) <= 1

        if len(z0) == 0 or len(z1) == 0:
            assert len(z0) != 0 or len(z1) == 1
            assert len(z1) != 0 or len(z0) == 1
            return

        assert len(z0) != 0 and len(z1) != 0
        assert z0[0] != z1[0]

        if z0[0] > z1[0]:        
            z1, z0 = z0, z1

        for i in range(len(z1)):
            assert z0[i] < z1[i]
            if i+1 < len(z0):
                assert z1[i] < z0[i+1]

    def intervals( self, delta):
        results = []
        delta = 0.001
        poles = self.den.zeros()
        start = None
        for pole in poles:
            if start is None:
                if pole > 0:
                    results.append( (0, pole-delta/2))
                start = pole+delta/2
            else:
                results.append( (start, pole-delta/2))
                start = pole+delta/2

        results.append( (start, start+2-delta))
        return results

    def plot(self):
        delta = 0.001
        xs = []
        ys = []
        print( self.intervals(delta))

        for start, end in self.intervals(delta):
            xxs = np.arange( start, end+delta/4, delta)
            yys = np.imag(self.eval(xxs * 1j))
            xs.extend( list(xxs))
            ys.extend( list(yys))
            xs.append( None)
            ys.append( None)

        t1 = go.Scatter( x=xs, y=ys, name="reactance")
        ly = go.Layout(
            title='Reactance Plot',
            xaxis={'title': 'Frequency (rad/s)'},
            yaxis={'title': 'Impedance (ohms)', 'range': [-20,20]}
        )
        fig = go.Figure(data=[t1], layout=ly)
        fig.show()

    def residue( self, term):
        assert self.den.has_matched_terms( term)
        
        new_den = self.den.remove_term( term)

        if type(term) == LinearTerm:
            s = -term.c
            return self.num.eval(s) / new_den.eval(s)

        if type(term) == QuadraticTerm:
            assert term.c == 0
            s2 = -term.w0**2

            dc_zero = LinearTerm()
            factor = 1
            new_num = self.num
            if new_den.has_matched_terms( dc_zero):
                new_den = new_den.remove_term( dc_zero)
                factor = 1/s2
            elif self.num.has_matched_terms( dc_zero):
                new_num = new_num.remove_term( dc_zero)
            else:
                assert False

            return factor * new_num.eval_squared(s2) / new_den.eval_squared(s2)

        assert False

    def reciprocal(self):
        return ReactanceFunction( self.den, self.num)

    def foster( self):
        result = []
        for term in self.den.terms:
            r = self.residue( term)
            if type(term) == LinearTerm:
                result.append( ReactanceFunction( Poly( r, []), Poly( 1.0, [term])))
            elif type(term) == QuadraticTerm:
                result.append( ReactanceFunction( Poly( r, [LinearTerm()]), Poly( 1.0, [term])))
            else:
                assert False

        if self.has_pole_at_infinity():
            result.append( ReactanceFunction( Poly( self.num.k/self.den.k, [LinearTerm()]), Poly( 1.0, [])))

        return result

def eval_foster( s, f):
    result = 0*s
    for rf in f:
        result += rf.eval(s) 
    return result

def compare( f, Z):
    delta = 0.001
    for start, end in Z.intervals(delta):
        xxs = np.arange( start, end+delta/4, delta)
        ff = eval_foster(xxs, f)
        zz = Z.eval(xxs)
        assert np.allclose( ff, zz)
