
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot( f, tag=''):
  xs = 10**(np.arange( -3, 3.005, .01))     
  t1 = go.Scatter(x=xs,y=f(xs),name="amplitude")
  ly = go.Layout(
      title=f'Plot {tag}',
      xaxis={'title': 'Frequency (rads/sec)', 'type': 'log'},
      yaxis={'title': tag}
  )
  fig = go.Figure(data=[t1], layout=ly)
  fig.show()


def plot_real_part( f):
  xs = 10**(np.arange( -3, 3.005, .01))     
  t1 = go.Scatter(x=xs,y=f(xs*1j).real,name="amplitude")
  ly = go.Layout(
      title='Real Part Plot',
      xaxis={'title': f'Frequency (rads/sec)', 'type': 'log'},
      yaxis={'title': f'Real part'}
  )
  fig = go.Figure(data=[t1], layout=ly)
  fig.show()


class Cascade:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def __str__(self):
        return f"A,B,C,D: {self.A},{self.B},{self.C},{self.D}"

    def hit( self, other):
        # A B     A B
        # C D     C D
        return Cascade( self.A*other.A + self.B*other.C,
                        self.A*other.B + self.B*other.D,
                        self.C*other.A + self.D*other.C,
                        self.C*other.B + self.D*other.D)

    def terminate( self, r):
        """Returns an impedence"""
        # V1 = A V2 + B I2
        # I1 = C V2 + D I2
        # V2 = r I2
        # V1 = A r I2 + B I2 = I2 ( A r + B )
        # I1 = C r I2 + D I2 = I2 ( C r + D )
        # V1 / ( A r + B ) = I1 / ( C r + D )
        # V1 / I1 = ( A r + B ) / ( C r + D )
        return (self.A*r + self.B) / (self.C*r + self.D)

    def terminate_with_admittance( self, g):
        """Returns an admittance"""
        # V1 = A V2 + B I2
        # I1 = C V2 + D I2
        # I2 = g V2
        # V1 = A V2 + B g V2 = V2 ( A + B g )
        # I1 = C V2 + D g V2 = V2 ( C + D g )
        # V1 / ( A + B g ) = I1 / ( C + D g )
        # I1 / V1 = ( C + D g ) / ( A + B g ) 
        return (self.C + self.D*g) / (self.A + self.B*g)

    @classmethod
    def Series(cls, r):
        return cls( 1, r, 0, 1)

    @classmethod
    def Shunt(cls, g):
        return cls( 1, 0, g, 1)

