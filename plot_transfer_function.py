import numpy as np
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_transfer_function( n_k, n_a, d_k, d_a, xunits='rads/sec'):

  log_xs = np.arange( -3, 3.005, 0.01)
  xs = np.power( 10, log_xs)
  if xunits == 'Hz':
    ss = 2j * xs * np.pi
  elif xunits == 'rads/sec':
    ss = 1j * xs
  else:
    assert False, f'xunits ({xunits}) should be either "Hz" or "rads/sec".'

  def eval_poly( a, s):
    prod = np.ones( (len(s),), dtype=np.complex128)
    result = np.ones( (len(s),), dtype=np.complex128)*a[0]    
    for x in a[1:]:
      prod *= s
      result += prod*x
    return result

  points = (n_k*eval_poly( n_a, ss))/(d_k * eval_poly( d_a, ss))

  dBs = 20*np.log10( np.abs(points))
  degrees = np.angle(points,deg=True)

  if True:
      t1 = go.Scatter(x=xs,y=dBs,name="amplitude")
      t2 = go.Scatter(x=xs,y=degrees,name="phase",yaxis='y2')
      ly = go.Layout(
          title='Bode Plot',
          xaxis={'title': f'Frequency ({xunits})', 'type': 'log'},
          yaxis={'title': f'Amplitude (dB)'},
          yaxis2={'title': f'Phase (degrees)', 'overlaying':'y', 'side':'right'}
      )
      fig = go.Figure(data=[t1,t2], layout=ly)
      fig.show()

  if False:
      fig = make_subplots( rows=2, cols=1, xaxis_type="log", shared_xaxes=True, vertical_spacing=0.02)
      fig.add_trace(
          go.Scatter(x=xs,y=dBs),
          row=1, col=1
      )
      fig.add_trace(
          go.Scatter(x=xs,y=degrees),
          row=2, col=1
      )

      fig.update_layout( height=600, width=800, title_text="Bode Plot")
      fig.show()

  if False:
      plt.xscale( "log")
      plt.xlabel( f"freq ({xunits})")
      plt.ylabel( "amplitude (dB)")
      plt.plot( xs, dBs)
      plt.show()

      plt.xscale( "log")
      plt.xlabel( f"freq ({xunits})")
      plt.ylabel( "phase (degrees)")
      plt.plot( xs, degrees)
      plt.show()
