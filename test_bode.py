
from plot_transfer_function import plot_transfer_function

def test_A():
    n_k = d_k = 1
    n_a = [1,0]
    d_a = [1,1,1]
    plot_transfer_function( n_k, n_a, d_k, d_a)

def test_B():
    n_k = d_k = 1
    n_a = [1,0]
    d_a = [1,0.1,1]
    plot_transfer_function( n_k, n_a, d_k, d_a, xunits='Hz')
    
