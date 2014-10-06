import numpy as np
import scipy as sp
from scipy import signal
from signal_processing.extensions import scipy_extension

def test_smart_convolve():
    v = np.arange(2 ** 10)
    m = np.arange(2 ** 5)
    assert np.allclose(np.convolve(v, m), scipy_extension.smart_convolve(v,m))
    assert np.allclose(sp.signal.fftconvolve(v, m), scipy_extension.smart_convolve(v,m))
    
test_smart_convolve()
