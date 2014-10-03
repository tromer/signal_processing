# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 17:57:02 2014

@author: noam
"""

import numpy as np
import scipy as sp
from scipy import signal

import numpy_extension

def smart_convolve(vec, mask, mode='full'):
    """
    determines which implementation of convolve to use, depending
    on the properties of the inputs. chooses the faster algorithm,
    between regular convolve, and fft-convolve.
    
    I assume that the vec is longer then the mask.
    
    parameters
    --------------------
    sig : np.ndarray
    mask : np.ndarray
    mode : like in np.convolve
        
    returns
    ------------
    convolved_signal : np.ndarray
    
    TODO: optimize the parametrs of decision.
    I checked a little bit.
    

    """
    
    case_short_mask = len(mask) <= 8
    case_not_power_of_2 = not numpy_extension.is_power_of_2(len(vec))
    case_naive = case_short_mask or case_not_power_of_2
    case_fft = not case_naive
    
    if case_naive:
        return np.convolve(vec, mask, mode)
    elif case_fft:
        return sp.signal.fftconvolve(vec, mask, mode)
        
        
def test_smart_convolve():
    v = np.arange(2 ** 10)
    m = np.arange(2 ** 5)
    assert np.allclose(np.convolve(v, m), smart_convolve(v,m))
    assert np.allclose(sp.signal.fftconvolve(v, m), smart_convolve(v,m))
    
test_smart_convolve()