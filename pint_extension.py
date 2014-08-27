# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:25:50 2014

@author: noam
"""

import numpy as np
import scipy as sp
from scipy import signal

#from . import ureg, Q_

def get_units(x):
    if type(x.magnitude) == np.ndarray:
        return 1.0 * x[0] / x[0].magnitude
    else:
        return 1.0 * x / x.magnitude
#%%

def magnitude_of_dimensionless(vec):
    assert vec.dimensionless
    return vec.magnitude
    
#%%
    
def test_magnitude_of_dimensionless():
    raise NotImplementedError

@uerg.wraps(None, (None, uerg.Hz, uerg.Hz, None, None, None, uerg.Hz))    
def firwin_pint(numtaps, cutoff, width, window, pass_zero, scale, nyq):
    return sp.signal.firwin(numtaps, cutoff, width, window, pass_zero, scale, nyq)
#%%
    
def histogram(a, bins=10, range_=None, weights=None, density=None):
    if not type(a) == pint.unit.Quantity:
        return np.histogram(a, bins, range_, weights, density)
    else:
        base_units = get_units(a)
        a = a.magnitude
        if not type(bins) == int:
            bins = bins.to(base_units)
            bins = bins.magnitude
        if range_:
            range_ = range_.to(base_units)
            range_ = range_.magnitude
       
        hist, edges = np.histogram(a, bins, range_, weights, density)
        return hist, edges * base_units