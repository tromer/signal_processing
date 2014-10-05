# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:49:27 2014

@author: noam
"""

import numpy as np

def is_power_of_2(n):
    """
    checks whether a number is a power of 2
    
    returns:
    bool
    """
    if n < 0:
        raise ValueError

    return int(np.log2(n)) == np.log2(n)

   
def close_power_of_2(n, mode='smaller'):
    """
    find the a close power of 2 of an int.
    useful when doing some FFTs
    parameters:
    n - int
    mode - 'smaller', 'bigger', 'closer'
    """
    if type(n) == np.ndarray:
        assert (n >= 1).all()
    else:
        assert n >= 1
    
    the_power = np.log(n) / np.log(2)
    if mode == 'smaller':
        the_power = np.floor(the_power)
    elif mode == 'bigger':
        the_power = np.ceil(the_power)
    elif mode == 'closer':
        the_power = np.round(the_power)
    
    if type(n) == np.ndarray:
        return np.array(2 ** the_power, dtype=int)
    else:
        return int(2 ** the_power)
    
   
#%%
def normalize(vec, ord=None, axis=None):
    """
    normalize a vector according to a chosen order
    usful when creating masks for convolutions
    
    parameters:
    vec - np.ndarray
    ord - order of norm
    axis - axis to work on
    """
    return 1.0 * vec / np.linalg.norm(vec, ord, axis)
    
def deviation_from_reference(vec, ref):
    """
    like standard deviation, but not from mean, but from an arbitrary value.
    for example, when trying to fit data to half a gaussian, centerred at zero
    
    TODO: idea for possible noise robust algorithm:
    center : use median instead of mean
    deviation : use median of delta_from_center ** 2 instead of mean
    """
    return np.sqrt(np.mean((vec - ref) ** 2))
    
#%%
def running_max(vec, m):
    """ to implement in C """
    raise NotImplementedError

def running_median(vec, m):
     """ to implement in C """
     raise NotImplementedError

def running_quantile(vec, m, ratio):
    """ to implement in C """
    raise NotImplementedError
