# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:49:27 2014

@author: noam
"""

import numpy as np

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
    
def test_close_power_of_2():
    print "testing"
    inputs = np.array([1, 1.6, 2, 3, 4, 10, 20])
    expeced_outputs_smaller = np.array([1, 1, 2, 2, 4, 8, 16])
    expeced_outputs_bigger = np.array([1, 2, 2, 4, 4, 16, 32])
    expeced_outputs_closer = np.array([1, 2, 2, 4, 4, 8, 16])
    assert np.allclose(close_power_of_2(inputs, 'smaller'), expeced_outputs_smaller)
    assert np.allclose(close_power_of_2(inputs, 'bigger'), expeced_outputs_bigger)
    assert np.allclose(close_power_of_2(inputs, 'closer'), expeced_outputs_closer)
    assert close_power_of_2(30) == 16
    
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
    
def test_normalize():
    vec = np.array([1, 1])
    vec_n_1 = vec / 2.0
    assert np.allclose(normalize(vec, ord=1), vec_n_1)
    vec_n_2 = vec / np.sqrt(2)
    assert np.allclose(normalize(vec), vec_n_2)
    
    
    
test_close_power_of_2()
test_normalize()

#%%
def running_max(vec, m):
    """ to implement in C """
    return