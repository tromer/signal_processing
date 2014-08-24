# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 17:49:27 2014

@author: noam
"""

import numpy as np

def smaller_power_of_2(n):
    if type(n) == np.ndarray:
        assert (n >= 1).all()
    else:
        assert n >= 1
    the_power = np.floor(np.log(n) / np.log(2))
    if type(n) == np.ndarray:
        return np.array(2 ** the_power, dtype=int)
    else:
        return int(2 ** the_power)
    
def test_smaller_power_of_2():
    print "testing"
    inputs = np.array([1, 1.5, 2, 3, 4, 10, 20])
    expeced_outputs = np.array([1, 1, 2, 2, 4, 8, 16])
    assert np.allclose(smaller_power_of_2(inputs), expeced_outputs)
    assert smaller_power_of_2(30) == 16
    
 #%%
    
test_smaller_power_of_2()