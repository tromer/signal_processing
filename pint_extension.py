# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:25:50 2014

@author: noam
"""
import warnings

import numpy as np
import scipy as sp
from scipy import signal
from global_uerg import uerg, Q_

#%%
#from . import ureg, Q_

def allclose(a, b, rtol=1e-5, atol=None):
    """
    it's an np.allclose version for vectors with units (of pint module)
    
    parameters:
    a, b, - vectors with units.
    rtol - relative tolerance
    atol - if not None, absolute tolerance (with units)
    if None, the absolute tolerance would be the default of np.allclose in the base units
    """
    
    # TODO: add assert a.dimensionality == b.dimensionality
    # TODO: make it fetch the units of a_, chnge docstring accordingly
    a_ = a.to_base_units()
    b_ = b.to_base_units()
    
    if not atol:
        return np.allclose(a_.magnitude, b_.magnitude, rtol)
        
    else:
        atol_ = atol.to_base_units()
        return np.allclose(a_.magnitude, b_.magnitude, rtol, atol_.magnitude)

def test_allclose():
    a = 3 * uerg.meter
    b = 3 * uerg.meter
    c = 3 * uerg.centimeter
    d = 300 * uerg.centimeter
    atol= 1 * uerg.centimeter
    assert allclose(a, b)
    assert allclose(a, d)
    assert not allclose(a, c)
    assert allclose(a, b, atol=atol)
    #TODO: check that using a different unit for atol raises exception.
    #TODO: add assertions. this is a very fondemental function.
    
test_allclose()
#%%

def get_units(x):
    """
    helper function to get 1 with the same units like x
    I didn't find any pint option to do that,
    and the implementatino is ugly, based on devision with the magnitude
    XXX
    """
    if type(x.magnitude) == np.ndarray:
        x_ = x[np.nonzero(x)]
        if len(x) == 0:
            raise NotImplementedError
        return 1.0 * x_[0] / x_[0].magnitude
    else:
        if x.magnitude == 0:
            raise NotImplementedError
        return 1.0 * x / x.magnitude
        
def test_get_units():
    x = 3 * uerg.meter
    assert allclose(get_units(x), 1 * uerg.meter)
    vec = np.arange(1, 5) * uerg.meter
    assert allclose(get_units(vec), uerg.meter)
    
test_get_units()
#%%
def units_list_to_ndarray(l):
    """
    takes a list / tuple of numbers with units, rescale them, and converts to
    a vector with units
    """
    assert len(l)
    unit = False
    i = 0
    while not unit and i < len(l):
        if l[i].magnitude != 0:
            unit = get_units(l[i])
        i = i + 1
    
    l_magnitude = []
    for x in l:
        l_magnitude.append(x.to(unit).magnitude)
        
    return np.array(l_magnitude) * unit
    
def test_units_list_to_ndarray():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    assert allclose(units_list_to_ndarray(l), np.array([1, 2, 1]) * uerg.meter)
    
test_units_list_to_ndarray()
    

#%%
   
def histogram(a, bins=10, range_=None, weights=None, density=None):
    """
    histogram for vectors with quantities.
    it's basically a wrap around np.histogram
    
    obviously a, range_ should have the same units, and also bins if it's not the 
    number of bins
    """
    # maybe should accept also Range object?
    if not type(a) == uerg.Quantity:
        raise NotImplementedError
        #return np.histogram(a, bins, range_, weights, density)
    else:
        base_units = get_units(a)
        a = a.magnitude
        if not type(bins) == int:
            bins = bins.to(base_units)
            bins = bins.magnitude
        
        if range_ != None:
            range_ = range_.to(base_units)
            range_ = range_.magnitude
            
        hist, edges = np.histogram(a, bins, range_, weights, density)
        return hist, edges * base_units

        
def test_histogram():
    a = (np.arange(10) + 0.5) * uerg.meter
    range_ = np.array([0, 10]) * uerg.meter
    expected_hist = np.ones(10)
    expected_edges = np.arange(11) * uerg.meter
    hist, edges = histogram(a, bins=10, range_=range_)
    assert np.allclose(hist, expected_hist)
    assert allclose(edges, expected_edges)
    
test_histogram()


def minimum(a, b):
    warnings.warn("not tested")
    unit = get_units(a)
    a = a.magnitude
    b = b.to(unit).magnitude
    return unit * np.minimum(a, b)
    
def maximum(a, b):
    warnings.warn("not tested")
    # copied from pint_extension.minimum
    unit = get_units(a)
    a = a.magnitude
    b = b.to(unit).magnitude
    return unit * np.maximum(a, b)
    

#%%


"""
TODO: making pint work well with matplotlib
Herlpers: matplotlib.units?
http://matplotlib.org/examples/units/basic_units.html
"""
