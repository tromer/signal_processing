# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 17:25:50 2014

@author: noam
"""
import warnings

import numpy as np
import scipy as sp
from scipy import signal

from signal_processing import U_, Q_

from signal_processing.extensions import numpy_extension

# basic functions, compare, rescale, and so
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

def get_units(x):
    """
    helper function to get 1 with the same units like x
    """

    """
    I didn't find any pint option to do that,
    and the implementatino is ugly, based on devision with the magnitude
    XXX
    """

    """
    old old old old
    if type(x.magnitude) == np.ndarray:
        x_ = x[np.nonzero(x)]
        if len(x) == 0:
            raise NotImplementedError
        return 1.0 * x_[0] / x_[0].magnitude
    else:
        if x.magnitude == 0:
            raise NotImplementedError
        return 1.0 * x / x.magnitude
    """
    return Q_(1.0, x.units)

def units_list_to_ndarray(l):
    """
    takes a list / tuple of numbers with units, rescale them, and converts to
    a vector with units
    """
    warnings.warn("deprecated, use rescale_all")
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

def rescale_all(l, unit=None):
    """
    rescales objects to the same unit
    such as a list of vectors with units

    note
    ----------
    there is no need for a specific case for arrays with units, because arrays with units have the same unit in the first place.

    see also
    -----------
    strip_units
    array
    """
    if unit != None:
        raise NotImplementedError

    if unit == None:
        unit = get_units(l[0])

    for v in l:
        if not v.dimensionality == unit.dimensionality:
            raise ValueError("not same dimensionality")

    scaled = map(lambda v : v.to(unit), l)
    return scaled

def strip_units(l, unit=None):
    """
    returns:
    -----------
    mag : list of vectors without untis

    unit : the unit
    """
    if type(l) == np.ndarray:
        mag, unit =  l, U_.dimensionless

   # case it's an array
    elif type(l) == U_.Quantity:
        mag, unit =  l.magnitude, get_units(l)

    else:
        scaled = rescale_all(l, unit)
        unit = get_units(scaled[0])
        mag = map(lambda(v) : v.magnitude, scaled)

    return mag, unit


def array(vec):
    """
    vec or list with units -> array with units
    """
    mag, unit = strip_units(vec)
    return unit * np.array(mag)

# non mathematical manipulations
def concatenate(vec_list):
    """
    returns:
    -------------
    concatenated vectors with rescaled units
    """
    mag, unit = strip_units(vec_list)
    return unit * np.concatenate(mag)

# some functions that are connected to plotting and presentations

def get_dimensionality_str(unit):
    """
    mostly to present 1/time as frequency

    parameters
    -------------
    unit : U_.Quantity

    returns
    ----------
    dim_str : str
    """
    dim_str = str(unit.dimensionality)

    if dim_str == str((U_.Hz).dimensionality):
        dim_str = "[frequency]"
    elif dim_str == str((U_.volt).dimensionality):
        dim_str = "[voltage]"
    return dim_str



def get_units_beautiful_str(unit):
    """
    presents the units nicely with brackets
    if dimensionless, returns [AU]

    parameters
    -------------
    unit : U_.Quantity

    returns
    ------------
    units_str : str

    """
    warnings.warn("not tested")

    ARBITRARY_UNITS_STR = "[AU]"
    if unit.unitless:
        units_str = ARBITRARY_UNITS_STR
    # here comes an ugly solution to printing "Hz" instead of 1 / second
    elif unit.units == (1.0 / U_.sec).units:
        units_str = "[Hz]"
    elif unit.units == (1.0 / U_.msec).units:
        units_str = "[kHz]"
    elif unit.units == (1.0 / U_.usec).units:
        units_str = "[MHz]"
    else:
        units_str = "".join(["[", str(unit.units), "]"])

    return units_str


def prepare_data_and_labels_for_plot(x, y, x_description='', curv_description=''):
    """
    parameters
    -------------
    x : U_.Quantity

    y : U_.Quantity

    x_description : str
        defualt ''

    curv_description : str
        default ''

    refactor
    ------------
    maybe if x_description, curv_description are not given, the function
    should put their dimensionalities as descriptions

    maybe create the same function for single axis, let this function use it two times, (for each axis)
    """
    x_bare, x_unit = strip_units(x)
    y_bare, y_unit = strip_units(y)

    x_label = x_description + " " + get_units_beautiful_str(x_unit)
    # TODO copied last line
    curv_label = curv_description + " " + get_units_beautiful_str(y_unit)

    return x_bare, y_bare, x_label, curv_label


# mathematical calcualations and manipulations
def histogram(a, bins=10, range_=None, weights=None, density=None):
    """
    histogram for vectors with quantities.
    it's basically a wrap around np.histogram

    obviously a, range_ should have the same units, and also bins if it's not the
    number of bins
    """
    # maybe should accept also Range object?
    if not type(a) == U_.Quantity:
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


def median(vec):
    return get_units(vec) * np.median(vec.magnitude)


def fft(vec, mode='accurate', n_fft=None):
    """
    fft for a vector with units
    a wrap arround np.fft.fft

    parameters:
    ----------------
   mode : str
        copied from determine_n_fft
        'accurate' like n
        'trim' - smaller then n
        'zero-pad' - bigger then n
        'closer' - either trim or zero pad, depends which is closer (logarithmic scale)

    n_fft : int
        number of samples for fft


    returns
    ---------
    vector after fft and np.fft.fftshift
    """
    n_fft = numpy_extension.determine_n_fft(len(vec), mode, n_fft)

    spectrum_values_no_units = np.fft.fftshift(np.fft.fft(vec.magnitude, n_fft))
    spectrum_values_with_units = spectrum_values_no_units * get_units(vec)

    return spectrum_values_with_units

def hilbert(vec, mode='accurate', n_fft=None):
    """

    """
    warnings.warn("not tested")
    n_fft = numpy_extension.determine_n_fft(len(vec), mode, n_fft)
    analytic_vec = sp.signal.hilbert(vec.magnitude, n_fft) * get_units(vec)
    return analytic_vec

