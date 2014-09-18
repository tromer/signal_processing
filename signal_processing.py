# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:31:11 2014

@author: noam
"""

import warnings

import numpy as np
import scipy as sp
from scipy import  signal
import matplotlib.pyplot as plt

import segments
from segments import Segments
import continuous_data
from continuous_data import ContinuousDataEven

import pint_extension
from global_uerg import uerg, Q_
#%%


"""
this is a signal processing package.
it's heavily based on numpy, scipy, and uses pint for units support.

Data types
---------------
the package introduces a few data types to represent different datas
easily and efficiently:
1. ContinuousData - every continuous data, like signals.
2. Segments - a quantisation / clustering / finding areas of interest
within signals

Processes
-------------------
various processes to manipulate ContinuousData and Segments:
frequency filtering, adjoining segments together, filtering by some properties.
automatic parameters finding for various processes (thresholds)

some conventions
---------------------------------
1. XXX is a marker of highly problematic process.
2. is_each_..... a name for a boolian array which specifies whether
each element in another array fullfils some condition. it's a mask.



"""

IS_DEBUG = False

def fast_convolve(sig, mask, mode):
    """
    XXX XXX this function was put at scipy_extension as a mathematical
    function
    determines which implementation of convolve to use, depending
    on the properties of the inputs. chooses the faster algorithm,
    between regular convolve, and fft-convolve
    
    parameters
    --------------------
    sig : ContinuousData
        signal
    mask : ContinuousData or np.ndarray
    TODO: choose the signature
    
    returns
    ------------
    convolved_signal : ContinuousData
    
    

    """
    raise NotImplementedError
    if case_regular:
        return np.convolve(sig, mask, mode)
    elif case_fft:
        return sp.signal.fftconvolve(sig, mask, mode)
        
#%%
def threshold_crosses(sig, threshold, is_above=True):
    """
    returns segments instance, where a given signal values are above
    a certain threshold.
    
    parameters
    --------------- 
    sig : ContinuousData
        a signal
    threshold : float with units like the values of sig
        TODO: for non-constant threshold, we may allow passing
        a vector of values, or a thrshold which is a signal
        TODO: allow passing a range of allowed thresholds
    is_above : bool
        whether the segments of interest are above the threshold
        
    returns
    -----------
    segments : Segments
    
    
    Note: this function assumes that the domain of the signal has
    a mathematical order defined upon it. this applies to most
    cases. exceptions are when the domain cells are modulu of
    something, or frequencies.
    if the domain is modulu derived / cyclic, it's still possible to
    implement. the "first" sample is "connected" to the last.
    we find the indexes of threshold crosses the same way,
    we have a segments instance. now we just have to decide whether
    it's the segments of interest, or the others are. we check a
    single value in one of the segments to determine that
    """
    above = sig.values > threshold
    if not is_above:
        above = np.logical_not(above)
    # the beginning and end count as non pulse
    crossings = np.logical_xor(np.concatenate([above, [False,]]), np.concatenate([[False], above]))
    crossings_indexes = np.where(crossings)[0]
    crossings_times = crossings_indexes * sig.sample_step + sig.first_sample
    starts = crossings_times[::2]
    ends = crossings_times[1::2]
    return Segments(starts, ends)
#%%
def test_threshold_crosses():
    sig = ContinuousDataEven(np.array([3, 3, 3, 0, 0, 0, 3, 3, 0]) * uerg.mamp, uerg.sec)
    threshold = 2 * uerg.mamp
    starts_expected = np.array([0, 6])
    ends_expected = np.array([3, 8])
    segments_expected = Segments(starts_expected, ends_expected)
    segments = threshold_crosses(sig, threshold)
    assert segments.is_close(segments_expected)
    

test_threshold_crosses()

#%%
def threshold_adjoin_filter_short_segments(sig, threshold, max_distance, min_duration):
    """
    concatanates 3 processes one after another:
    threshold, adjoin, filter_short_segments
    """
    warnings.warn("not tested")
    p = threshold_crosses(sig, threshold)
    # note that it's important to adjoin before filtering short segments
    p = segments.adjoin_close_segments(p, max_distance)
    p = segments.filter_short_segments(p, min_duration)
    return p

#%%
def data_to_continuous_histogram(a, bins=10, range_=None, weights=None, density=None):
    """
    returns a histogram of some data.
    it's a wrap aroud pint_extension.histogram
    
    returns:
    -------------
    hist_continuous : ContinuousData
        the histogram as a ContinuousData
    """
    if not type(bins) == int:
        raise NotImplementedError
        # reurning not evenly sampled
    else:
        hist, edges = pint_extension.histogram(a, bins, range_, weights, density)
        bin_size = 1.0 *(edges[-1] - edges[0]) / bins
        first_bin = edges[0] + 0.5 * bin_size
        hist = hist * uerg.dimensionless
        hist_continuous = ContinuousDataEven(hist, bin_size, first_bin)
        return hist_continuous
        
def test_data_to_continuous_histogram():
    # copied from pint_extension.test_histogram
    a = (np.arange(10) + 0.5) * uerg.meter
    range_ = np.array([0, 10]) * uerg.meter
    expected_hist = np.ones(10) * uerg.dimensionless
    # expected_edges = np.arange(11) * uerg.meter
    expected_bin_size = uerg.meter
    expected_first_bin = 0.5 * uerg.meter
    expected_hist_continuous = ContinuousDataEven(expected_hist, expected_bin_size, expected_first_bin)
    hist_continuous = data_to_continuous_histogram(a, bins=10, range_=range_)
    assert hist_continuous.is_close(expected_hist_continuous)
    
test_data_to_continuous_histogram()

#%%
def signal_values_hist(contin, bins=100, range_=None, weights=None, density=None):
    """
    returns the histogram of the values of a ContinuousData
    a wrap around data_to_continuous_histogram
    
    returns:
    -----------
    hist : ContinuousData
    
    TODO: maybe this should be a method of ContinuousData
    """
    warnings.warn("signal_value_hist is not tested")
    hist = data_to_continuous_histogram(contin.values, bins, range_, weights, density)
    return hist


def segments_attribute_hist(segments, attribute, bins=100, range_=None, weights=None, density=None):
    """
    TODO: test, maybe should be in module segments
    """
    warnings.warn("not tested")
    hist = data_to_continuous_histogram(getattr(segments, attribute), bins, range_, weights, density)
    return hist


def mark_threshold(fig, threshold, colors="y", label="threshold"):
    """
    
    """
    warnings.warn("bad behaviour of units, just strips them, not tested")
    plt.figure(fig.number)
    x_min, x_max = plt.xlim()
    threshlines = plt.hlines(threshold.magnitude, x_min, x_max, colors=colors, label=label)
    plt.legend(loc='best')
    return threshlines

#%%
def cluster1d(vec, resolution, threshold):
    """
    find main values in histigram. should be probably implemented
    like finding segments in every ContinuousData
    the only difference is defining a resolution here
    """
    raise NotImplementedError
    bins_num = np.ceil(1.0 * vec.ptp() / resolution)
    hist, edges = pint_extension.histogram(vec, bins_num, density=True)
    clusters = threshold_crosses(vec, 1, threshold)
    return clusters
