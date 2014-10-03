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
import numpy_extension
from global_uerg import uerg, Q_


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


