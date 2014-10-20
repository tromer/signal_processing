
"""
.. module:: histograms
    :synopsis: uses data to create histogram as a ContinuousData object.

Manipulation of histograms is just manipulation of ContinuousData or
ContinuousDataEven instances.


refactor
-----------
change the names of functions here to not include the word hist (it's already
in the name of the module)

"""


import warnings


from continuous.continuous_data_even_obj import ContinuousDataEven

from .extensions import pint_extension

from signal_processing import U_


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
        hist = hist * U_.dimensionless
        hist_continuous = ContinuousDataEven(hist, bin_size, first_bin)
        return hist_continuous

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


