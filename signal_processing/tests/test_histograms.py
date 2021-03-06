import numpy as np
from signal_processing import U_
from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven
from signal_processing import histograms

def test_data_to_continuous_histogram():
    # copied from pint_extension.test_histogram
    a = (np.arange(10) + 0.5) * U_.meter
    range_ = np.array([0, 10]) * U_.meter
    expected_hist = np.ones(10) * U_.dimensionless
    # expected_edges = np.arange(11) * U_.meter
    expected_bin_size = U_.meter
    expected_first_bin = 0.5 * U_.meter
    expected_hist_continuous = ContinuousDataEven(expected_hist, expected_bin_size, expected_first_bin)
    hist_continuous = histograms.data_to_continuous_histogram(a, bins=10, range_=range_)
    assert hist_continuous.is_close(expected_hist_continuous)



