import numpy as np

from signal_processing import U_

from signal_processing.extensions import pint_extension

from signal_processing.continuous.continuous_data_obj import ContinuousData

from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments

def test_ContinuousData():
    t = np.arange(10) * U_.sec
    vals = np.arange(10) * U_.volt
    sig = ContinuousData(vals, t)

    assert pint_extension.allclose(sig.domain_samples, t)
    assert pint_extension.allclose(sig.values, vals)
    assert sig.n_samples == 10
    assert sig.domain_start == 0 * U_.sec
    assert sig.domain_end == 9 * U_.sec
    assert sig.domain_range.is_close(Segment([0,9], U_.sec))
    assert sig.domain_unit == U_.sec
    assert sig.values_unit == U_.volt

    assert sig.domain_description == pint_extension.get_dimensionality_str(U_.sec)
    assert sig.values_description == pint_extension.get_dimensionality_str(U_.volt)
    sig.domain_description = "domain"
    assert sig.domain_description == "domain"
    sig.values_description = "values"
    assert sig.values_description == "values"
    sig.describe("domain_2", "values_2")
    assert sig.domain_description == "domain_2"
    assert sig.values_description == "values_2"

    expected_sig_str = "domain: " + "domain_2" + str(sig.domain_range) + "\n" + "values: " + "values_2"

    assert sig.is_close(sig)
    assert not sig.is_close(ContinuousData(vals, t + 1 * U_.sec))
    assert not sig.is_close(ContinuousData(vals + 1 * U_.volt, t))

    # ___getitem__
    t_range = Segment(np.array([2.5, 6.5]) * U_.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousData(vals[expected_slice], t[expected_slice])
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)

    t_ranges = Segments(np.array([2.5, 7.5]) * U_.sec, np.array([6.5, 8.5]) * U_.sec)
    sig_list = sig[t_ranges]
    ranges = t_ranges.to_segments_list()
    expected_sig_list = [sig[ranges[0]], sig[ranges[1]]]
    for i in range(len(sig_list)):
        assert sig_list[i].is_close(expected_sig_list[i])

def visual_test_plot():
    x = np.arange(10) * U_.sec
    y = np.arange(10) * U_.m
    sig = ContinuousData(y, x)
    sig.describe("times", "dist from source")
    sig.plot()
