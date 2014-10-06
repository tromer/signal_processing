import numpy as np

from signal_processing import uerg

from signal_processing.extensions import pint_extension

from signal_processing.continuous.continuous_data_obj import ContinuousData

from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments

def test_ContinuousData():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    assert pint_extension.allclose(sig.domain_samples, t)
    assert pint_extension.allclose(sig.values, vals)
    assert sig.n_samples == 10
    assert sig.first_sample == 0 * uerg.sec
    assert sig.last_sample == 9 * uerg.sec
    
    assert sig.is_close(sig)
    assert not sig.is_close(ContinuousData(vals, t + 1 * uerg.sec))
    assert not sig.is_close(ContinuousData(vals + 1 * uerg.volt, t))

    # ___getitem__
    t_range = Segment(np.array([2.5, 6.5]) * uerg.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousData(vals[expected_slice], t[expected_slice])
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)
    
    t_ranges = Segments(np.array([2.5, 7.5]) * uerg.sec, np.array([6.5, 8.5]) * uerg.sec)
    sig_list = sig[t_ranges]
    ranges = t_ranges.to_segments_list()
    expected_sig_list = [sig[ranges[0]], sig[ranges[1]]]
    for i in range(len(sig_list)):
        assert sig_list[i].is_close(expected_sig_list[i])
    
test_ContinuousData()

