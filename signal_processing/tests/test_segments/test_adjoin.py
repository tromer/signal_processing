import numpy as np


from signal_processing.segments.segments_obj import Segments
from signal_processing.segments import adjoin


from signal_processing import U_


def test_max_dist():
    starts = np.array([0, 2, 4, 10]) * U_.meter
    ends = np.array([1, 3, 5, 11]) * U_.meter
    segments_0 = Segments(starts, ends)
    max_distance = 2 * U_.meter
    adjoined_segments_expected = Segments(np.array([0, 10]) * U_.meter, np.array([5, 11]) * U_.meter)
    adjoined_segments = adjoin.max_dist(segments_0, max_distance)
    assert adjoined_segments.is_close(adjoined_segments_expected)


#%%
def test_consider_duration():
    # copied from test_adjoin_segments_max_distance
    starts = np.array([0, 2, 4, 10]) * U_.meter
    ends = np.array([1, 3, 5, 11]) * U_.meter
    segments_0 = Segments(starts, ends)

    ratio = 1.2
    adjoined_segments_expected = Segments(np.array([0, 10]) * U_.meter, np.array([5, 11]) * U_.meter)
    adjoined_segments = adjoin.consider_duration(segments_0, ratio)
    assert adjoined_segments.is_close(adjoined_segments_expected)

    ratio = 0.8
    adjoined_segments_expected = segments_0
    adjoined_segments = adjoin.consider_duration(segments_0, ratio)
    assert adjoined_segments.is_close(adjoined_segments_expected)

    ratio = 1.2
    max_dist = 0.8 * U_.meter
    adjoined_segments_expected = segments_0
    adjoined_segments = adjoin.consider_duration(segments_0, ratio, max_dist)
    assert adjoined_segments.is_close(adjoined_segments_expected)

def test_consider_duration_mode_min():
    starts = np.array([0, 2]) * U_.meter
    ends = np.array([1, 2.1]) * U_.meter
    segments_0 = Segments(starts, ends)

    ratio = 1.2
    adjoined_segments_expected = segments_0
    adjoined_segments = adjoin.consider_duration(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)

    starts = np.array([0, 1]) * U_.meter
    ends = np.array([0.1, 2]) * U_.meter
    segments_0 = Segments(starts, ends)

    ratio = 1.2
    adjoined_segments_expected = segments_0
    adjoined_segments = adjoin.consider_duration(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)

    starts = np.array([0, 2]) * U_.meter
    ends = np.array([1, 3]) * U_.meter
    segments_0 = Segments(starts, ends)

    ratio = 1.2
    adjoined_segments_expected = Segments(np.array([0,]) * U_.meter, np.array([3,]) * U_.meter)
    adjoined_segments = adjoin.consider_duration(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)
