import numpy as np

from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments
from signal_processing.segments import manipulate


from signal_processing import U_


def test_switch_segments_and_gaps():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    segments_0 = Segments(starts, ends)
    expected_gaps = Segments(np.array([1, 3, 5]), np.array([2, 4, 10]))
    gaps = manipulate.switch_segments_and_gaps(segments_0)
    assert gaps.is_close(expected_gaps)
    """
    starts = np.array([0, 2, 4, 10]) * U_.meter
    ends = np.array([1, 3, 5, 10.5]) * U_.meter
    segments = Segments(starts, ends)
    abs_start = -5 * U_.meter
    abs_end = 20 * U_.meter
    expected_gaps = Segments(np.array([-5, 1, 3, 5, 10.5]) * U_.meter, np.array([0, 2, 4, 10, 20]) * U_.meter)
    gaps = switch_segments_and_gaps(segments, abs_start, abs_end)
    print gaps.starts
    print gaps.ends
    print expected_gaps.starts
    print expected_gaps.ends
    assert gaps.is_close(expected_gaps)
    """


def test_concatenate():
    s_1 = Segments(np.array([1, 3]) * U_.meter, np.array([2, 4]) * U_.meter)
    s_2 = Segments(np.array([5, 7]) * U_.meter, np.array([6, 8]) * U_.meter)
    s_3 = Segments(np.array([1, 3, 5, 7]) * U_.meter, np.array([2, 4, 6, 8]) * U_.meter)
    assert s_3.is_close(manipulate.concatenate([s_1, s_2]))




#def test_concatecate_single_segments():
    #s = Segments(U_.meter * np.array([1, 3]), U_.meter * np.array([2,4]))
    #seg_list = [s[0], s[1]]
    #s_2 = manipulate.concatenate_single_segments(seg_list)
    #assert s.is_close(s_2)

    ## other test

    #s_list = [Segment([0, 1], U_.m), Segment([2, 3], U_.m)]
    #expected_segments = Segments(np.array([0, 2]) * U_.m, np.array([1, 3]) * U_.m)
    #segments = manipulate.concatenate_single_segments(s_list)
    #assert segments.is_close(expected_segments)

    #s_list = [Segment([0, 3], U_.m), Segment([1, 2], U_.m)]
    #expected_segments = Segments(np.array([0, ]) * U_.m, np.array([3,]) * U_.m)
    #segments = Segments.from_segments_list(s_list)
    #assert segments.is_close(expected_segments)



def test_filter_short_segments():
    starts = np.array([0, 2, 4, 10]) * U_.meter
    ends = np.array([1, 3, 5, 10.5]) * U_.meter
    segments_0 = Segments(starts, ends)
    min_duration = 0.75 * U_.meter
    only_long_segments_expected = Segments(np.array([0, 2, 4]) * U_.meter, np.array([1, 3, 5]) * U_.meter)
    only_long_segments = manipulate.filter_short_segments(segments_0, min_duration)
    assert only_long_segments.is_close(only_long_segments_expected)



