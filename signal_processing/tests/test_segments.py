import numpy as np
from signal_processing import uerg
from signal_processing.segment import Segment
from signal_processing.segments import Segments

from signal_processing import segments


def test_segments():
    starts = np.array([0, 2, 4, 10]) * uerg.sec
    ends = np.array([1, 3, 5, 10.5]) * uerg.sec
    durations = ends - starts
    start_to_start = starts[1:] - starts[:-1]
    end_to_start = starts[1:] - ends[:-1]
    is_each_in = np.array([True, True, False, False])
    s = slice(1, 3)

    p = Segments(starts, ends)
    assert np.allclose(starts, p.starts)
    assert len(p) == len(starts)
    assert np.allclose(ends, p.ends)
    assert np.allclose(durations, p.durations)
    assert np.allclose(start_to_start, p.start_to_start)
    assert np.allclose(end_to_start, p.end_to_start)
    # __getitem__
    assert np.allclose(starts[is_each_in], p[is_each_in].starts)
    assert np.allclose(starts[s], p[s].starts)
    """
    print p
    print p.starts
    print p.starts[0]
    print p[0]
    print type(p[0])
    """
    assert p[0].is_close(Segment([0,1], uerg.sec))
    assert p[1].is_close(Segment([2,3], uerg.sec))
    assert p[-1].is_close(Segment([10,10.5], uerg.sec))
    
    assert p.is_close(p)
    
def test_is_each_in_range():
    starts = np.array([0, 2, 4, 10]) * uerg.sec
    ends = np.array([1, 3, 5, 10.5]) * uerg.sec
    p = Segments(starts, ends)
    
    duration_range = Segment([0.8, 1.2], uerg.sec)
    expected_is_each_in = np.array([True, True, True, False])
    is_each_in = p.is_each_in_range('durations', duration_range)
    assert np.allclose(is_each_in, expected_is_each_in)

def test_filter_by_range():
    # copied from test_is_each_in_range
    starts = np.array([0, 2, 4, 10]) * uerg.sec
    ends = np.array([1, 3, 5, 10.5]) * uerg.sec
    p = Segments(starts, ends)
    
    duration_range = Segment([0.8, 1.2], uerg.sec)
    expected_is_each_in = np.array([True, True, True, False])
    expected_p_filterred = p[expected_is_each_in]
    p_filterred = p.filter_by_range('durations', duration_range)
    assert p_filterred.is_close(expected_p_filterred)
    

def test_is_single_segment():
    starts = np.array([0,]) * uerg.sec
    ends = np.array([1,]) * uerg.sec
    p = Segments(starts, ends)
    assert p.is_single_segment()
    
    starts = np.array([0, 2]) * uerg.sec
    ends = np.array([1, 3]) * uerg.sec
    p = Segments(starts, ends)
    assert not p.is_single_segment()
    
def test_to_single_segment():
    starts = np.array([0,]) * uerg.sec
    ends = np.array([1,]) * uerg.sec
    p = Segments(starts, ends)
    expected_single_segment = Segment([0,1], uerg.sec)
    single_segment = p.to_single_segment()
    assert single_segment.is_close(expected_single_segment)
    

def test_is_empty():
    s = Segments(np.array([]) * uerg.m, np.array([]) * uerg.m)
    assert s.is_empty()
    
    starts = np.array([0,]) * uerg.sec
    ends = np.array([1,]) * uerg.sec
    p = Segments(starts, ends)
    assert not p.is_empty()    
    
def test_to_segments_list():
    starts = np.array([0, 2]) * uerg.sec
    ends = np.array([1, 3]) * uerg.sec
    p = Segments(starts, ends)
    l = p.to_segments_list()
    expected_l = [Segment([0, 1], uerg.sec), Segment([2, 3], uerg.sec)]
    for i in range(len(l)):
        assert l[i].is_close(expected_l[i])
    
test_segments()
test_is_each_in_range()
test_filter_by_range()
test_is_single_segment()
test_to_single_segment()
test_is_empty()
test_to_segments_list()

#################
def test_filter_short_segments():
    starts = np.array([0, 2, 4, 10]) * uerg.meter
    ends = np.array([1, 3, 5, 10.5]) * uerg.meter
    segments_0 = Segments(starts, ends)
    min_duration = 0.75 * uerg.meter
    only_long_segments_expected = Segments(np.array([0, 2, 4]) * uerg.meter, np.array([1, 3, 5]) * uerg.meter)
    only_long_segments = segments.filter_short_segments(segments_0, min_duration)
    assert only_long_segments.is_close(only_long_segments_expected)
    
    
test_filter_short_segments()

def test_switch_segments_and_gaps():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    segments_0 = Segments(starts, ends)
    expected_gaps = Segments(np.array([1, 3, 5]), np.array([2, 4, 10]))
    gaps = segments.switch_segments_and_gaps(segments_0)
    assert gaps.is_close(expected_gaps)
    """
    starts = np.array([0, 2, 4, 10]) * uerg.meter
    ends = np.array([1, 3, 5, 10.5]) * uerg.meter
    segments = Segments(starts, ends)
    abs_start = -5 * uerg.meter
    abs_end = 20 * uerg.meter
    expected_gaps = Segments(np.array([-5, 1, 3, 5, 10.5]) * uerg.meter, np.array([0, 2, 4, 10, 20]) * uerg.meter)
    gaps = switch_segments_and_gaps(segments, abs_start, abs_end)
    print gaps.starts
    print gaps.ends
    print expected_gaps.starts
    print expected_gaps.ends
    assert gaps.is_close(expected_gaps)
    """
test_switch_segments_and_gaps()

def test_adjoin_segments_max_distance():
    starts = np.array([0, 2, 4, 10]) * uerg.meter
    ends = np.array([1, 3, 5, 11]) * uerg.meter
    segments_0 = Segments(starts, ends)
    max_distance = 2 * uerg.meter
    adjoined_segments_expected = Segments(np.array([0, 10]) * uerg.meter, np.array([5, 11]) * uerg.meter)
    adjoined_segments = segments.adjoin_segments_max_distance(segments_0, max_distance)
    assert adjoined_segments.is_close(adjoined_segments_expected)
    
    
test_adjoin_segments_max_distance()
#%%
def test_adjoin_segments_considering_durations():
    # copied from test_adjoin_segments_max_distance
    starts = np.array([0, 2, 4, 10]) * uerg.meter
    ends = np.array([1, 3, 5, 11]) * uerg.meter
    segments_0 = Segments(starts, ends)
    
    ratio = 1.2
    adjoined_segments_expected = Segments(np.array([0, 10]) * uerg.meter, np.array([5, 11]) * uerg.meter)
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio)
    assert adjoined_segments.is_close(adjoined_segments_expected)
    
    ratio = 0.8
    adjoined_segments_expected = segments_0
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio)
    assert adjoined_segments.is_close(adjoined_segments_expected)
    
    ratio = 1.2
    max_dist = 0.8 * uerg.meter
    adjoined_segments_expected = segments_0
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio, max_dist)
    assert adjoined_segments.is_close(adjoined_segments_expected)

def test_adjoin_segments_considering_durations_mode_min():
    starts = np.array([0, 2]) * uerg.meter
    ends = np.array([1, 2.1]) * uerg.meter
    segments_0 = Segments(starts, ends)    

    ratio = 1.2
    adjoined_segments_expected = segments_0
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)
    
    starts = np.array([0, 1]) * uerg.meter
    ends = np.array([0.1, 2]) * uerg.meter
    segments_0 = Segments(starts, ends)    

    ratio = 1.2
    adjoined_segments_expected = segments_0
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)

    starts = np.array([0, 2]) * uerg.meter
    ends = np.array([1, 3]) * uerg.meter
    segments_0 = Segments(starts, ends)    

    ratio = 1.2
    adjoined_segments_expected = Segments(np.array([0,]) * uerg.meter, np.array([3,]) * uerg.meter)
    adjoined_segments = segments.adjoin_segments_considering_durations(segments_0, ratio, mode='min')
    assert adjoined_segments.is_close(adjoined_segments_expected)

    
test_adjoin_segments_considering_durations()
test_adjoin_segments_considering_durations_mode_min()

#
def test_from_segments_list():
    s_list = [Segment([0, 1], uerg.m), Segment([2, 3], uerg.m)]
    expected_segments = Segments(np.array([0, 2]) * uerg.m, np.array([1, 3]) * uerg.m)
    segments = Segments.from_segments_list(s_list)
    assert segments.is_close(expected_segments)
    
    s_list = [Segment([0, 3], uerg.m), Segment([1, 2], uerg.m)]
    expected_segments = Segments(np.array([0, ]) * uerg.m, np.array([3,]) * uerg.m)
    segments = Segments.from_segments_list(s_list)
    assert segments.is_close(expected_segments)    
    
#test_from_segments_list()
def test_concatenate():
    s_1 = Segments(np.array([1, 3]) * uerg.meter, np.array([2, 4]) * uerg.meter)
    s_2 = Segments(np.array([5, 7]) * uerg.meter, np.array([6, 8]) * uerg.meter)
    s_3 = Segments(np.array([1, 3, 5, 7]) * uerg.meter, np.array([2, 4, 6, 8]) * uerg.meter)
    assert s_3.is_close(segments.concatenate([s_1, s_2]))
    
test_concatenate()

def test_concatecate_single_segments():
    s = Segments(uerg.meter * np.array([1, 3]), uerg.meter * np.array([2,4]))
    seg_list = [s[0], s[1]]
    s_2 = segments.concatenate_single_segments(seg_list)
    assert s.is_close(s_2)
    
test_concatecate_single_segments()

def test_from_single_segment():
    s = Segment([2, 3], uerg.meter)
    expected_segments = Segments(np.array([2,]) * uerg.meter, np.array([3,]))
    segments = from_single_segment(s)
    assert segments.is_close(expected_segments)

test_from_single_segment()
