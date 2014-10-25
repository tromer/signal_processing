import numpy as np

from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments


from signal_processing import U_



def test_segments():
    starts = np.array([0, 2, 4, 10]) * U_.sec
    ends = np.array([1, 3, 5, 10.5]) * U_.sec
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

    # other ways to create segments
    p_2 = Segments([0, 2, 4, 10], [1, 3, 5, 10.5], unit=U_.sec)
    assert p.is_close(p_2)
    """
    print p
    print p.starts
    print p.starts[0]
    print p[0]
    print type(p[0])
    """
    assert p[0].is_close(Segment([0,1], U_.sec))
    assert p[1].is_close(Segment([2,3], U_.sec))
    assert p[-1].is_close(Segment([10,10.5], U_.sec))

    assert p.is_close(p)

def test_is_each_in_range():
    starts = np.array([0, 2, 4, 10]) * U_.sec
    ends = np.array([1, 3, 5, 10.5]) * U_.sec
    p = Segments(starts, ends)

    duration_range = Segment([0.8, 1.2], U_.sec)
    expected_is_each_in = np.array([True, True, True, False])
    is_each_in = p.is_each_in_range('durations', duration_range)
    assert np.allclose(is_each_in, expected_is_each_in)

def test_filter_by_range():
    # copied from test_is_each_in_range
    starts = np.array([0, 2, 4, 10]) * U_.sec
    ends = np.array([1, 3, 5, 10.5]) * U_.sec
    p = Segments(starts, ends)

    duration_range = Segment([0.8, 1.2], U_.sec)
    expected_is_each_in = np.array([True, True, True, False])
    expected_p_filterred = p[expected_is_each_in]
    p_filterred = p.filter_by_range('durations', duration_range)
    assert p_filterred.is_close(expected_p_filterred)


def test_is_single_segment():
    starts = np.array([0,]) * U_.sec
    ends = np.array([1,]) * U_.sec
    p = Segments(starts, ends)
    assert p.is_single_segment()

    starts = np.array([0, 2]) * U_.sec
    ends = np.array([1, 3]) * U_.sec
    p = Segments(starts, ends)
    assert not p.is_single_segment()

def test_to_single_segment():
    starts = np.array([0,]) * U_.sec
    ends = np.array([1,]) * U_.sec
    p = Segments(starts, ends)
    expected_single_segment = Segment([0,1], U_.sec)
    single_segment = p.to_single_segment()
    assert single_segment.is_close(expected_single_segment)


def test_is_empty():
    s = Segments(np.array([]) * U_.m, np.array([]) * U_.m)
    assert s.is_empty()

    starts = np.array([0,]) * U_.sec
    ends = np.array([1,]) * U_.sec
    p = Segments(starts, ends)
    assert not p.is_empty()

def test_to_segments_list():
    starts = np.array([0, 2]) * U_.sec
    ends = np.array([1, 3]) * U_.sec
    p = Segments(starts, ends)
    l = p.to_segments_list()
    expected_l = [Segment([0, 1], U_.sec), Segment([2, 3], U_.sec)]
    for i in range(len(l)):
        assert l[i].is_close(expected_l[i])



def test_from_single_segment():
    s = Segment([2, 3], U_.meter)
    expected_segments = Segments([2,], [3,], U_.meter)
    segments = Segments.from_single_segment(s)
    assert segments.is_close(expected_segments)

