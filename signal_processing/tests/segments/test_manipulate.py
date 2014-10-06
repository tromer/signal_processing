
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


def test_concatenate():
    s_1 = Segments(np.array([1, 3]) * uerg.meter, np.array([2, 4]) * uerg.meter)
    s_2 = Segments(np.array([5, 7]) * uerg.meter, np.array([6, 8]) * uerg.meter)
    s_3 = Segments(np.array([1, 3, 5, 7]) * uerg.meter, np.array([2, 4, 6, 8]) * uerg.meter)
    assert s_3.is_close(segments.concatenate([s_1, s_2]))
    
test_concatenate()




def test_filter_short_segments():
    starts = np.array([0, 2, 4, 10]) * uerg.meter
    ends = np.array([1, 3, 5, 10.5]) * uerg.meter
    segments_0 = Segments(starts, ends)
    min_duration = 0.75 * uerg.meter
    only_long_segments_expected = Segments(np.array([0, 2, 4]) * uerg.meter, np.array([1, 3, 5]) * uerg.meter)
    only_long_segments = segments.filter_short_segments(segments_0, min_duration)
    assert only_long_segments.is_close(only_long_segments_expected)
    
    
test_filter_short_segments()

