def test_threshold_crosses():
    sig = ContinuousDataEven(np.array([3, 3, 3, 0, 0, 0, 3, 3, 0]) * uerg.mamp, uerg.sec)
    threshold = 2 * uerg.mamp
    starts_expected = np.array([0, 6])
    ends_expected = np.array([3, 8])
    segments_expected = Segments(starts_expected, ends_expected)
    segments = threshold_crosses(sig, threshold)
    assert segments.is_close(segments_expected)
    

test_threshold_crosses()


