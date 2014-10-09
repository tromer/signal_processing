import numpy as np

from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven
from signal_processing.segments.segments_obj import Segments

from signal_processing import uerg

from signal_processing import threshold

def test_crosses():
    sig = ContinuousDataEven(np.array([3, 3, 3, 0, 0, 0, 3, 3, 0]) * uerg.mamp, uerg.sec)
    threshold_0 = 2 * uerg.mamp
    starts_expected = np.array([0, 6])
    ends_expected = np.array([3, 8])
    segments_expected = Segments(starts_expected, ends_expected)
    segments = threshold.crosses(sig, threshold_0)
    assert segments.is_close(segments_expected)
    



