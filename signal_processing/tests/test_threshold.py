import numpy as np

from signal_processing.extensions import pint_extension

from signal_processing.continuous.continuous_data_even_obj import \
    ContinuousDataEven
from signal_processing.segments.segments_obj import Segments
from signal_processing.segments.segments_of_continuous_obj import \
    SegmentsOfContinuous

from signal_processing import U_

from signal_processing import threshold


def test_crosses():
    sig = ContinuousDataEven(np.array([3, 3, 3, 0, 0, 0, 3, 3, 0]) * U_.mamp,
                             U_.sec)
    threshold_0 = 2 * U_.mamp
    starts_expected = np.array([0, 6])
    ends_expected = np.array([3, 8])
    segments_expected = Segments(starts_expected, ends_expected, U_.sec)
    segments_and_sig_expected = SegmentsOfContinuous(segments_expected, sig)
    segments_and_sig = threshold.crosses(sig, threshold_0)
    assert pint_extension.allclose(segments_and_sig.starts,
                                   segments_and_sig_expected.starts)
    assert pint_extension.allclose(segments_and_sig.ends,
                                   segments_and_sig_expected.ends)
    assert segments_and_sig.is_close(segments_and_sig_expected)




