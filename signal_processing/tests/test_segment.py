import numpy as np

from signal_processing import U_
from signal_processing.extensions import pint_extension
from signal_processing.segment import Segment

def test_Segment():
    segment_1 = Segment(np.array([3, 5]) * U_.meter)
    assert pint_extension.allclose(segment_1.start, 3 * U_.meter)
    assert pint_extension.allclose(segment_1.end, 5 * U_.meter)
    assert pint_extension.allclose(segment_1.edges, np.array([3,5]) * U_.meter)
    assert str(segment_1) == str(segment_1.edges)

    assert pint_extension.allclose(segment_1.center, 4 *U_.meter)
    assert pint_extension.allclose(segment_1.width, 2 * U_.meter)
    assert pint_extension.allclose(segment_1.width_half, 1 * U_.meter)

    #print len(segment_1)
    #assert pint_extension.allclose(len(segment_1) , 2 * U_.meter)
    assert 4 * U_.meter in segment_1
    assert not 2 * U_.meter in segment_1
    assert np.allclose(np.array([True, True]), segment_1.is_each_in(np.array([4, 4]) * U_.meter))

    assert segment_1.is_close(segment_1)
    segment_2 = Segment(np.array([3, 4]) * U_.meter)
    assert not segment_1.is_close(segment_2)

    assert segment_1.is_close(Segment((3 * U_.meter, 5 * U_.meter)))
    assert segment_1.is_close(Segment((3, 5), U_.meter))

    assert Segment([0, 1 * U_.meter]).is_close(Segment([0, 1], U_.meter))

    segment_3 = Segment(np.array([2.5, 6.5]) * U_.sec)

def test_from_center():
    segment_1 = Segment(np.array([3, 5]) * U_.meter)
    segment_2 = Segment.from_center(4, 1, U_.meter)
    segment_3 = Segment.from_center(4, 2, U_.meter, mode='width')
    segment_4 = Segment.from_center(4, 1, U_.meter, mode='width')
    assert segment_1.is_close(segment_2)
    assert segment_1.is_close(segment_3)
    assert not segment_1.is_close(segment_4)

