import numpy as np

from signal_processing import uerg
from signal_processing.extensions import pint_extension
from signal_processing.segment import Segment
        
def test_Segment():
    segment_1 = Segment(np.array([3, 5]) * uerg.meter)
    assert pint_extension.allclose(segment_1.start, 3 * uerg.meter)
    assert pint_extension.allclose(segment_1.end, 5 * uerg.meter)
    assert pint_extension.allclose(segment_1.edges, np.array([3,5]) * uerg.meter)
    assert pint_extension.allclose(segment_1.center, 4 *uerg.meter)
    assert pint_extension.allclose(segment_1.width, 2 * uerg.meter)
    assert pint_extension.allclose(segment_1.width_half, 1 * uerg.meter)
    
    #print len(segment_1)
    #assert pint_extension.allclose(len(segment_1) , 2 * uerg.meter)
    assert 4 * uerg.meter in segment_1
    assert not 2 * uerg.meter in segment_1
    assert np.allclose(np.array([True, True]), segment_1.is_each_in(np.array([4, 4]) * uerg.meter))
    
    assert segment_1.is_close(segment_1)
    segment_2 = Segment(np.array([3, 4]) * uerg.meter)
    assert not segment_1.is_close(segment_2)
    
    assert segment_1.is_close(Segment((3 * uerg.meter, 5 * uerg.meter)))
    assert segment_1.is_close(Segment((3, 5), uerg.meter))
    
    assert Segment([0, 1 * uerg.meter]).is_close(Segment([0, 1], uerg.meter))

    segment_3 = Segment(np.array([2.5, 6.5]) * uerg.sec)

def test_from_center():
    segment_1 = Segment(np.array([3, 5]) * uerg.meter)
    segment_2 = Segment.from_center(4, 1, uerg.meter)
    segment_3 = Segment.from_center(4, 2, uerg.meter, mode='width')
    segment_4 = Segment.from_center(4, 1, uerg.meter, mode='width')
    assert segment_1.is_close(segment_2)
    assert segment_1.is_close(segment_3)
    assert not segment_1.is_close(segment_4)
    
