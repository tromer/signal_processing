# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 02:57:11 2014

@author: noam
"""
import numpy as np
from .extensions import pint_extension
from global_uerg import uerg, Q_
import operator
#%%


class Segment(object):
    """
    a range in a certain domain. such as 3 to 5 meters.
    it's built with units, (using pint module)
    TODO: maybe it should be renamed Container / Segment
    """
    def __init__(self, edges, unit=None):
        """
        parameters:
        ----------------
        edges : list / tuple / np.ndarray of length 2
            can have units.
            a leading zero can be without units, so Segments beginning
            in 0 are easy to make
        units : pint.Quantity
            the units in case edges is without units
        
        """
        # protecting the data
        edges = edges[:]
        
        assert len(edges) == 2
        
        if hasattr(edges[1], 'units'):
            # case leading 0
            if edges[0] == 0 and not hasattr(edges[0], 'units'):
                edges[0] = edges[0] * pint_extension.get_units(edges[1])
            edges = pint_extension.rescale_all(edges)
        else:
            edges = np.array(edges) * unit
                
        self._edges = edges #should take care of case where it's a toople. mind units!
        
    @classmethod
    def from_center(cls, center, deviation, unit=None, mode='half_width'):
        """
        construct a Segment based on center value, and deviation (half width)
        
        http://coding.derkeiler.com/Archive/Python/comp.lang.python/2005-02/1294.html
        http://stackoverflow.com/a/682545
        
        parameters:
        -----------------
        mode : str
            'half_width', default
            'width'
        """
        
        if mode == 'half_width':
            half_width = deviation
        elif mode == 'width':
            half_width = 0.5 * deviation
        
        edges = [center - half_width, center + half_width]
        return cls(edges, unit)
        
        
        
    @property
    def start(self):
        return self._edges[0]
        
    @property
    def end(self):
        return self._edges[1]
        
    @property
    def center(self):
        return 0.5 * (self.start + self.end)
        
    @property
    def edges(self):
        return self._edges
        
    @property
    def width(self):
        return self.end - self.start
        
    @property
    def width_half(self):
        return 0.5 * self.width
    
    """   
    def __len__(self):
    # there is some problem. it seems that len(x) and x.__len__() is not the same
        return self.end - self.start
    """
        
    def __contains__(self, x):
        return np.logical_and(x > self.start, x < self.end)
        
    def is_each_in(self, x):
        return self.__contains__(x)
        
    def is_close(self, other, rtol=1e-5, atol=None):
        return pint_extension.allclose(self.edges, other.edges, rtol, atol)
        
    def shift(self, delta):
        """
        returns:
        ----------
        shifted : Segment
            the segment shifted by delta
        """
        raise NotImplementedError
        
        
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

def test_from_center():
    segment_1 = Segment(np.array([3, 5]) * uerg.meter)
    segment_2 = Segment.from_center(4, 1, uerg.meter)
    segment_3 = Segment.from_center(4, 2, uerg.meter, mode='width')
    segment_4 = Segment.from_center(4, 1, uerg.meter, mode='width')
    assert segment_1.is_close(segment_2)
    assert segment_1.is_close(segment_3)
    assert not segment_1.is_close(segment_4)
    
test_Segment()
test_from_center()
