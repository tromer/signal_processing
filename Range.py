# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 02:57:11 2014

@author: noam
"""
import numpy as np
from pint_extension import allclose
from global_uerg import uerg


class Range(object):
    """
    a range in a certain domain. such as 3 to 5 meters
    """
    def __init__(self, edges):
        assert len(edges) == 2
        self._edges = edges #should take care of case where it's a toople. mind units!
        
    @property
    def start(self):
        return self._edges[0]
        
    @property
    def end(self):
        return self._edges[1]
        
    @property
    def edges(self):
        return self._edges
        
    @property
    def width(self):
        raise NotImplementedError
        return self.end - self.start
    
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
        return allclose(self.edges, other.edges, rtol, atol)
        
        
def test_Range():
    range_1 = Range(np.array([3, 5]) * uerg.meter)
    assert allclose(range_1.start, 3 * uerg.meter)
    assert allclose(range_1.end, 5 * uerg.meter)
    assert allclose(range_1.edges, np.array([3,5]) * uerg.meter)
    #print len(range_1)
    #assert allclose(len(range_1) , 2 * uerg.meter)
    assert 4 * uerg.meter in range_1
    assert not 2 * uerg.meter in range_1
    assert np.allclose(np.array([True, True]), range_1.is_each_in(np.array([4, 4]) * uerg.meter))
    
    assert range_1.is_close(range_1)
    range_2 = Range(np.array([3, 4]) * uerg.meter)
    assert not range_1.is_close(range_2)
    
    
test_Range()