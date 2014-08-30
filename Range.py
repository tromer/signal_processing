# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 02:57:11 2014

@author: noam
"""
import numpy as np

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
        
    def __contains__(self, x):
        return np.logical_and(x > self.start, x < self.end)