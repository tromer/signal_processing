# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:14:13 2014

@author: noam
"""

import numpy as np
from segment import Segment
from global_uerg import uerg
#%%
class Segments(object):
    """
    Note: see also the object ContinuousData. they go hand in hand together, refer to different aspects of the same subjects
    this class represents any kind of segments / segments / ranges / containers (one dimensional).
    It includes a few kinds that first seem different from each other, has a lot in common.
    
    different logical "types" of Segments
    ------------------------------------------------------------
    1. Segments that are derived from a ContinuousData
    it's a ContinuousData that was clusterred / qunatised
    in a way, it describes some aspect of the ContinuousData, but with less data,
    so it's a "dimensionallity reduction"
    
    it may be used to calculate some properties of each Segment, that are some kind of summation
    of all the samples within the Segment
    such as: mean amplitude, max amplitude, total number of occurances
    
    2. Segments that are "containers", used to filter / sort / mark same samples of
    an existing ContinuousData
    
    Note: it's quite probable that these "types" would be other objects which inherit Segments
    Note: it should be possible to "extract containers" from Segments based on data
    
    
    examples (corresponding to the examples in ContinuousData):
    1. segments - times of interest in a corresponding signal.
    such as: when the amplitude is above a certain threshold.
    2. spatial segments: locations of interest in a coresponding spatial measurement.
    such as: locations of mountains, or locations of downhill ares.
    another example: the spatial measurement is stress as a function of place.
    the spatial segments can be places that will probably brake.
    3. ranges: certain "containers" of interest upon a distribution
    such as: ranges of 'height' wich are common in the population.
    4. times of interest, when a certain kinematic property of a system had special values
    5. frequency ranges, that are frequencies of interest within a spectrum.
    
    Note:
    2. this class is intentioned to be used with units. it's real world measurements.
    
    """
    def __init__(self, starts, ends):
        """
        parameters:
        ---------------------
        
        """
        # or np.recarray, or pandas.DataFrame
        self._starts = starts
        self._ends = ends
        # global_start, global_end?
        # pointer to internal data?
        
    def __len__(self):
        # maybe should be the length in time?
        return len(self.starts)
    
    @property
    def starts(self):
        return self._starts
        
    @property
    def ends(self):
        return self._ends
        
    @property
    def durations(self):
        return self.ends - self.starts
        
    @property
    def start_to_start(self):
        return np.diff(self.starts)
        
    @property
    def end_to_end(self):
        raise NotImplementedError
        
    @property
    def end_to_start(self):
        return self.starts[1:] - self.ends[:-1]
        
    def __getitem__(self, key):
        return Segments(self.starts[key], self.ends[key])
        
    def is_close(self, other, rtol=1e-05, atol=1e-08):
        """
        using np.allclose
        
        Returns
        -------------
        allclose : bool
            whether two different Segments are more or less the same properties
            
        TODO: check that atol works well enough with units.
        """
        if len(self) != len(other):
            return False
        return np.allclose(self.starts, other.starts, rtol, atol) and np.allclose(self.ends, other.ends, rtol, atol)
        
    def is_each_in_range(self, attribute, range_):
        """
        checks whether some attribute of each of the segments is within a certain range
        
        parameters
        -------------------------
        attribute : str
            the attribute of Segments we need
        range_ : Segment
            the range of interest
            
        returns
        ---------------
        is_each_in : np.ndarray
            a boolian np.ndarray
        """
        values = getattr(self, attribute)
        is_each_in = range_.is_each_in(values)
        return is_each_in
        
    def filter_by_range(self, attribute, range_, mode='include'):
        """
        checks whether some attribute of each of the segments is within a certain range
        filter out Segments that are out of range
        see documentation of Segments.is_each_in
        
        parameters
        ---------------------
        mode str
            'include' (leave segments in range), 'remove' - remove pulses in range
            
        returns
        ----------
        filterred: Segments
            only the Segments within range
        """
        assert mode in ['include', 'remove']
        is_each_in = self.is_each_in_range(attribute, range_)
        if mode == 'remove':
            is_each_in = np.logical_not(is_each_in)
        
        return self[is_each_in]
        
    def _is_each_in_many_values(self, x):
        raise NotImplementedError
        
    def _is_each_in_many_segments(self, x):
        raise NotImplementedError
        
    def is_each_in(self, x):
        """
        returns
        ------------
        is_each_in : np.ndarray
            a boolian array, for each value of x, whether it's
            included in one of the segments
        """
        raise NotImplementedError

def test_segments():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    durations = ends - starts
    start_to_start = np.diff(starts)
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
    assert np.allclose(starts[is_each_in], p[is_each_in].starts)
    assert np.allclose(starts[s], p[s].starts)
    assert p.is_close(p)
    
def test_is_each_in_range():
    starts = np.array([0, 2, 4, 10]) * uerg.sec
    ends = np.array([1, 3, 5, 10.5]) * uerg.sec
    p = Segments(starts, ends)
    
    duration_range = Segment([0.8, 1.2], uerg.sec)
    expected_is_each_in = np.array([True, True, True, False])
    is_each_in = p.is_each_in_range('durations', duration_range)
    assert np.allclose(is_each_in, expected_is_each_in)

def test_filter_by_range():
    # copied from test_is_each_in_range
    starts = np.array([0, 2, 4, 10]) * uerg.sec
    ends = np.array([1, 3, 5, 10.5]) * uerg.sec
    p = Segments(starts, ends)
    
    duration_range = Segment([0.8, 1.2], uerg.sec)
    expected_is_each_in = np.array([True, True, True, False])
    expected_p_filterred = p[expected_is_each_in]
    p_filterred = p.filter_by_range('durations', duration_range)
    assert p_filterred.is_close(expected_p_filterred)
    
    
test_segments()
test_is_each_in_range()
test_filter_by_range()

#%%
def switch_segments_and_gaps(pulses, absolute_start=None, absolute_end=None):
    """
    returns the gaps between the segments as a Segments instance
    rational: sometimes it's easier to understand what are the segments which dosn't
    interest us, and then switch
    
    parameters
    --------------
    segs : Segments
    
    absolute_start, absolute_end : of the same unit like segments.starts
        if given, they represent the edges of the "signal", and thus
        create another "gap-segment" at the start / end.
        
    returns
    ---------------
    gaps: Segments
        the gaps
    
    """
    # maybe absolute start and end should be taken from the segments object?
    starts_gaps = segments.ends[:-1]
    ends_gaps = segments.starts[1:]
    if absolute_start:
        starts_gaps = np.concatenate([np.ones(1) * absolute_start, starts_gaps])
    if absolute_end:
        ends_gaps = np.concatenate([ends_gaps, np.ones(1) * absolute_end])
    
    return Segments(starts_gaps, ends_gaps)
#%%
def test_switch_segments_and_gaps():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    segments = Segments(starts, ends)
    expected_gaps = Segments(np.array([1, 3, 5]), np.array([2, 4, 10]))
    gaps = switch_segments_and_gaps(pulses)
    assert gaps.is_close(expected_gaps)
    

test_switch_segments_and_gaps()

#%%
def adjoin_close_segments(pulses, max_distance):
    """
    if the segments are close enough, maybe they represent the same segment of interest,
    that was "broken" due to noise / wring threshold / mistake
    TODO: determine smartly max_distance, width of segments?
    TODO: iterative process
    """
    is_each_gap_big_enough = segments.end_to_start > max_distance
    is_each_real_start = np.concatenate([[True,], is_each_gap_big_enough])
    is_each_real_end = np.concatenate([is_each_gap_big_enough, [True,]])
    return Segments(segments.starts[is_each_real_start], pulses.ends[is_each_real_end])
    
    """
    another approach is: raw_signal -> threshold -> convolve with mask=np.ones(n, dtype=bool)
    then xoring with a shift to find ends and starts, then trim the edges
    """
#%%
def test_adjoin_close_segments():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 11])
    segments = Segments(starts, ends)
    max_distance = 2
    adjoined_segments_expected = Segments(np.array([0, 10]), np.array([5, 11]))
    adjoined_segments = adjoin_close_pulses(pulses, max_distance)
    assert adjoined_segments.is_close(adjoined_pulses_expected)
    
    
test_adjoin_close_segments()
#%%
def filter_short_segments(pulses, min_duration):
    """
    TODO: it should be based on filter_by_range
    """
    is_each_long_enough = segments.durations > min_duration
    return segments[is_each_long_enough]
#%%    
def test_filter_short_segments():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    segments = Segments(starts, ends)
    min_duration = 0.75
    only_long_segments_expected = Segments(np.array([0, 2, 4]), np.array([1, 3, 5]))
    only_long_segments = filter_short_pulses(pulses, min_duration)
    assert only_long_segments.is_close(only_long_pulses_expected)
    
    
test_filter_short_segments()

#%%
def plot_quick(segments):
    raise NotImplementedError
    
