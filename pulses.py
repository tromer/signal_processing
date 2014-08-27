# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:14:13 2014

@author: noam
"""

import numpy as np
#%%
class Pulses:
    def __init__(self, starts, ends):
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
    def end_to_start(self):
        return self.starts[1:] - self.ends[:-1]
        
    def __getitem__(self, key):
        return Pulses(self.starts[key], self.ends[key])
        
    def close(self, other, rtol=1e-05, atol=1e-08):
        # maybe should be names is_close ?
        if len(self) != len(other):
            return False
        return np.allclose(self.starts, other.starts, rtol, atol) and np.allclose(self.ends, other.ends, rtol, atol)
#%%
def test_pulses():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    durations = ends - starts
    start_to_start = np.diff(starts)
    end_to_start = starts[1:] - ends[:-1]
    is_each_in = np.array([True, True, False, False])
    s = slice(1, 3)

    p = Pulses(starts, ends)
    assert np.allclose(starts, p.starts)
    assert len(p) == len(starts)
    assert np.allclose(ends, p.ends)
    assert np.allclose(durations, p.durations)
    assert np.allclose(start_to_start, p.start_to_start)
    assert np.allclose(end_to_start, p.end_to_start)
    assert np.allclose(starts[is_each_in], p[is_each_in].starts)
    assert np.allclose(starts[s], p[s].starts)
    assert p.close(p)
#%%
test_pulses()


#%%
def adjoin_close_pulses(pulses, max_distance):
    is_each_gap_big_enough = pulses.end_to_start > max_distance
    is_each_real_start = np.concatenate([[True,], is_each_gap_big_enough])
    is_each_real_end = np.concatenate([is_each_gap_big_enough, [True,]])
    return Pulses(pulses.starts[is_each_real_start], pulses.ends[is_each_real_end])
    
    """
    another approach is: raw_signal -> threshold -> convolve with mask=np.ones(n, dtype=bool)
    then xoring with a shift to find ends and starts, then trim the edges
    """
#%%
def test_adjoin_close_pulses():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 11])
    pulses = Pulses(starts, ends)
    max_distance = 2
    adjoined_pulses_expected = Pulses(np.array([0, 10]), np.array([5, 11]))
    adjoined_pulses = adjoin_close_pulses(pulses, max_distance)
    assert adjoined_pulses.close(adjoined_pulses_expected)
    
    
test_adjoin_close_pulses()
#%%
def filter_short_pulses(pulses, min_duration):
    is_each_long_enough = pulses.durations > min_duration
    return pulses[is_each_long_enough]
#%%    
def test_filter_short_pulses():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    pulses = Pulses(starts, ends)
    min_duration = 0.75
    only_long_pulses_expected = Pulses(np.array([0, 2, 4]), np.array([1, 3, 5]))
    only_long_pulses = filter_short_pulses(pulses, min_duration)
    assert only_long_pulses.close(only_long_pulses_expected)
    
    
test_filter_short_pulses()
#%%
def switch_pulses_and_gaps(pulses, absolute_start=None, absolute_end=None):
    # maybe absolute start and end should be taken from the pulses object?
    starts_gaps = pulses.ends[:-1]
    ends_gaps = pulses.starts[1:]
    if absolute_start:
        starts_gaps = np.concatenate([np.ones(1) * absolute_start, starts_gaps])
    if absolute_end:
        ends_gaps = np.concatenate([ends_gaps, np.ones(1) * absolute_end])
    
    return pulses(starts_gaps, ends_gaps)
#%%
def test_switch_pulses_and_gaps():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    expected_starts_gaps = np.array([1, 3, 5])
    expected_ends_gaps = np.array([2, 4, 10])
    starts_gaps, ends_gaps = switch_pulses_and_gaps(starts, ends)
    assert np.allclose(starts_gaps, expected_starts_gaps)
    assert np.allclose(ends_gaps, expected_ends_gaps)

test_switch_pulses_and_gaps()
    