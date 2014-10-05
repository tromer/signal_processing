# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:14:13 2014

@author: noam
"""

import warnings
#%%

from .extensions import pint_extension
import numpy as np
from segment import Segment
from signal_processing import uerg

#%%
import matplotlib.pyplot as plt
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
    such as:
    * mean amplitude
    * max amplitude
    * total number of occurances (histogram) / total energy (pulse)
    
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
    
    
    TODO: add methods, or functions for unifying segments, for intersection,
        for subtruction. so on. it's natural. It may make it easier to remove
        unwanted segments..
        maybe this should be only for "container type"
        
    TODO: maybe allow the definition of a segment with +inf or -inf as edge.
        probably only for container type
        
    TODO: add tests that refer to segments with units. it's not tested well enough
    
    TODO: add interface of starts, ends, unit=None
    """
    def __init__(self, starts, ends):
        """
        parameters:
        ---------------------
        
        """
        # or np.recarray, or pandas.DataFrame
        """ TODO: make this gaurdian work
        if len(starts) != len(ends):
            raise ValueError("ends and starts mush have same len")
        if not ((starts[1:] - starts[:-1]) > 0).magnitude.all():
            warnings.warn("the segments are not strictly one after another")
        if not ((ends[1:] - ends[:-1]) > 0).magnitude.all():
            warnings.warn("the segments are not strictly one after another")
        """
        
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
    def centers(self):
        raise NotImplementedError
        return 0.5 * (self.starts + self.ends)
    
    @property
    def durations(self):
        return self.ends - self.starts
        
    @property
    def start_to_start(self):
        # return np.diff(self.starts) # not working because of units
        return self.starts[1:] - self.starts[:-1]
        
    @property
    def end_to_end(self):
        raise NotImplementedError
        
    @property
    def end_to_start(self):
        return self.starts[1:] - self.ends[:-1]
        
    def __getitem__(self, key):
        if type(key) == int:
            return Segment([self.starts[key], self.ends[key]])
        elif type(key) in [type(slice(0,1)), np.ndarray]:
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
            Note: range_ could also be another Segments instance,
            with domain with the same units like self.attribute
            it could be any object with the method is_each_in
            
        returns
        ---------------
        is_each_in : np.ndarray
            a boolian np.ndarray
        """
        assert hasattr(range_, 'is_each_in')
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
        mode : str
            'include' (leave segments in range), 'remove' - remove segments in range
            
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
        
    def is_single_segment(self):
        return len(self) == 1
        
    def to_single_segment(self):
        if self.is_single_segment():
            return Segment([self.starts[0], self.ends[0]])
        else:
            raise ValueError("cannot convert to single segment")
            
    def is_empty(self):
        """
        maybe there are no segments at all?
        """
        return len(self) == 0

    def to_segments_list(self):
        """
        returns:
        ------------
        a list of segment instances
        """
        return map(Segment, zip(self.starts, self.ends))

    def tofile(self, f):
        """
        read doc of fromfile
        """
        raise NotImplementedError
    
#%%
def filter_short_segments(segments, min_duration):
    """
    TODO: it should be based on filter_by_range
    """
    return segments.filter_by_range('durations', Segment([0, min_duration]), mode='remove')
#%%    
#%%
def switch_segments_and_gaps(segments, absolute_start=None, absolute_end=None):
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
        raise NotImplementedError
        starts_gaps = np.concatenate([np.ones(1) * absolute_start, starts_gaps])
    if absolute_end:
        raise NotImplementedError
        ends_gaps = np.concatenate([ends_gaps, np.ones(1) * absolute_end])
    
    return Segments(starts_gaps, ends_gaps)
#%%
#%%
def adjoin_segments_max_distance(segments, max_distance):
    """
    if the segments are close enough, maybe they represent the same segment of interest,
    that was "broken" due to noise / wring threshold / mistake

    parameters:
    ----------------
    segments : Segments
    max_distance : float with units like segments' domain.
    
    TODO: determine smartly max_distance, width of segments?
    TODO: iterative process
    IMPROVE: not a max distance - but a range of distances.
        thus it's possible to not allow some unwanted distances
    """
    assert segments.starts.dimensionality == max_distance.dimensionality
    
    is_each_gap_big_enough = segments.end_to_start > max_distance
    is_each_real_start = np.concatenate([[True,], is_each_gap_big_enough])
    is_each_real_end = np.concatenate([is_each_gap_big_enough, [True,]])
    return Segments(segments.starts[is_each_real_start], segments.ends[is_each_real_end])
    
    """
    another approach is: raw_signal -> threshold -> convolve with mask=np.ones(n, dtype=bool)
    then xoring with a shift to find ends and starts, then trim the edges
    """
#%%
def adjoin_segments_considering_durations(segments, segment_gap_ratio, absolute_max_dist=None, mode='mean'):
    """
    to determine whether to adjoin two nearby segments, we consider their durations and the gap duration.
    we calculate a reference_duration for each gap, for comparison
    
    parameters:
    -----------
    segments : Segments
    
    segment_gap_ratio : float
        positive
        the ratio between the segment duration and max gap

    absolute_max_dist : float with units like segments' domain.
        when the segments are very small, we want a big "reach",
        so the segments stick together. when they are big,
        we want to prevent them from sticking all together.
    
    mode : str
      the reference_duration is determined by the durations before and after the gap.
      'mean'
      'min'
      'max'
      
      
    returns:
    ---------
    adjoined_segments : Segments
    
    TODO:
    ----------
    an obvious problem with 'min' mode - when there are two big pulses close, and another small one in the middle.
    """
    assert segment_gap_ratio > 0
    if hasattr(segment_gap_ratio, 'dimensionality'):
        assert segment_gap_ratio.dimensionality == uerg.dimensionless.dimensionality
    
    durations = segments.durations
    
    if mode == 'mean':
        reference_duration_for_each_gap = 0.5 * (durations[:-1] + durations[1:])
    elif mode == 'min':
        reference_duration_for_each_gap = pint_extension.minimum(durations[:-1], durations[1:])
    elif mode == 'max':
        raise NotImplementedError
        reference_duration_for_each_gap = pint_extension.maximum(durations[:-1], durations[1:])
        
    max_distance_due_to_duration = reference_duration_for_each_gap * segment_gap_ratio
    
    if absolute_max_dist != None:
        assert absolute_max_dist.dimensionality == segments.starts.dimensionality
        max_distance = pint_extension.minimum(max_distance_due_to_duration, absolute_max_dist)
    else:
        max_distance = max_distance_due_to_duration
    
    adjoined_segments = adjoin_segments_max_distance(segments, max_distance)
    return adjoined_segments
    
def adjoin_segments(segments, delta=0, ratio=0, max_dist=None, n=1):
    """
    parameters:
    ----------------
    n : int
        number of iterations
    """
    warnings.warn("adjoin_segments is not tested")
    if delta != 0: 
        assert delta.dimensionality == segments.starts.dimensionality
    if max_dist != None:
        assert max_dist.dimensionality == segments.starts.dimensionality
    

    adjoined_segments = segments
    for i in xrange(n):
        if delta != 0:
            adjoined_segments = adjoin_segments_max_distance(adjoined_segments, delta)
        if ratio != 0:
            adjoined_segments = adjoin_segments_considering_durations(adjoined_segments, ratio, max_dist)
            
    return adjoined_segments
    
#%%
    
def mark_starts_ends(segments, fig, color_start='r', color_end='g'):
    """
    get a figure and plot on it vertical lines according to
    starts and ends
    
    returns:
    -----------
    lines_start
    lines_end
    """
    warnings.warn("mark_starts_ends not tested, careful with units")
    plt.figure(fig.number)
    y_min, y_max = plt.ylim()
    starts_lines = plt.vlines(segments.starts, y_min, y_max, colors=color_start, label='starts')
    ends_lines = plt.vlines(segments.ends, y_min, y_max, colors=color_end, label='ends')
    plt.legend(loc='best')
    return starts_lines, ends_lines
    
def plot_quick(segments):
    raise NotImplementedError


def from_segments_list(segments_list):
    """
    assumes 
    XXXXXXXX XXX
    """
    raise NotImplementedError
    sorted_by_start = sorted(segments_list, key=lambda s : s.start)
    starts = pint_extension.array(map(lambda s : s.start, sorted_by_start))
    ends = pint_extension.array(map(lambda s : s.end, sorted_by_start))
    segments_maybe_overlap = Segments(starts, ends)
    segments = adjoin_segments_max_distance(segments_maybe_overlap, max_distance=0 * pint_extension.get_units(segments_maybe_overlap.starts))
    return segments
    
   
    
    

def fromfile(f):
    """
    reads Segments instance from file
    TODO
    --------
    decide which format. probably csv, what about the units?
    put them in the headers?
    """
    raise NotImplementedError


#%%
def concatenate(segments_list):
    """
    concatenates segments, if they are all one after another
    TOOD: test when there are empty segments
    """
    # filter out empty segments instances
    segments_list = filter(None, segments_list)
    
    for i in xrange(len(segments_list) - 1):
        if segments_list[i].ends[-1] > segments_list[i + 1].starts[0]:
            raise ValueError("not in order")
    

        
    all_starts = map(lambda segments : segments.starts, segments_list)
    all_ends = map(lambda segments : segments.ends, segments_list)
    return Segments(pint_extension.concatenate(all_starts), pint_extension.concatenate(all_ends))
    

def from_single_segment(segment):
    """
    returns:
    -------------
    segments : Segments
        containing only one segment
    """
    return Segments(pint_extension.array([segment.start,]), pint_extension.array([segment.end,]))
        
    
def test_from_single_segment():
    s = Segment([2, 3], uerg.meter)
    expected_segments = Segments(np.array([2,]) * uerg.meter, np.array([3,]))
    segments = from_single_segment(s)
    assert segments.is_close(expected_segments)

test_from_single_segment()


def concatenate_single_segments(segs_list):
    """
    have to be one after another
    """
    as_segments = map(from_single_segment, segs_list)
    return concatenate(as_segments)
    


    


class SegmentsOfContinuous(Segments):
    """
    this object represents segments of interest of some continuous data.
    for the documentation refer the Segments object.
    it's main purpose is to enable easy access to the "inside" of each segment.
    for example: calculate the mean / max amplitude of a pulse, calculate the total size of a peak in a histogram
    
    design issue
    -----------------------
    it's natural to write functions that process the internal of each segment (each pulse)
    obviously these functions would need the information about both the location of the segments, and the continuous data within.
    now two different concepts can be chosen:
    a) create an object that inherits from Segments, and is actually Segments with richer information.
    b) create an object that holds a segments object, and continuous data object, and used only for calculations of each semgnet.
    
    the decision so far - I try to think about what the object DOES rather then HAS. So in a way it's just a debate of internal
    implementation. I chose the first (Flat is better then nested)
    
    TODO
    ----------
    adjust all the threshold functions to support SegmentsOfContinuous object (return it instead of regular Segments)
    
    
    """
    def __init__(self, segments, source):
        """
        """
        self._segments = segments
        self._source = source
        
    @classmethod
    def from_starts_ends(cls, starts, ends, source):
        """
        
        """
        raise NotImplementedError
        segments = Segments(starts, ends)
        return cls(segments, source)
        
    @property
    def source(self):
        return self._source
        
    @property
    def segments(self):
        """
        return a Segments instance without continuous data as source
        """
        return self._segments
    
    def each_max_value(self):
        raise NotImplementedError
    
    def each_mean_value(self):
        raise NotImplementedError
        
    def each_sum_of_values(self):
        """
        mostly for histograms
        """
        raise NotImplementedError
        
    def each_integral_value_over_domain(self):
        """
        that is the correct way to refer to an energy of a pulse
        """
        raise NotImplementedError
        
    def each_carrier_frequency(self):
        """
        it assumes that each segment / pulse has a single carrier frequency, and finds it.
        implementation options: fm-demodulation and mean, or fft and take the strongest freq
                
        returns:
        --------------
        the carrier frequency of each segment / pulse
        
        TODO
        --------
        maybe should return a number with uncertainty? or segment?
        """
        return NotImplementedError
