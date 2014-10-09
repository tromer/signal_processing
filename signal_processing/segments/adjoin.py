import warnings

import numpy as np

from signal_processing import uerg

from signal_processing.extensions import pint_extension

from segments_obj import Segments


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
def adjoin_segments_considering_durations(segments_, segment_gap_ratio, absolute_max_dist=None, mode='mean'):
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
    
    durations = segments_.durations
    
    if mode == 'mean':
        reference_duration_for_each_gap = 0.5 * (durations[:-1] + durations[1:])
    elif mode == 'min':
        reference_duration_for_each_gap = pint_extension.minimum(durations[:-1], durations[1:])
    elif mode == 'max':
        raise NotImplementedError
        reference_duration_for_each_gap = pint_extension.maximum(durations[:-1], durations[1:])
        
    max_distance_due_to_duration = reference_duration_for_each_gap * segment_gap_ratio
    
    if absolute_max_dist != None:
        assert absolute_max_dist.dimensionality == segments_.starts.dimensionality
        max_distance = pint_extension.minimum(max_distance_due_to_duration, absolute_max_dist)
    else:
        max_distance = max_distance_due_to_duration
    
    adjoined_segments = adjoin_segments_max_distance(segments_, max_distance)
    return adjoined_segments
    
def adjoin_segments(segments_, delta=0, ratio=0, max_dist=None, n=1):
    """
    parameters:
    ----------------
    n : int
        number of iterations

    TODO:
    ---------
    add parameter to allow maximal / ultimate adjoining. adjoin again and again until it cannot adjoin anymore
    """
    warnings.warn("adjoin_segments is not tested")
    if delta != 0: 
        assert delta.dimensionality == segments_.starts.dimensionality
    if max_dist != None:
        assert max_dist.dimensionality == segments_.starts.dimensionality
    

    adjoined_segments = segments_
    for i in xrange(n):
        if delta != 0:
            adjoined_segments = adjoin_segments_max_distance(adjoined_segments, delta)
        if ratio != 0:
            adjoined_segments = adjoin_segments_considering_durations(adjoined_segments, ratio, max_dist)
            
    return adjoined_segments
 
