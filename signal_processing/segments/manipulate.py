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


def filter_short_segments(segments, min_duration):
    """
    TODO: it should be based on filter_by_range
    """
    return segments.filter_by_range('durations', Segment([0, min_duration]), mode='remove')
 
