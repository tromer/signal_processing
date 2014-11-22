import warnings

import numpy as np

from signal_processing.segment import Segment

from signal_processing.extensions import pint_extension

from segments_obj import Segments

import adjoin

def concatenate(segments_list, XXX_allow_not_in_order=False):
    """
    concatenates segments, if they are all one after another
    TOOD: test when there are empty segments
    """
    # filter out empty segments instances
    segments_list = filter(None, segments_list)

    if not XXX_allow_not_in_order:
        for i in xrange(len(segments_list) - 1):
            if segments_list[i].ends[-1] > segments_list[i + 1].starts[0]:
                raise ValueError("not in order")

    # this is a thin patch for cases where we don't enforce segments in order
    else:
        pass


    all_starts = map(lambda segments : segments.starts, segments_list)
    all_ends = map(lambda segments : segments.ends, segments_list)
    return Segments(pint_extension.concatenate(all_starts), pint_extension.concatenate(all_ends))


def concatenate_single_segments(segs_list):
    """
    similar to from_single_segments
    remove one of the constructors!
    have to be one after another

    TODO:
    ----------
    in order to make it work, use a recursive adjoin, there is the right function for it in the adjoin file. only by max distance of zero, again and again.
    Note, that the constructor of Segments is going to have a gaurdian that the locations are strictly increasing. to bypass it (because in the process of the function it would be necessary) or make a flag that allows to skip the gaurdian, or make a "temporary" subclass
    also note, that the adjoin.max_dist assumes that the first start and last end are True, which may cause a problem here, with the last end

    other option, which may work - start with sorting by start, and also by end of the segment. maybe it will say something smart, and thus avoid the loop
    """
    #as_segments = map(Segments.from_single_segment, segs_list)
    #return concatenate(as_segments)

    raise NotImplementedError
    sorted_by_start = sorted(segments_list, key=lambda s : s.start)
    starts = pint_extension.array(map(lambda s : s.start, sorted_by_start))
    ends = pint_extension.array(map(lambda s : s.end, sorted_by_start))
    segments_maybe_overlap = Segments(starts, ends)
    segments = adjoin.max_dist(segments_maybe_overlap, max_distance=0 * pint_extension.get_units(segments_maybe_overlap.starts))
    return segments


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
    warnings.warn("deprecated, use Segments.gaps instead")
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

