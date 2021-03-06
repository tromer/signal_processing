# -*- coding: utf-8 -*-
"""

.. module:: threshold
    :synopsis: takes threshold on a ContinuousData to find segments of interest.

The core function is crosses.
There are other function that apply other processes after the threshold

"""
"""
Created on Thu Aug 21 18:31:11 2014

@author: noam
"""

import warnings

import numpy as np

from signal_processing.segments.segments_obj import Segments
from signal_processing.segments.segments_of_continuous_obj import \
    SegmentsOfContinuous
from signal_processing.segments import adjoin
from signal_processing.segments import manipulate


from .extensions import pint_extension
from .extensions import numpy_extension


IS_DEBUG = False


def crosses(sig, threshold, is_above=True):
    """
    returns segments instance, where a given signal values are above
    a certain threshold.

    parameters
    ---------------
    sig : ContinuousData
        a signal
    threshold : float with units like the values of sig
        TODO: for non-constant threshold, we may allow passing
        a vector of values, or a thrshold which is a signal
        TODO: allow passing a range of allowed thresholds
    is_above : bool
        whether the segments of interest are above the threshold

    returns
    -----------
    segments : Segments


    Note
    --------
    this function assumes that the domain of the signal has
    a mathematical order defined upon it. this applies to most
    cases. exceptions are when the domain cells are modulu of
    something, or frequencies.
    if the domain is modulu derived / cyclic, it's still possible to
    implement. the "first" sample is "connected" to the last.
    we find the indexes of threshold crosses the same way,
    we have a segments instance. now we just have to decide whether
    it's the segments of interest, or the others are. we check a
    single value in one of the segments to determine that
    """
    above = sig.values > threshold
    if not is_above:
        above = np.logical_not(above)
    # the beginning and end count as non pulse
    crossings = np.logical_xor(
        np.concatenate([above, [False, ]]), np.concatenate([[False], above]))
    crossings_indexes = np.where(crossings)[0]
    crossings_times = crossings_indexes * sig.sample_step + sig.domain_start
    starts = crossings_times[::2]
    ends = crossings_times[1::2]
    segs = Segments(starts, ends)
    sig_and_segs = SegmentsOfContinuous(segs, sig)
    return sig_and_segs

def threshold_adjoin_filter_short_segments(
    sig, threshold, max_distance, min_duration):
    """
    concatanates 3 processes one after another:
    threshold, adjoin, filter_short_segments
    """
    warnings.warn("not tested")
    p = crosses(sig, threshold)
    # note that it's important to adjoin before filtering short segments
    p = adjoin.adjoin_close_segments(p, max_distance)
    p = manipulate.filter_short_segments(p, min_duration)
    return p


def threshold_aggregate_filter_short(
    sig, threshold, max_dist, duration_ratio,
    absolute_max_dist, max_pulse_duration):
    """
    TODO: add documentation, test

    refactor
    ----------------
    this function and the function threshold_adjoin_filter_short_segments
    are similar. should probably unite them as one
    """
    warnings.warn("threshold_aggregate_filter_short not tested")
    backgrounds = crosses(sig, threshold)
    # remove small drops in the middle, and drops near the edge
    backgrounds = adjoin.max_dist(backgrounds, max_dist)


    # remove big drops in the middle
    backgrounds = adjoin.consider_duration(
        backgrounds, duration_ratio, absolute_max_dist, mode='min')
    # remove interrupts. some interupts are a little bit too close
    # so removing only by a factor multiplied by the interrupt duration
    # is not enough. we filter harder
    backgrounds = manipulate.filter_short_segments(
        backgrounds, max_pulse_duration)
    # TODO: fine tuning - get entire background pulse
    return backgrounds


def estimate_noise_level(sig, mode, factor):
    """
    a auto threshold of values of signal

    parameters:
    ------------------
    mode : str
        'mean'
        'median'
        'zero'

    returns:
    -----------
    thresh

    TODO: Idea for a "flexible" auto threshold.
    idea 1: sliding window.
    idea 2: the flexible mean by convolution with 1 / n * np.ones(n)
    then (vec - mean) ** 2
    then another convolution for mean
    then sqrt

    refactor
    ------------
    maybe this function should be in continuous.math module?
    """
    warnings.warn("not tested")
    vals = sig.values
    if mode == 'mean':
        center = np.mean(vals)
        deviation = np.std(vals)
    elif mode == 'median':
        center = pint_extension.median(vals)
        deviation = numpy_extension.deviation_from_reference(vals, center)
    elif mode == 'zero':
        center = 0
        deviation = numpy_extension.deviation_from_reference(vals, center)

    thresh = center + factor * deviation
    return thresh


def cluster1d(vec, resolution, threshold):
    """
    find main values in histigram. should be probably implemented
    like finding segments in every ContinuousData
    the only difference is defining a resolution here
    """
    raise NotImplementedError
    bins_num = np.ceil(1.0 * vec.ptp() / resolution)
    hist, edges = pint_extension.histogram(vec, bins_num, density=True)
    clusters = crosses(vec, 1, threshold)
    return clusters
