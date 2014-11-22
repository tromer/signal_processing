# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:14:13 2014

@author: noam
"""
from signal_processing import U_
from signal_processing.extensions import pint_extension
from signal_processing.extensions.plt_extension import mark_vertical_lines
import numpy as np
import pandas as pd
from signal_processing.segment import Segment

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
    def __init__(self, starts, ends, unit=None):
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
        assert len(starts) == len(ends)

        if hasattr(starts, 'units') and hasattr(ends, 'units'):
            if starts.dimensionality != ends.dimensionality:
                raise ValueError(
                    'starts and ends with diffrent dimensionalities')

        elif unit != None:
            if hasattr(starts, 'units') or hasattr(ends, 'units'):
                raise ValueError('only starts / ends has units')
            starts = np.array(starts) * unit
            ends = np.array(ends) * unit

        else:
            raise ValueError("units not given")

        self._starts = starts
        self._ends = ends
        # global_start, global_end?
        # pointer to internal data?

    @classmethod
    def from_single_segment(cls, segment):
        """
        returns:
        -------------
        segments : Segments
            containing only one segment
        """
        return Segments(pint_extension.array([segment.start,]), pint_extension.array([segment.end,]))



    def __str__(self):
        line_1 = "starts: " + str(self.starts)
        line_2 = "ends: " + str(self.ends)

        self_str = "\n".join([line_1, line_2])
        return self_str

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
    def gaps(self):
        """
        returns the gaps

        returns
        ----------
        Segments
        """
        raise NotImplementedError

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

    def shift(self, delta):
        """
        returns:
        -----------
        shifted : Segments
            the segments shifted

        """
        raise NotImplementedError
        return Segments(self.starts + delta, self.ends + delta)

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

    def to_csv(self, path):
        """
        writes the segments instance to csv file

        parameters
        -------------
        path : str
            path to file

        returns
        ----------
        None
        """
        starts_ends, unit = pint_extension.strip_units(
            [self.starts, self.ends])
        headers = ['starts_' + str(unit.units), 'ends_' + str(unit.units)]
        table = pd.DataFrame(
            np.vstack(starts_ends).transpose(), columns=headers)
        table.to_csv(path)

    @classmethod
    def from_csv(cls, path):
        """
        reads from csv

        parameter
        -----------
        path : str
            path of file

        returns
        ----------
        Segments
        """
        table = pd.DataFrame.from_csv(path)
        headers = table.columns
        data = table.as_matrix()

        starts = data[:, 0] * U_(headers[0].split("_")[-1])
        ends = data[:, 1] * U_(headers[1].split("_")[-1])
        return cls(starts, ends)

    def mark_edges(self, fig):
        """
        mark the edges of the segments on a fig
        get a figure and plot on it vertical lines according to
        starts and ends

        returns:
        -----------
        lines_start
        lines_end

        TODO: allow subplot as well,
        should use plt_extension.focus_on_figure_and_subplot
        """
        start_lines = mark_vertical_lines(self.starts, fig, color='g', label="starts")
        ends_lines = mark_vertical_lines(self.ends, fig, color='r', label="ends")
        return start_lines, ends_lines

def fromfile(f):
    """
    reads Segments instance from file
    TODO
    --------
    decide which format. probably csv, what about the units?
    put them in the headers?
    """
    raise NotImplementedError

