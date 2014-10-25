import os
from os import path

from segments_obj import Segments
import signal_processing.continuous as cont

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

    on the same issue: in a way, this object is a coupling of Segments and ContinuosData,
    it's not clear that the "order 0" interface should be the one of the segments.
    I am not sure....

    maybe it should inherit from both of them?????

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

    @property
    def starts(self):
        return self.segments.starts

    @property
    def ends(self):
        return self.segments.ends

    def is_close(self, other):
        return self.segments.is_close(other.segments) and self.source.is_close(other.source)

    def shift(self, delta):
        raise NotImplementedError
        new_segments = self.segments.shift(delta)
        new_source = self.source.shift(delta)
        return SegmentsOfContinuous(new_segments, new_source)

    def __getitem__(self, key):
        """
        """
        raise NotImplementedError

        if type(key) in [type(slice(0,1)), np.ndarray]:
            return SegmentsOfContinuous(self.segments[key], self.source)

        elif type(key) == int:
            return self.source[self.segments[key]]

    def to_file(self, path):
        """
        writes the instance to file. the file is a folder that contains a wav
        and a csv

        parameters
        -------------
        path : str
            folder

        returns
        -----------
        None
        """
        if os.path.isfile(path):
            raise ValueError("need a path for a directory")
        if not os.path.isdir(path):
            os.mkdir(path)

        path_segs = os.path.join(path, 'segments.csv')
        path_source = os.path.join(path, 'source.wav')

        self.segments.to_csv(path_segs)
        cont.io.write_wav(self.source, path_source)

    @classmethod
    def from_file(cls, path):
        """
        parameters
        ------------
        path : str
            folder!

        returns
        -----------
        SegmentsOfContinuous

        """
        if os.path.isfile(path):
            raise ValueError("need a path for a directory")

        # refactor: it's coppied from to_file
        path_segs = os.path.join(path, 'segments.csv')
        path_source = os.path.join(path, 'source.wav')

        segments = Segments.from_csv(path_segs)
        source = cont.io.read_wav(path_source)

        out = cls(segments, source)
        return out

    def __str__(self):
        # not tested
        return '\n'.join([str(self.source), str(self.segments)])

    def plot(self, fig=None):
        fig = self.source.plot(fig)
        self.segments.mark_edges(fig)

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
