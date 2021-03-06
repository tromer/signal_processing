# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 01:29:39 2014

@author: noam
"""

raise NotImplementedError("this is an old file. it's contents are\
                          split between various other modules. don't use it\
                          it's mere purpose is to enable tracking of\
                          the history in git.")
import warnings
import tempfile
#%%
import numpy as np
import scipy as sp
from scipy.io import wavfile
import matplotlib.pyplot as plt

from global_uerg import uerg, Q_


from segment import Segment
import segments
from segments import  Segments
from .extensions import numpy_extension
from .extensions import scipy_extension
from .extensions import pint_extension


#%%

ARBITRARY_UNITS_STR = "[AU]"
#%%

class ContinuousData(object):
    """
    Note: see also the object Segments. they go hand in hand together, refer to different aspects of the same subjects
    this class represents any kind of continuous data (one dimensional).
    It includes a few kinds that first seem different from each other, has a lot in common.

    examples:
    1. a "signal" - measurement of the electromagnetic field / voltage as a function of time.
    similar examples: sound (as a function of time), seismic measurments.
    2. a spatial measurement: height as a function of place, some material density as a function of place.
    similar examples: stress as a function of place.
    3. a distribution: the number of occurances in a population, as a function of age / height / amplitude
    4. a kinematic property of a system: position / velocity / acceleration / angle as a function of time.
    5. even a spectrum of a signal - the magnitude as a function of frequency.
    6. any connection between two continuous variables, such as a response curv of harmonic ocsillator:
    amplitude of the ocsillator as a function of the frequency of external force.

    There are some differences beween these kinds of data. Maybe some of them would be implemented as a subclass

    basic assumptions:
    1. the acctual data in the real world can be really continuous.
    here we of course use sample points. every sample point has exectly one corresponding value.
    2. We assume that our data really representes the reallity, spesifically, that
    represents well also the times / places which we didn't measure.
    So, we assume that the data is not changing "too fast" compared to our resolution.
    In signal processing terms, we assume that we didn't under-sample.

    Note:
    1. we *do not* assume even sampling distance. that would be included in a subclass.
    2. this class is intentioned to be used with units. it's real world measurements.
    3. measurement errors are not handled here.
    4. It may seem like a big overhead to use this object instead of just numpy.ndarray
    this is not so. This object prevents errors of bad handling of sample rate.
    It also works naturally with units, that prevents other errors.
    For example: diff sould take the distance between samples into account.
    FFT sould take the sample rate into account, and so on.
    When you encounter an operation that is not implemented for ContinuousData,
    the correct thing to do is to wrap the numpy or scipy operation.

    TODO: maybe I want to implement a domain_samples object. it would have
    a subclass of even samples

    TODO: maybe it's smart to implement a similar object with few channels.
    It may be useful in some implementation and performance issues,
    since the channels would be a 2D np.ndarray, and channel-wise
    operations like fft would be applied along axis, and be
    efficient.

    TODO: maybe add a self.base attribute, like in np.ndarrays

    """
    def __init__(self, values, domain_samples):
        assert len(values) == len(domain_samples)
        self._domain_samples = domain_samples
        self._values = values

    @property
    def domain_samples(self):
        return self._domain_samples

    @property
    def values(self):
        return self._values

    @property
    def n_samples(self):
        return len(self.values)

    @property
    def first_sample(self):
        return self.domain_samples[0]

    @property
    def last_sample(self):
        return self.domain_samples[-1]

    def is_same_domain_samples(self, other):
        raise NotImplementedError


    def is_close(self, other, domain_rtol=1e-5, domain_atol=None, values_rtol=1e-5, values_atol=None):
        """ TODO: use is_same_domain_samples in this func """
        return pint_extension.allclose(self.domain_samples, other.domain_samples, domain_rtol, domain_atol) \
        and pint_extension.allclose(self.values, other.values, values_rtol, values_atol)

    def is_close_l_1(self, other, param_1, param_2):
        """
        checks if 2 signals are close using l_1 norm
        TODO: maybe should be a norm parameter, to decide which norm to use
        """
        raise NotImplementedError

    """
    # maybe len should really return the number of sample points
    # I am not shure whether the number of sample points should be a part of the interface/
    # but in many implementations of functions I need it, so use len(contin.values)
    def __len__(self):
        raise NotImplementedError
        return self.domain_samples.ptp()
    """

    def __getitem__(self, key):
        """
        parameters:
        -------------
        domain_range : Segment
            the range, from the domain, of which we want the slice.
            for example: which time range?

        TODO: since the domain samples should be sorted, maybe there
        is a more efficient implementation
        """
        if type(key) in [int, float]:
            raise KeyError("wrong key. key for ContinuousData is Segment or Segments of the same domain")

        if type(key) in [Segment,]:
            domain_range = key
            is_each_in_range = domain_range.is_each_in(self.domain_samples)
            return ContinuousData(self.values[is_each_in_range], self.domain_samples[is_each_in_range])

        elif type(key) in [Segments,]:
            return [self[domain_range] for domain_range in key]

    def gain(self, factor):
        """
        multiplies the values by the factor
        """
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def abs(self):
        raise NotImplementedError

    def DFT(self):
        raise NotImplementedError
        # maybe there is an issue regarding using DFT or IDTF, depending the domain
        # maybe it should be an extra param. seying which one to use
        # maybe should be an external function, not a method

    def tofile(self, f):
        """
        read the docs of fromfile
        """
        raise NotImplementedError

def test_ContinuousData():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    assert pint_extension.allclose(sig.domain_samples, t)
    assert pint_extension.allclose(sig.values, vals)
    assert sig.n_samples == 10
    assert sig.first_sample == 0 * uerg.sec
    assert sig.last_sample == 9 * uerg.sec

    assert sig.is_close(sig)
    assert not sig.is_close(ContinuousData(vals, t + 1 * uerg.sec))
    assert not sig.is_close(ContinuousData(vals + 1 * uerg.volt, t))

    # ___getitem__
    t_range = Segment(np.array([2.5, 6.5]) * uerg.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousData(vals[expected_slice], t[expected_slice])
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)

    t_ranges = Segments(np.array([2.5, 7.5]) * uerg.sec, np.array([6.5, 8.5]) * uerg.sec)
    sig_list = sig[t_ranges]
    ranges = t_ranges.to_segments_list()
    expected_sig_list = [sig[ranges[0]], sig[ranges[1]]]
    for i in range(len(sig_list)):
        assert sig_list[i].is_close(expected_sig_list[i])

test_ContinuousData()


#%%
def plot_quick(contin, is_abs=False, fmt="-"):
    """
    contin is a ContinuousData object
    is_abs - whether to plot the abs of the y values. for spectrums.
    Maybe - there sould be an input of lambda functions. to preprocess x,y
    """
    warnings.warn("plot_quick is not tested")
    #TODO: test this function!
    # creat the figure here
    return plot(contin, fig=None, is_abs=is_abs, fmt=fmt)
#%%

def visual_test_plot_quick():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    plot_quick(sig)

#visual_test_plot_quick()

def plot(contin, fig=None, subplot=None, share_x=None, is_abs=False, fmt="-", ):
    """
    add a plot of ContinuousData instance, to an existing figure
    TODO: allow passing every parameter the plt.plot accepts. i.e - making ot a complete
    wrapper around plt.plot
    TODO: make sure somehow that all the plots on the same figure, share x axis dimensionality
    and rescale them - (fig, x_untis) tuple
    TODO: instead of putting units on y axis, use legend and put units there

    parameters:
    -------------
    contin
    fig - a plt.plot figure object
    subplot - list indicating subplot like [3,1,1] - of 3 lines, and 1 col, subplot 1 (on top)
    is_abs: whether to use np.abs() on the values. mostly for plotting power spectrums
    fmt - format, like plt.plot fmt
    """
    # assert contin type?
    # TODO: add support for legend
    warnings.warn("plot is not tested")
    warnings.warn("plot dosn't rescale the last signal according to axes")

    if fig == None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    if subplot != None:
        plt.subplot(*subplot, sharex=share_x)

    x = contin.domain_samples
    y = contin.values
    if is_abs:
        y = np.abs(y)

    line = plt.plot(x, y, fmt)[0]
    plt.xlabel(ARBITRARY_UNITS_STR)
    plt.ylabel(ARBITRARY_UNITS_STR)
    if type(x) == uerg.Quantity:
        if not x.unitless:
            plt.xlabel(str(x.dimensionality) + " [" + str(x.units) + "]")

    if type(y) == uerg.Quantity:
        if not y.unitless:
            plt.ylabel(str(y.dimensionality) + " [" + str(y.units) + "]")

    return fig, line
    #raise NotImplementedError
        # return fig, axes??


def visual_test_plot():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    plot(sig)

# visual_test_plot_quick()

def plot_under(contin_list, fig=None, is_abs=False, fmt="-"):
    """
    plot a few signals one above the other




    OLD OLD OLD XXX
    add subplot of the signal, to an existing plot of another signal.
    the x axis would be coordinated.
    should enable easier examining of signals

    TODO: maybe add parameter of subplot or something
    """
    warnings.warn("not tested well for units, nots tested")
    if fig != None:
        raise NotImplementedError

    f = plt.figure()
    lines = []
    N = len(contin_list)
    ax = plt.subplot(N, 1, 1)

    for i in xrange(N):
        junk, line = plot(contin_list[i], f, [N, 1, i + 1], share_x=ax)
        lines.append(line)

    return f, lines

def visual_test_plot_under():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.amp
    sig = ContinuousData(vals, t)
    sig_2 = sig
    sig_list = [sig, sig_2]
    plot_under(sig_list)

#%%
class ContinuousDataEven(ContinuousData):
    """
    read the ContinuousData documentation.
    the domain samples are evenly spaced
    """
    def __init__(self, values, sample_step, first_sample=0):
        self._values = values
        self._sample_step = sample_step
        if not first_sample:
            self._first_sample = 0 * sample_step
        else:
            self._first_sample = first_sample


    @property
    def sample_step(self):
        return self._sample_step

    @property
    def sample_rate(self):
        return 1.0 / self.sample_step

    @property
    def first_sample(self):
        return self._first_sample


    @property
    def total_domain_range(self):
        """

        returns:
        ------------
        domain_range : Segment
        """
        raise NotImplementedError

    @property
    def total_domain_width(self):
        """
        returns the total length of the signal

        Note:
        maybe it's possible to base the implementation on total_domain_range
        at the moment it's done differently

        TODO: use this method as a first check in comparison
        """
        return self.n_samples * self.sample_step

    """
    @property
    def last_sample(self):
        # TODO: possible to create a better implementation then in the father class
    """

    @property
    def domain_samples(self):
        #print "******"
        #print self.values
        """ TODO: mayebe some cashing would be helpful? """
        return np.arange(len(self.values)) * self.sample_step + self.first_sample

    def __getitem__(self, key):
        """
        Note: it's coppied from __getitem__ of ContinuousData
        parameters:
        -------------
        domain_range : Segment
            the range, from the domain, of which we want the slice.
            for example: which time range?
        """
        if type(key) in [int, float]:
            raise KeyError("wrong key. key for ContinuousData is Segment or Segments of the same domain")


        if type(key) in [Segment,]:
            domain_range = key
            bottom_index = np.ceil(1.0 * domain_range.start / self.sample_step)
            top_index = np.floor(domain_range.end / self.sample_step)
            return ContinuousDataEven(self.values[bottom_index:top_index + 1], self.sample_step, first_sample=bottom_index * self.sample_step)

        elif type(key) in [Segments,]:
            return [self[domain_range] for domain_range in key]


    def move_first_sample(self, new_first_sample=0):
        """

        """
        warnings.warn("not tested")
        return ContinuousDataEven(self.values, self.sample_step, new_first_sample)


    def is_same_domain_samples(self, other):
        return self.n_samples == other.n_samples and \
        pint_extension.allclose(self.first_sample, other.first_sample) and \
        pint_extension.allclose(self.sample_step, other.sample_step)

    def _extract_values_from_other_for_continuous_data_arithmetic(self, other):
        """
        core method to help arithmency between methods
        TODO: add more test not tested enough. some bugs
        """

        if type(other) in [float, int]:
            values = other

        elif type(other) == uerg.Quantity:
            if type(other.magnitude) in [np.ndarray,]:
                raise ValueError("add const value, or other ContinuousData with same domain samples")
            else:
                values = other
        else:
            # TODO: add gaurdian, that other is another ContinuousData
            if not self.is_same_domain_samples(other):
                raise ValueError("diffrent domain samples")
            else:
                values = other.values

        return values

    def __add__(self, other):
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values + values, self.sample_step, self.first_sample)

    def __radd__(self, other):
        raise NotImplementedError
        return self + other

    def __sub__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values - values, self.sample_step, self.first_sample)


    def __mul__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values * values, self.sample_step, self.first_sample)

    def __rmul__(self, other):
        raise NotImplementedError
        return self * other

    def __div__(self, other):
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values / values, self.sample_step, self.first_sample)

    def abs(self):
        return ContinuousDataEven(np.abs(self.values), self.sample_step, self.first_sample)


    def gain(self, factor):
        """
        see doc of base class
        """
        return ContinuousDataEven(self.values * factor, self.sample_step, self.first_sample)

    def down_sample(self, down_factor):
        assert down_factor > 0
        if int(down_factor) != down_factor:
            raise NotImplementedError
        # maybe there should be another interface, with "new sample rate"
        return ContinuousDataEven(self.values[::down_factor], down_factor * self.sample_step, self.first_sample)

    def is_power_of_2_samples(self):
        """
        check for performance issues
        """
        return numpy_extension.is_power_of_2(self.n_samples)

    def trim_to_power_of_2_XXX(self):
        """
        trancate data to power of 2 sample points
        loss of data, very dangareous
        """
        warnings.warn("XXX trim_to_power_of_2_looses_data")
        new_n = numpy_extension.close_power_of_2(self.n_samples, mode='smaller')
        trimmed = ContinuousDataEven(self.values[:new_n], self.sample_step, self.first_sample)
        assert trimmed.n_samples == new_n
        return trimmed

    def get_chunks(self, domain_duration, is_power_of_2_samples=True, is_overlap=False, mode_last_chunk='throw'):
        """
        get small chunks of the signal

        parameters
        -----------------
        domain_duration : uerg.Quantity
            the duration / len / width of the chunk
        is_power_of_2_samples : bool
            whether to return bigger chunks, with 2 ** m samples
            for performence / efficiency issues
        is_overlap : bool
            whether to give chunks with overlap of half the duration
            in order to cope with splits
        mode_last_chunk : str
            what to do with the last chunk, which is smaller?
            'throw' - just don't give it back

        returns
        ----------
        a list of signals.
        if performence issues erise, it's better to return a
        generator using yield.

        TODO:
        --------
        there is a lot of code duplication with the odd and even
        maybe should return a sig array object
            channels object is another option, but I think that it's a bad idea
            since the chunks do not share the domain samples. (different time)
        maybe should enable passing Segment, or Segments to specify where to chunk
        cope with case the signal is too short compared to chunk asked

        """
        n_samples_chunk = np.ceil(domain_duration / self.sample_step)
        if is_power_of_2_samples:
            n_samples_chunk = numpy_extension.close_power_of_2(n_samples_chunk, 'bigger')

        n_samples_tot = self.n_samples

        chunks_odd = []
        n_of_chunks_odd = np.floor(n_samples_tot / n_samples_chunk)
        chunks_odd_data = self.values[:n_samples_chunk * n_of_chunks_odd].reshape((n_of_chunks_odd, n_samples_chunk)).transpose()
        chunk_odd_first_samples = self.first_sample + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_odd)
        assert chunks_odd_data.shape[1] == len(chunk_odd_first_samples)

        for i in xrange(len(chunk_odd_first_samples)):
            chunk = ContinuousDataEven(chunks_odd_data[:,i], self.sample_step, chunk_odd_first_samples[i])
            chunks_odd.append(chunk)

        if not is_overlap:
            chunks = chunks_odd
        else:
            chunks_even = []
            n_of_chunks_even = np.floor((n_samples_tot - 0.5 * n_samples_chunk) / n_samples_chunk)
            chunks_even_data = self.values[0.5 * n_samples_chunk : 0.5 * n_samples_chunk + n_samples_chunk * n_of_chunks_even].reshape((n_of_chunks_even, n_samples_chunk)).transpose()
            chunk_even_first_samples = self.first_sample + self.sample_step * 0.5 * n_samples_chunk + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_even)
            assert chunks_even_data.shape[1] == len(chunk_even_first_samples)

            for i in xrange(len(chunk_even_first_samples)):
                chunk = ContinuousDataEven(chunks_even_data[:,i], self.sample_step, chunk_even_first_samples[i])
                chunks_even.append(chunk)

            chunks = chunks_odd + chunks_even

        if mode_last_chunk != 'throw':
            raise NotImplementedError

        return chunks



def test_ContinuousDataEven():
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    assert pint_extension.allclose(sig.sample_step, sample_step)
    assert pint_extension.allclose(sig.sample_rate, 1.0 / sample_step)
    assert pint_extension.allclose(sig.values, values)
    assert pint_extension.allclose(sig.total_domain_width, 10 * uerg.sec)
    assert pint_extension.allclose(sig.domain_samples, np.arange(10) * sample_step)
    assert sig.is_close(ContinuousData(values, np.arange(10) * sample_step))
    assert pint_extension.allclose(sig.first_sample, 0 * sample_step)

    # testing a __getitem__ (slicing) is mostly copied from the tester of ContinuousData
    t_range = Segment(np.array([2.5, 6.5]) * uerg.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousDataEven(values[expected_slice], sample_step, expected_slice[0] * sample_step)
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)

def test_down_sample():
    # copied from the test of fft
    sig = ContinuousDataEven(np.arange(32) * uerg.amp, 1.0 * uerg.sec)
    down_factor = 2
    expected_down = ContinuousDataEven(np.arange(0, 32, 2) * uerg.amp, 2.0 * uerg.sec)
    down = sig.down_sample(down_factor)
    assert down.is_close(expected_down)


def test_gain():
    # copied from test_ContinuousDataEven
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    factor = 2
    expected_sig_gain = ContinuousDataEven(values * factor, sample_step)
    sig_gain = sig.gain(factor)
    assert sig_gain.is_close(expected_sig_gain)

def test_is_same_domain_samples():
    step_1 = uerg.sec
    step_2 = uerg.sec * 2
    start_1 = 0
    start_2 = 1 * uerg.sec
    vals_1 = np.arange(10) * uerg.mamp
    vals_2 = 2 * np.arange(10) * uerg.amp
    vals_3 = np.arange(5) * uerg.amp
    assert ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_2, step_1))
    assert not ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_1, step_2))
    assert not ContinuousDataEven(vals_1,step_1, start_1).is_same_domain_samples(ContinuousDataEven(vals_1, step_1, start_2))
    assert not ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_3, step_1))

def test__extract_values_from_other_for_continuous_data_arithmetic():
    # copied from test___add__
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    expected_values = sig.values
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(sig)
    assert pint_extension.allclose(values, expected_values)

    num = 2 * uerg.mamp
    expected_values = num
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(num)
    assert pint_extension.allclose(values, expected_values)

def test___add__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    assert (sig + sig).is_close(sig.gain(2))
    num = 2 * uerg.mamp
    add_1 = sig + num
    expected_add_1 = ContinuousDataEven((2 + np.arange(10)) * uerg.mamp, uerg.sec)
    assert add_1.is_close(expected_add_1)

def test___sub__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    sig_2 = ContinuousDataEven(np.ones(10) * uerg.mamp, uerg.sec)
    dif = ContinuousDataEven(np.arange(-1,9) * uerg.mamp, uerg.sec)
    assert (sig - sig_2).is_close(dif)

def test___mul__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    expected_sig_pow_2 = ContinuousDataEven(np.arange(10) ** 2 * uerg.mamp ** 2, uerg.sec)
    sig_pow_2 = sig * sig
    assert sig_pow_2.is_close(expected_sig_pow_2)

def test___div__():
        sig = ContinuousDataEven(np.arange(1, 10) * uerg.mamp, uerg.sec)
        assert (sig / sig).is_close(ContinuousDataEven(1 * uerg.dimensionless * np.ones(9), uerg.sec))
        assert (sig / 2.0).is_close(ContinuousDataEven(0.5 * np.arange(1, 10) * uerg.mamp, uerg.sec))

def test_abs():
    sig = ContinuousDataEven((-1) * np.ones(10) * uerg.mamp, uerg.sec)
    expected_sig_abs = ContinuousDataEven(np.ones(10) * uerg.mamp, uerg.sec)
    sig_abs = sig.abs()
    assert sig_abs.is_close(expected_sig_abs)

def test_is_power_of_2_samples():
    sig = ContinuousDataEven(np.ones(16) * uerg.mamp, uerg.sec)
    assert sig.is_power_of_2_samples()

    sig = ContinuousDataEven(np.ones(13) * uerg.mamp, uerg.sec)
    assert not sig.is_power_of_2_samples()

def test_trim_to_power_of_2_XXX():
    sig = ContinuousDataEven(uerg.mamp * np.arange(12), 1 * uerg.sec)
    expected_sig_trim = ContinuousDataEven(uerg.mamp * np.arange(8), 1 * uerg.sec)
    sig_trim = sig.trim_to_power_of_2_XXX()
    assert sig_trim.is_close(expected_sig_trim)


def test_get_chunks():
    N = 32
    sig = ContinuousDataEven(np.arange(N) * uerg.mamp, uerg.sec)
    chunk_duration = 3 * uerg.sec
    chunked_odd = sig.get_chunks(chunk_duration, is_overlap=False)
    chunked = sig.get_chunks(chunk_duration, is_overlap=True)
    expected_chunked_odd = [ContinuousDataEven(np.arange(4 * i, 4 * (i + 1)) * uerg.mamp, uerg.sec, uerg.sec * 4 * i) for i in range(8)]
    for i in xrange(len(chunked_odd)):
        assert chunked_odd[i].is_close(expected_chunked_odd[i])

    expected_chunked_even = [ContinuousDataEven(np.arange(2 + 4 * i, 2 + 4 * (i + 1)) * uerg.mamp, uerg.sec, uerg.sec * ( 4 * i + 2)) for i in range(7)]
    expected_chunked = expected_chunked_odd + expected_chunked_even

    for i in xrange(len(chunked)):
        assert chunked[i].is_close(expected_chunked[i])



test_ContinuousDataEven()
test_down_sample()
test_gain()
test_is_same_domain_samples()
test__extract_values_from_other_for_continuous_data_arithmetic()
test___add__()
test___sub__()
test___mul__()
test___div__()
test_abs()
test_is_power_of_2_samples()
test_trim_to_power_of_2_XXX()
test_get_chunks()
#%%


"""
io

"""

def read_wav(filename, domain_unit=uerg.sec, first_sample=0, value_unit=uerg.milliamp, expected_sample_rate_and_tolerance=None, channels=None):
    """
    read wav file to ContinuousDataEven.
    implemented only for one channal
    for multiple channels we probably want to return a list of ContinuousDataEven

    parameters:
    ------------
    domain_unit
        the unit of the domain. usually sec

    first_sample
        in case it's not 0

    value_unit
        the unit of the values

    channels
        if it's list, it says which channels to return

    XXX TODO: understand whether it reads 16pcm correctly. it's not sure
    some source:
    http://nbviewer.ipython.org/github/mgeier/python-audio/blob/master/audio-files/audio-files-with-scipy-io.ipynb#Reading

    returns:
    ------------
    if channels == None, returns signal
    if channels != None, returns a list of signals
    """
    sample_rate, raw_sig = sp.io.wavfile.read(filename)
    sample_rate = 1.0 * sample_rate / domain_unit
    raw_sig = raw_sig * value_unit
    if expected_sample_rate_and_tolerance != None:
        # shold raise a meaningful excepion.
        is_sample_rate_as_expected = np.abs(sample_rate - expected_sample_rate_and_tolerance[0]) < expected_sample_rate_and_tolerance[1]
        if not is_sample_rate_as_expected:
            warnings.warn("sample rate is not as expected")

    if channels == None:
        sig = ContinuousDataEven(raw_sig, 1.0 / sample_rate, first_sample)
        return sig

    else:
        warnings.warn("reading multiple channels is not tested")
        sig_list = []
        for c in channels:
            sig_c = ContinuousDataEven(raw_sig[:, c], 1.0 / sample_rate, first_sample)
            sig_list.append(sig_c)

        return sig_list


    #return signal

def write_wav(contin, filename):
    """
    write contin to wav file, and return the units of the axis, and the first sample

    Note: I didn't think deeply about the signature of this function
    TODO: add way to rescale between the domain unit and sec

    example
    --------------
    s = continuous_data.read_wav("/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus.wav")
    s_cut = s[Segment([5.720, 6.610], uerg.sec)]
    continuous_data.write_wav(s_cut, "/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus_cut.wav")

    """
    if contin.domain_samples.dimensionality != uerg.sec.dimensionality:
        raise NotImplementedError
    else:
        sp.io.wavfile.write(filename, rate=contin.sample_rate.to(uerg.Hz).magnitude, data=contin.values.magnitude)


def fromfile(f):
    """
    read ContinuousData / ContinuousDataEven from file
    TODO:
    ---------
    decide about file format.
    probably a folder with wav file, and txt/csv/xml file for units
    etc
    """
    raise NotImplementedError


def test_read_wav():
    values = np.arange(10) * uerg.milliamp
    sample_rate = 1.0 * uerg.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    f_temp = tempfile.TemporaryFile()
    sp.io.wavfile.write(f_temp, sample_rate.magnitude, values.magnitude)
    sig_read = read_wav(f_temp)

    assert sig.is_close(sig_read)
    f_temp.close()

def test_write_wav():
    # copied from test_read_wav
    values = np.arange(10) * uerg.milliamp
    sample_rate = 1.0 * uerg.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    f_temp = tempfile.TemporaryFile()
    write_wav(sig, f_temp)
    sig_read = read_wav(f_temp)

    assert sig.is_close(sig_read)
    f_temp.close()

test_read_wav()
test_write_wav()


#%%

"""
some constructors of interesting signals

"""

def generate_const(sample_step, n_samples, value):
    raise NotImplementedError

def generate_sine(sample_step, n_samples, amplitude, sine_freq, phase_at_0=0, first_sample=0):
    """
    returns:
    a ContinuousDataEven which is a sine

    TODO: add DC parameter
    """
    if np.abs(phase_at_0) > 2 * np.pi:
        warnings.warn("you are using phase_at_0 not from [-2 pi, 2 pi], weird")
    if sine_freq > 0.5 * 1.0 / sample_step:
        raise("trying to generate undersampled sine signal, abbort! consider the nyquist!")
    t = np.arange(n_samples) * sample_step + first_sample
    phase = 2 * np.pi * sine_freq * t + phase_at_0
    sine = ContinuousDataEven(amplitude * np.sin(phase), sample_step, first_sample)
    return sine

def test_generate_sine():
    sample_step = 1 * uerg.sec
    n_samples = 128
    sine_freq = 0.15 * uerg.Hz
    amplitude = 1 * uerg.mamp
    expected_sine = ContinuousDataEven(amplitude * np.sin(2 * np.pi * sine_freq * sample_step * np.arange(n_samples)), sample_step)
    sine = generate_sine(sample_step, n_samples, amplitude, sine_freq)
    assert sine.is_close(expected_sine)

def generate_white_noise():
    raise NotImplementedError

def generate_square(sample_step, n_samples, amplitude, period, duty=0.5, phase_at_0=0, first_sample=0):
    """
    returns:
    a ContinuousDataEven which is suqare wave with min at zero and max at amplitude

    TODO: maybe add a parameter of base level.
    """
    if np.abs(phase_at_0) > 2 * np.pi:
        warnings.warn("you are using phase_at_0 not from [-2 pi, 2 pi], weird")
    if sample_step > min(duty * period, (1-duty) * period):
        warnings.warn("the sample step is larger then 'up time' or 'down time', you can miss some wave-fronts")
    t = np.arange(n_samples) * sample_step + first_sample
    phase = 2 * np.pi * 1.0 / period * t + phase_at_0
    square = ContinuousDataEven(amplitude * 0.5 * (1 + sp.signal.square(phase)), sample_step, first_sample)
    return square

def test_generate_square():
    sample_step = 1 * uerg.sec
    n_samples = 128
    period = 10 * uerg.sec
    amplitude = 1 * uerg.mamp
    expected_square = ContinuousDataEven(amplitude * 0.5 * (1 + sp.signal.square(2 * np.pi * 1.0 / period * sample_step * np.arange(n_samples))), sample_step)
    square = generate_square(sample_step, n_samples, amplitude, period)
    assert square.is_close(expected_square)
    #plot_quick(square)

def generate_square_freq_modulated(sample_step, n_samples, amplitude, sine_freq, period, duty=0.5, sine_phase_at_0=0, square_phase_at_t_0=0, first_sample=0):
    """
    returns:
    ContinuousDataEven which is a square wave modulated by sine. it's coherentic,
    means that all the "pulses" are taken from the same sine unstopped
    """
    envelope = generate_square(sample_step, n_samples, 1 * uerg.dimensionless, period, duty, square_phase_at_t_0, first_sample)
    sine = generate_sine(sample_step, n_samples, amplitude, sine_freq, sine_phase_at_0, first_sample)
    modulated = envelope * sine
    return modulated

def test_generate_square_freq_modulated():
    sample_step = 1 * uerg.sec
    n_samples = 2 ** 12
    sine_freq = 0.15 * uerg.Hz
    amplitude = 1 * uerg.mamp
    period = 100 * uerg.sec
    modulated = generate_square_freq_modulated(sample_step, n_samples, amplitude, sine_freq, period)
    envelope = generate_square(sample_step, n_samples, 1 * uerg.dimensionless, period)
    sine = generate_sine(sample_step, n_samples, amplitude, sine_freq)
    assert modulated.is_close(envelope * sine)


test_generate_sine()
test_generate_square()
test_generate_square_freq_modulated()
#%%

"""
non mathematical manipulations

"""



def concatenate(sig_list):
    """
    concatenate signals

    parameters:
    -------------------
    sig_list : list of ContinuousData

    returns:
    -----------------
    sig : ContinuousData
    """
    if len(sig_list) == 0:
        raise ValueError("no signals in the list")

    if len(sig_list) == 1:
        return sig_list[0]

    if not np.unique(map(type, sig_list))[0] == ContinuousDataEven:
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type")

    sample_steps = pint_extension.array(map(lambda(s) : s.sample_step, sig_list))
    sample_step = sig_list[0].sample_step
    if not pint_extension.allclose(sample_step, sample_steps):
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type, with same sample rate")



    first_samples = pint_extension.array(map(lambda(s) : s.first_sample, sig_list))
    last_samples = pint_extension.array(map(lambda(s) : s.last_sample, sig_list))

    gaps = first_samples[1:] - last_samples[:-1]
    if not pint_extension.allclose(sample_step, gaps):
        print gaps
        print sample_step
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type, with same sample rate, and right one after the other, at diffrence of sample_step")

    values = pint_extension.concatenate(map(lambda s : s.values, sig_list))
    sig = ContinuousDataEven(values, sample_step, sig_list[0].first_sample)
    return sig

def test_concatenate():
    sig_1 = ContinuousDataEven(np.arange(32) * uerg.mamp, uerg.sec)
    chunks = sig_1.get_chunks(15 * uerg.sec)
    """
    print len(chunks)
    print sig_1.values
    print chunks[0].values
    """
    sig_2 = concatenate(chunks)
    assert sig_1.is_close(sig_2)

test_concatenate()










#%%

"""
mathematical manipulations - except fouriers

"""

def diff(contin, n=1):
    """
    numeric differentiation of a ContinuousData
    a wrap around numpy.diff

    returns:
    ContinuousData of the same type, of the same same length
    for n == 1:
    all points except the last one are calculated using np.diff,
    the last one is defined to be like the one before it.

    TODO: Design issues:
    --------------
    it's not clean / beautiful definition for the last sample, but it hardly matters.
    I decided that it returns a ContinuousData of the same length, so it
    desn't hurt signals of length 2 ** m, which are easier to fft
    maybe it's better to return a signal that have samples in the middle between each two samples of the original signal
    """
    if type(contin) != ContinuousDataEven:
        raise NotImplementedError

    new_vals = np.empty(len(contin.values))
    if n != 1:
        raise NotImplementedError
    elif n == 1:
        new_vals[:-1] = np.diff(contin.values.magnitude, 1)
        new_vals[-1] = new_vals[-2]
        new_vals = new_vals * pint_extension.get_units(contin.values) * contin.sample_rate ** n

    return ContinuousDataEven(new_vals, contin.sample_step, contin.first_sample)

def test_diff():
    #copied from other test
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    expected_diffs = np.ones(10) * uerg.amp / uerg.sec
    expected_sig_diff = ContinuousDataEven(expected_diffs, sample_step)
    sig_diff = diff(sig)
    assert sig_diff.is_close(expected_sig_diff)


def correlate(sig_stable, sig_sliding, mode='valid'):
    """
    a correlation between 2 signals. we try to relocate the sliding sig, to fit the location of the stable sig

    parameters:
    --------------------
    sig_stable : ContinuousData

    sig_sliding : ContinuousData

    returns:
    -------------
    the correlation as signal. the peak of the correlation should inticate the bast location for the first sample of sig_sliding


    """
    warnings.warn("correlate is not tested")
    if not type(sig_stable) in [ContinuousDataEven,] or not type(sig_sliding) in [ContinuousDataEven,]:
        raise NotImplementedError("implemented only for ContinuousDataEven")

    if not pint_extension.allclose(sig_stable.sample_step, sig_sliding.sample_step):
        raise NotImplementedError("implemented only for same sample step signals")

    if sig_stable.n_samples < sig_sliding.n_samples:
        warnings.warn("note that sig_stable has less points then sig_sliding, why is thay?")

    # values
    a = sig_stable.values.magnitude
    b = sig_sliding.values.magnitude
    c = np.correlate(a, b, mode)
    sig_c_values = c * pint_extension.get_units(sig_stable.values) * pint_extension.get_units(sig_sliding.values) * sig_stable.sample_step

    #times
    if mode == 'full':
        first_sample = (-1) * sig_stable.sample_step * (-1 + 0.5 * (sig_stable.n_samples + sig_sliding.n_samples)) + sig_stable.first_sample
    elif mode == 'same':
        raise NotImplementedError("timing the correlation not implemented for same mode")
    elif mode == 'valid':
        raise NotImplementedError("timing the correlation not implemented for valid mode")

    sig_c = ContinuousDataEven(sig_c_values, sig_stable.sample_step, first_sample)
    return sig_c

def visual_test_correlate():
    v = np.concatenate([np.arange(10), np.arange(10)[::-1]])
    sig_stable = ContinuousDataEven(v * uerg.mamp, uerg.sec, 10 * uerg.sec)
    sig_sliding = ContinuousDataEven(v * uerg.mamp, uerg.sec, 20 * uerg.sec)
    sig_c = correlate(sig_stable, sig_sliding, mode='full')
    plot_quick(sig_stable, "o")
    plot_quick(sig_sliding, "o")
    plot_quick(sig_c, "o")


def correlate_find_new_location(sig_stable, sig_sliding, mode='valid', is_return_max=False):
    """
    for most of the documentation refer to correlate
    TODO: the signature of this function is not stable, according to user input it returns either 1 or 2 values

    parameters:
    --------------------
    is_return_max : bool
        can return also the max value, in order to compare the success of different correlations



    behind the scences:
    ----------------------
    using correlate and np.argmax
    """
    corr = correlate(sig_stable, sig_sliding, mode)
    top_index = np.argmax(corr.values)
    top_domain_sample = corr.first_sample + corr.sample_step * top_index
    max_value = corr.values[top_index]

    if not is_return_max:
        return top_domain_sample
    else:
        return top_domain_sample, max_value

def test_correlate_find_new_location():
    v = np.concatenate([np.arange(10), np.arange(10)[::-1]])
    sig_stable = ContinuousDataEven(v * uerg.mamp, uerg.sec, 10 * uerg.sec)
    sig_sliding = ContinuousDataEven(v * uerg.mamp, uerg.sec)
    new_location, max_val = correlate_find_new_location(sig_stable, sig_sliding, 'full', is_return_max=True)
    print new_location
    expected_new_location = 10 * uerg.sec
    expected_max_val = 2 * (np.arange(10) ** 2).sum() * uerg.mamp ** 2 * uerg.sec
    assert pint_extension.allclose(new_location, expected_new_location)
    assert pint_extension.allclose(max_val, expected_max_val)

def clip(sig, values_range):
    """
    parameters:
    ---------------------
    sig : ContinuousData

    values_range : Segment

    """
    if type(sig) != ContinuousDataEven:
        raise NotImplementedError

    clipped_vals = np.clip(sig.values, values_range.start, values_range.end)
    clipped = ContinuousDataEven(clipped_vals, sig.sample_step, sig.first_sample)
    return clipped

def test_clip():
    v = np.arange(10) * uerg.mamp
    sig = ContinuousDataEven(v, uerg.sec)
    Range = Segment([3, 6], uerg.mamp)
    clipped = clip(sig, Range)
    expected_clipped = ContinuousDataEven(np.clip(v, 3 * uerg.mamp, 6 * uerg.mamp), uerg.sec)
    assert clipped.is_close(expected_clipped)

test_diff()
#visual_test_correlate()
test_correlate_find_new_location()
test_clip()
#%%




"""
fourier, and demodulations
TODO: there are some problematic issues with fft / hilbert /demodulations with not 2 ** n samples signals.
"""




def determine_fft_len(n_samples, mode='accurate'):
    """
    helper function to determine the number of samples for a fft
    if mode is not 'accurate', it's a power of 2

    parameters:
    --------------
    n_samples : int
    mode : str
        'accurate' like n
        'trim' - smaller then n
        'zero-pad' - bigger then n
        'closer' - either trim or zero pad, depends which is closer (logarithmic scale)
    """
    modes_dict = {'trim': 'smaller', 'zero-pad' : 'bigger', 'fast' : 'closer'}
    if mode == 'accurate':
        n_fft = n_samples
    else:
        n_fft = numpy_extension.close_power_of_2(n_samples, modes_dict[mode])

    return n_fft

def test_determine_fft_len():
    assert determine_fft_len(14, 'accurate') == 14
    assert determine_fft_len(14, 'fast') == 16
    assert determine_fft_len(7, 'trim') == 4
    assert determine_fft_len(5, 'zero-pad') == 8

test_determine_fft_len()

#%%
def fft(contin, n=None, mode='accurate'):
    """
    fft of a ContinuousData instance.
    implemented only for ContinuousDataEven
    a wrap arround np.fft.fft

    parameters:
    ----------------
    n : int
        number of samples for fft

    mode : str
        copied from determine_fft_len
        'accurate' like n
        'trim' - smaller then n
        'zero-pad' - bigger then n
        'closer' - either trim or zero pad, depends which is closer (logarithmic scale)

    returns: a ContinuousDataEven object that represents the spectrum
    the frequencies are considerred from -0.5 nyq frequency to 0.5 nyq frequency
    """
    # shoult insert a way to enforce "fast", poer of 2 stuff
    n_sig = len(contin.values)
    # maybe the process deciding the fft len should be encapsulated

    if not n:
        n = determine_fft_len(n_sig, mode)

    freq_step = 1.0 * contin.sample_rate / n
    first_freq = - 0.5 * contin.sample_rate

    spectrum = np.fft.fftshift(np.fft.fft(contin.values.magnitude, n))
    spectrum = spectrum * pint_extension.get_units(contin.values) * contin.sample_step

    return ContinuousDataEven(spectrum, freq_step, first_freq)

def test_fft():
    sig = ContinuousDataEven(np.arange(32) * uerg.amp, 1.0 * uerg.sec)
    expected_freqs = np.fft.fftshift(np.fft.fftfreq(32)) / uerg.sec
    expected_freqs_vals = np.fft.fftshift(np.fft.fft(np.arange(32))) * uerg.amp * uerg.sec
    expected_spec = ContinuousData(expected_freqs_vals, expected_freqs)
    spec = fft(sig)

    assert spec.is_close(expected_spec)

    #mostly a copy of the other test
    sig = ContinuousDataEven(np.arange(31) * uerg.amp, 1.0 * uerg.sec)
    expected_freqs_fast = np.fft.fftshift(np.fft.fftfreq(32)) / uerg.sec
    expected_freqs_vals_fast = np.fft.fftshift(np.fft.fft(np.arange(31), 32)) * uerg.amp * uerg.sec
    expected_spec_fast = ContinuousData(expected_freqs_vals_fast, expected_freqs_fast)
    spec_fast = fft(sig, mode='fast')

    assert spec_fast.is_close(expected_spec_fast)


test_fft()
#%%





"""
def freq_filter(contin, freq_ranges, ?, ?, ?):
    raise NotImplementedError


    @uerg.wraps(None, (None, uerg.Hz, uerg.Hz, None, None, None, uerg.Hz))
    def firwin_pint(numtaps, cutoff, width, window, pass_zero, scale, nyq):
        return sp.signal.firwin(numtaps, cutoff, width, window, pass_zero, scale, nyq)
"""

def band_pass_filter(sig, freq_range, mask_len):
    """
    band pass filter of ContinuousDataEven

    parameters:
    freq_range: a Segment of frequencies

    implemented using np.convolve with a mask. maybe with fft is better

    XXX TODO: shuold change interface. maks len is something without units. shoult receive something else
    XXX this function is not stable with downsampling for this reason
    """
    warnings.warn('not tested well')
    #TODO: test well
    freq_range.edges.ito(sig.sample_rate.units)
    print freq_range.edges
    print sig.sample_rate
    assert freq_range.end.magnitude < 0.5 * sig.sample_rate.magnitude
    # if error rises with firwin with units, wrap it: http://pint.readthedocs.org/en/0.5.1/wrapping.html
    mask_1 = sp.signal.firwin(mask_len, freq_range.edges.magnitude, pass_zero=False, nyq=0.5 * sig.sample_rate.magnitude)
    filterred_values = scipy_extension.smart_convolve(sig.values.magnitude, mask_1, mode="same") * pint_extension.get_units(sig.values)
    filterred = ContinuousDataEven(filterred_values, sig.sample_step, sig.first_sample)
    return filterred

def test_band_pass_filter():
    sample_step = uerg.sec
    np.random.seed(13)
    white_noise = ContinuousDataEven((np.random.rand(2048) - 0.5)* uerg.mamp, sample_step)
    white_noise_spec = fft(white_noise)
    freq_range = Segment(np.array([0.3, 0.4]) * uerg.Hz)
    white_noise_filterred = band_pass_filter(white_noise, freq_range, 32)
    white_noise_filterred_spec = fft(white_noise_filterred)
    plot_quick(white_noise_spec, is_abs=True)
    plot_quick(white_noise_filterred_spec, is_abs=True)




# test_band_pass_filter()


#%%




def hilbert(sig, mode='fast'):
    """
    returns the analytic signal
    a wrap around sp.signal.hilbert
    """
    n_fft = determine_fft_len(sig.n_samples, mode)
    analytic_sig_values = sp.signal.hilbert(sig.values.magnitude, n_fft) * pint_extension.get_units(sig.values)
    new_sample_step = 1.0 * sig.sample_step * sig.n_samples / n_fft
    analytic_signal = ContinuousDataEven(analytic_sig_values, new_sample_step, sig.first_sample)
    return analytic_signal

def test_hilbert():
    # copied from test_pm_demodulation
    sample_step = 1.0 * uerg.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * uerg.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * uerg.mamp, sample_step)
    # that is what I would expect, but when I try to fft a sine, I get both real and imaginary values for amps of each freq. weird
    # expected_sine_hilbert = ContinuousDataEven((-1) * 1j *np.exp(1j * phase) * uerg.mamp, sample_step)
    expected_sine_hilbert = ContinuousDataEven(sp.signal.hilbert(np.sin(phase)) * uerg.mamp, sample_step)
    sine_hilbert = hilbert(sine)
    """
    plot_quick(sine)
    plot_quick(fft(sine), is_abs=True)
    plot_quick(fft(sine_hilbert), is_abs=True)
    plot_quick(fft(expected_sine_hilbert), is_abs=True)
    """
    assert sine_hilbert.is_close(expected_sine_hilbert)


def pm_demodulation(sig, mode='fast'):
    """
    based on hilbert transform.
    the pm demodulation at the edges is not accurate.
    TODO: map how much of the edges is a problem
    TODO: maybe it should return only the time without the edges.
    TODO: how to improve the pm demodulation at the edges?
    TODO: maybe should add a "n_fft" parameter
    TODO: maybe it's better to allow calculation of phase with separation to windows?
    """
    if True:
        warnings.warn("pm-demodulation is not tested well on signals that are not 2**n samples")
    if sig.n_samples < 2 ** 10:
        warnings.warn("this pm-modulation technique doesn't work well on short signals, the mistakes on the edges are big")
    analytic_sig = hilbert(sig, mode)
    phase_wrapped = np.angle(analytic_sig.values.magnitude)
    phase = np.unwrap(phase_wrapped) * uerg.dimensionless
    return ContinuousDataEven(phase, analytic_sig.sample_step, analytic_sig.first_sample)

def test_pm_demodulation():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * uerg.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * uerg.mamp, sample_step)
    expected_phase_sig = ContinuousDataEven(phase, sample_step)
    phase_sig = pm_demodulation(sine)
    assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)

    time = np.arange(2 ** 15 - 100) * sample_step
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * uerg.mamp, sample_step)
    expected_phase_sig = ContinuousDataEven(phase, sample_step)
    phase_sig = pm_demodulation(sine)
    print phase_sig.first_sample, phase_sig.last_sample
    print expected_phase_sig.first_sample, expected_phase_sig.last_sample
    # weird, it acctually gives phase diff of 0.5 pi from what I expect
    assert pint_extension.allclose(phase_sig.first_sample, expected_phase_sig.first_sample)
    assert pint_extension.allclose(phase_sig.last_sample, expected_phase_sig.last_sample, atol=min(phase_sig.sample_step, expected_phase_sig.sample_step))
    #fig, junk = plot_quick(expected_phase_sig)
    #plot(phase_sig, fig)
    #assert pint_extension.allclose(phase_sig.sample_step, expected_phase_sig.sample_step)
    #assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)

def fm_demodulation(sig, mode='fast'):
    """
    fm demodulation
    based on differentiating the pm demodulation
    """
    sig_phase = pm_demodulation(sig, mode)
    angular_freq = diff(sig_phase)
    freq = angular_freq.gain(1.0 / (2 * np.pi))
    return freq

def test_fm_demodulation():
    # copied from test_pm_demodulation
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * uerg.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * uerg.mamp, sample_step)
    expected_freq_sig = ContinuousDataEven(np.ones(2 ** 15) * freq, sample_step)
    freq_sig = fm_demodulation(sine)
    assert freq_sig[check_range].is_close(expected_freq_sig[check_range], values_rtol=0.01)

def am_demodulation_hilbert(sig, mode='fast'):
    #worning copied from pm_demodulation
    if sig.n_samples < 2 ** 10:
        warnings.warn("this pm-modulation technique doesn't work well on short signals, the mistakes on the edges are big")
    analytic_sig = hilbert(sig, mode)
    envelope = np.abs(analytic_sig.values.magnitude) * pint_extension.get_units(analytic_sig.values)
    sig_am = ContinuousDataEven(envelope, analytic_sig.sample_step, analytic_sig.first_sample)
    return sig_am

def test_am_demodulation_hilbert():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq = 0.15 * uerg.Hz
    amp = uerg.mamp
    sine = generate_sine(sample_step, n_samples, amp, sine_freq=freq)
    expected_sine_am = ContinuousDataEven(np.ones(sine.n_samples) * amp, sample_step)
    sine_am = am_demodulation_hilbert(sine)
    """
    plot_quick(sine)
    plot_quick(sine_am)
    plot_quick(expected_sine_am)
    """
    assert sine_am[check_range].is_close(expected_sine_am[check_range], values_rtol=0.01)

    """
    this test fails now, it needs is_close_l_1 to work properly
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_hilbert(sig)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)
    """


def am_demodulation_convolution(sig, t_smooth):
    """
    params:
    t_smooth is the width in domain units, that you want to smooth together
    """
    warnings.warn("not tested well")
    n_samples_smooth = np.ceil(t_smooth * sig.sample_rate)
    mask_am = numpy_extension.normalize(np.ones(n_samples_smooth), ord=1)
    values_am = np.convolve(np.abs(sig.values.magnitude), mask_am, mode="same") * pint_extension.get_units(sig.values)
    return ContinuousDataEven(values_am, sig.sample_step, sig.first_sample)

def test_am_demodulation_convolution():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq_1 = 0.15 * uerg.Hz
    freq_2 = 0.40 * uerg.Hz
    amp = uerg.mamp
    sine_1 = generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    sine_2 = generate_sine(sample_step, n_samples, amp, freq_2)
    sig = sine_1 + sine_2
    """
    #copied from test_am_demodulation_filter
    dt = 1.0 / freq_1 * 0.25
    am = am_demodulation_convolution(sig, dt)
    fig, junk = plot(sig)
    plot(sine_1, fig)
    plot(am, fig)
    plot_quick(sine_1 - am)
    assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * uerg.mamp)
    """
    dt = 1.0 / freq_1 * 3
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_convolution(sig, dt)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)


def am_demodulation_filter(sig, dt_smooth, mask_len):
    warnings.warn("not tested well")
    top_freq = 1.0 / dt_smooth
    band = Segment([1e-12 * pint_extension.get_units(top_freq), top_freq])
    return band_pass_filter(sig.abs(), band, mask_len = mask_len)


def test_am_demodulation_filter():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq_1 = 0.15 * uerg.Hz
    freq_2 = 0.40 * uerg.Hz
    amp = uerg.mamp
    dt = 1.0 / freq_1 * 0.5
    """
    sine_1 = generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    sine_2 = generate_sine(sample_step, n_samples, amp, freq_2)
    sig = sine_1 + sine_2


    am = am_demodulation_filter(sig, dt, 128)
    fig, junk = plot(sig)
    plot(sine_1.abs(), fig)
    plot(am, fig)
    plot_quick(sine_1.abs() - am)
    assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * uerg.mamp)
    """

    dt = 1.0 / freq_1 * 3
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_filter(sig, dt, 256)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)

test_hilbert()
test_pm_demodulation()
test_fm_demodulation()
test_am_demodulation_hilbert()
#test_am_demodulation_convolution()
#test_am_demodulation_filter()

#%%
def filter_downsample_fm_demodulation_base_band_filter(raw_sig, freq_range, freq_mask_len, down_factor, base_band_range, base_band_mask_len):
    """
    preprocessing of the signal.

    returns:
    -------------
    signal : ContinuousDataEven
        signal after filtering, de-fm

    """
    warnings.warn("filter_downsample_fm_demodulation_base_band_filter not tested")
    # step_2: filtering the raw_data
    # Note:  getting memory error with long signals (290 M-Byte and more). the problem is with the convolution. need to try fftconvolve
    # maybe we want to filter, only after finding where are the traces, and with the exact band of the real raw_sig (not noise)
    filterred = band_pass_filter(raw_sig, freq_range, freq_mask_len)

    if False:
        plot_quick(filterred[parameters.time_for_quick_plot])

    # step_3: down sampling, (which also aliases, and changed the freq of the signal)
    # XXX: maybe the frequency band is close to the +0.5 nyq, so the fm
    # signal would be split between +0.5nyq and -0.5nyq
    # TODO: cope with the freq wrap problems. acctually np.unwrap could
    # be a fine solution, after the fm-demosulation.
    # another approach - calculate where the signal should be "falling"
    # it's a modular / sawtooth calculation. if it's "splinched", then
    # just add and use modulu (roll it)
    down = filterred.down_sample(down_factor)

    if False:
        plot_quick(down)
        plot_quick(fft(down), is_abs=True)

    # step_4: fm demodulation
    de_fm = fm_demodulation(down, mode='fast')

    if False:
        plot_quick(de_fm)

    # step 4.1 filter the base-band signal
    sig = band_pass_filter(de_fm, freq_range=base_band_range, mask_len=base_band_mask_len)
    return sig



#%%
def resample(sig, new_sample_points):
    """
    create a new sig object, that represents the same signal, on different sample points.
    algorithm: linear intrapulation
    """
    raise NotImplementedError





