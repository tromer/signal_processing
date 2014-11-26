# -*- coding: utf-8 -*-
"""
.. module:: continuous_data_obj
    :synopsis: bla
"""


"""
"""
"""
from . import uerg, Q_
from . import Segment
from . import segments
from . import  Segments
from . import numpy_extension
from . import scipy_extension
from . import pint_extension
"""

"""
from signal_processing import uerg, Q_
from signal_processing.segment import Segment
from signal_processing import segments
from signal_processing.segments import Segments
from signal_processing.extensions import numpy_extension, scipy_extension, pint_extension
"""

from signal_processing.extensions import pint_extension
from signal_processing.extensions import plt_extension
from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments

#%%

#%%

class ContinuousData(object):
    """

    this class represents any kind of continuous data (one dimensional).
    It includes a few kinds that first seem different from each other, has a lot in common.

    **examples:**

    1. a "signal":
        measurement of the electromagnetic field / voltage as a function of time.
        similar examples: sound (as a function of time), seismic measurments.

    2. a spatial measurement:
        height as a function of place, some material density as a function of place.
        similar examples: stress as a function of place.

    3. a distribution:
        the number of occurances in a population, as a function of age / height / amplitude

    4. a kinematic property of a physical system:
        position / velocity / acceleration / angle as a function of time.

    5. spectrum of a signal:
        the magnitude as a function of frequency.

    6. the most general case:
        any connection between two continuous variables.
        example: a response curv of harmonic ocsillator:
        amplitude of the ocsillator as a function of the frequency of external force.

    There are some differences beween these kinds of data. Maybe some of them would be implemented as a subclass

    **basic assumptions:**

    1. the acctual data in the real world can be really continuous.
    here we of course use sample points. every sample point has exectly one corresponding value.

    2. We assume that our data really representes the reallity, spesifically, that
    represents well also the times / places which we didn't measure.
    So, we assume that the data is not changing "too fast" compared to our resolution.
    In signal processing terms, we assume that we didn't under-sample.

    notes
    -----------

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


    **design issues**

    1. this object is immutable, appart from changing the string describing
    the domain and values.

    2. in many cases, functions that except ContinuousData extract the values, do some mathematical operation, and construct a new object with the same domain.\n
    when constructing them they accept self.domain_samples.\n
    however, this becomes harder later: in ContinuousDataEven, the domain samples are evenly sampled, and thus need two variables for representation: sample_step and n_samples.\n
    as I add a domain_description property, it becomes uglier to move all these parameters from instance to constructor.\n
    A possible solution is defining a DomainSamples object, with sub-class DomainSamplesEven. \n
    a ContinuousData will hold one of those, and this object would be the interface to the domain.\n
    This would save code, and also enable writing the similar functions that accept both ContinuousData and ContinuousDataEven (in their interface).

    3. this solution has one aspect which I don't like: breaking the symetry between domain_samples and values. in a way they are really not symetrical:\n
    in every program there would be many instances sharing the same domain, but usually non sharing values.\n
    also, evenly samples domains are common, but evenly samples values are just linear data, and uncommon.\n
    What I don't like is breaking symetry in the sytax. It would be like:\n
    sig.domain.samples, sig.domain.description as opposed to:\n
    sig.values, sig.values_description.\n

    4. In order to restore symetry, we could also work with a Values object.\n
    the problem is that sig.values call is very very common in the functions of the package, and it would be pity to lenghen it to sig.values.samples

    See Also
    ----------
    signal_processing.Segments:  object that describe segments of interest within a `ContinuousData`
    signal_processing.SegmentsOfContinuous


    .. todo::
        1. maybe I want to implement a domain_samples object. it would have
        a subclass of even samples

        2. maybe it's smart to implement a similar object with few channels.
        It may be useful in some implementation and performance issues,
        since the channels would be a 2D np.ndarray, and channel-wise
        operations like fft would be applied along axis, and be
        efficient.

        design issue
        3. maybe add a self.base attribute, like in np.ndarrays

    .. todo::
        create a constructor from a np.ndarray of shape (N, 2)

    """
    def __init__(self, values, domain_samples, values_des=None, domain_des=None):
        assert len(values) == len(domain_samples)
        self._domain_samples = domain_samples
        self._values = values
        self._domain_description = domain_des
        self._values_description = values_des

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
    def domain_start(self):
        return self.domain_samples[0]

    @property
    def domain_end(self):
        return self.domain_samples[-1]

    @property
    def domain_range(self):
        range_ = Segment([self.domain_start, self.domain_end])
        return range_

    @property
    def domain_unit(self):
        return pint_extension.get_units(self.domain_samples)

    @property
    def values_unit(self):
        return pint_extension.get_units(self.values)


    @property
    def domain_description(self):
        if self._domain_description == None:
            return pint_extension.get_dimensionality_str(self.domain_unit)
        else:
            return self._domain_description

    @domain_description.setter
    def domain_description(self, value):
        self._domain_description = value

    @property
    def values_description(self):
        # TODO : copied from domain_description
        if self._values_description == None:
            return pint_extension.get_dimensionality_str(self.values_unit)
        else:
            return self._values_description

    @values_description.setter
    def values_description(self, value):
        self._values_description = value


    def describe(self, domain, values):
        self.domain_description = domain
        self.values_description = values

    def __str__(self):
        line_1 = "domain: " + self.domain_description + str(self.domain_range)
        line_2 = "values: " + self.values_description

        self_str = "\n".join([line_1, line_2])
        return self_str

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
        parameters
        -------------
        domain_range : Segment
            the range, from the domain, of which we want the slice.
            for example: which time range?

        TODO: since the domain samples should be sorted, maybe there
        is a more efficient implementation

        design issue
        ------------------
        in case the key is Segments
        maybe it should return a SegmentsOfContinuous instance, instead of a
        list of signals
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

    def _prepare_data_labels_for_plot(self):
        """
        this is a thin interface to pint_extension.prepare_data_and_labels_for_plot
        """
        data_labels_for_plot = \
            pint_extension.prepare_data_and_labels_for_plot(
                self.domain_samples,
                self.values,
                self.domain_description,
                self.values_description)

        return data_labels_for_plot

    def plot(self, fig=None, domain_range=None, is_show_legend=True):
        """
        basic plot

        parameters
        --------------
        fig : figure

        domain_range : Segment
            if you want to plot only a certain range from the signal

        returns
        ----------
        the fig only, for plotting other signals on top
        """
        if domain_range is None:
            sig_for_plot = self
        else:
            sig_for_plot = self[domain_range]

        x_bare, y_bare, x_label, curv_label =\
            sig_for_plot._prepare_data_labels_for_plot()

        return plt_extension.plot_with_labels(
            x_bare, y_bare, x_label, curv_label, fig, is_show_legend)[0]

    def tofile(self, f):
        """
        read the docs of fromfile
        """
        raise NotImplementedError


