# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 01:29:39 2014

@author: noam
"""

import numpy as np
import pint_extension

class ContinuousData(object):
    """
    this class represents any kind of continuous data (one dimensional).
    It includes a few kinds that first seem different from each other, has a lot in common.
    
    examples:
    1. a "signal" - measurement of the electromagnetic field / voltage as a function of time.
    similar examples: sound (as a function of time), seismic measurments.
    2. a spatial measurement: height as a function of place, some material density as a function of place.
    similar examples: stress as a function of place.
    3. a distribution: the number of occurances in a population, as a function of age / height / amplitude
    4. even a spectrum of a signal - the magnitude as a function of frequency.
    
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
    
    """
    def __init__(self, domain_samples, values):
        self._domain_samples = domain_samples
        self._values = values
        
    @property
    def domain_samples(self):
        return self._domain_samples
        
    @property
    def values(self):
        return self._values
        
    def __len__(self):
        return self.domain_samples.ptp()
        
    def __getitem__(self, domain_range):
        is_each_in_range = self.domain_samples in domain_range
        return ContinuousData(self.domain_range[is_each_in_range], self.values[is_each_in_range])
        
    def is_close(self, other, domain_rtol=None, domain_atol=None, values_rtol=None, values_atol=None):
        return pint_extension.allclose(self.domain_samples, other.domain_samples, domain_rtol, domain_atol) \
        and pint_extension.allclose(self.values, other.values, values_rtol, values_atol)
        
        
        
    def DFT(self):
        raise NotImplementedError
        # maybe there is an issue regarding using DFT or IDTF, depending the domain
        # maybe it should be an extra param. seying which one to use

  


def plot(contin):
    # should be more  inputs
    # maybe optional parameter of on which figure / axes to plot
    assert contin type?
    raise NotImplementedError
        # return fig, axes??
def plot_quick(contin):
    # creat the figure here
    
class ContinuousDataEven(ContinuousData):
    """
    read the ContinuousData documentation.
    the domain samples are evenly spaced
    """
    def __init__(self, values, sample_step, first_sample=0):
        # would there be a problem because of interface issues?
        self._sample_step = sample_step
        self._first_sample = first_sample
        self._values = values
        
    @property
    def sample_step(self):
        return self._sample_step
        
    @property
    def sample_rate(self):
        return 1.0 / sample_step
    
    @property
    def first_sample(self):
        return self._first_sample
        
    @property
    def domain_samples(self):
        return np.arange(len(self.values)) * self.sample_step + first_sample
        
    def __getitem__(self, domain_range):
        bottom_index = np.ceil(domain_range.start / self.sample_step)
        top_index = np.floor(domain_range.end / self.sample_step)
        return ContinuousDataEven(self.sample_step, self.values[bottom_index:top_index])
        
    def FFT(self, ?):
        raise NotImplementedError
        spectrum = 
        freq_step = 
        return ContinuousDataEven(freq_step, spectrum)
        
def read_wav(filename, domain_unit=uerg.sec, value_unit=uerg.volt, expected_sample_rate=None, sample_rate_tolerance=None):
    raise NotImplementedError
    return signal
    
def freq_filter(contin, freq_ranges, ?, ?, ?):
    raise NotImplementedError
    
    """   
    @uerg.wraps(None, (None, uerg.Hz, uerg.Hz, None, None, None, uerg.Hz))    
    def firwin_pint(numtaps, cutoff, width, window, pass_zero, scale, nyq):
        return sp.signal.firwin(numtaps, cutoff, width, window, pass_zero, scale, nyq)
    """