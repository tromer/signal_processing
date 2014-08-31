# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 01:29:39 2014

@author: noam
"""

import warnings
import tempfile
#%%
import numpy as np
import scipy as sp
from scipy.io import wavfile
import matplotlib.pyplot as plt

from pint import UnitRegistry
uerg = UnitRegistry()

from Range import Range
import pint_extension


#%%

ARBITRARY_UNITS_STR = "[AU]"
#%%

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
    4. a kinematic property of a system: position / velocity / acceleration / angle as a function of time.
    5. even a spectrum of a signal - the magnitude as a function of frequency.
    
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
        
    def is_close(self, other, domain_rtol=1e-5, domain_atol=None, values_rtol=1e-5, values_atol=None):
        return pint_extension.allclose(self.domain_samples, other.domain_samples, domain_rtol, domain_atol) \
        and pint_extension.allclose(self.values, other.values, values_rtol, values_atol)

    """
    # maybe len should really return the number of sample points
    # I am not shure whether the number of sample points should be a part of the interface/
    # but in many implementations of functions I need it, so use len(contin.values)
    def __len__(self):
        raise NotImplementedError
        return self.domain_samples.ptp()
    """
    
    def __getitem__(self, domain_range):
        is_each_in_range = domain_range.is_each_in(self.domain_samples)
        return ContinuousData(self.values[is_each_in_range], self.domain_samples[is_each_in_range])
        
    
    def DFT(self):
        raise NotImplementedError
        # maybe there is an issue regarding using DFT or IDTF, depending the domain
        # maybe it should be an extra param. seying which one to use
        # maybe should be an external function, not a method

def test_ContinuousData():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    assert pint_extension.allclose(sig.domain_samples, t)
    assert pint_extension.allclose(sig.values, vals)
    
    assert sig.is_close(sig)
    assert not sig.is_close(ContinuousData(vals, t + 1 * uerg.sec))
    assert not sig.is_close(ContinuousData(vals + 1 * uerg.volt, t))

    t_range = Range(np.array([2.5, 6.5]) * uerg.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousData(vals[expected_slice], t[expected_slice])
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)
    
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
    fig = plt.figure()
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
#%%

def plot(contin):
    # should be more  inputs
    # maybe optional parameter of on which figure / axes to plot
    # assert contin type?
    raise NotImplementedError
        # return fig, axes??

    
class ContinuousDataEven(ContinuousData):
    """
    read the ContinuousData documentation.
    the domain samples are evenly spaced
    """
    def __init__(self, values, sample_step, first_sample=0):
        # would there be a problem because of interface issues?
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
    def domain_samples(self):
        #print "******"
        #print self.values
        return np.arange(len(self.values)) * self.sample_step + self.first_sample
        
    def __getitem__(self, domain_range):
        bottom_index = np.ceil(1.0 * domain_range.start / self.sample_step)
        top_index = np.floor(domain_range.end / self.sample_step)
        return ContinuousDataEven(self.values[bottom_index:top_index + 1], self.sample_step, first_sample=bottom_index * self.sample_step)

def test_ContinuousDataEven():
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    assert pint_extension.allclose(sig.sample_step, sample_step)
    assert pint_extension.allclose(sig.sample_rate, 1.0 / sample_step)
    assert pint_extension.allclose(sig.values, values)
    assert pint_extension.allclose(sig.domain_samples, np.arange(10) * sample_step)
    assert sig.is_close(ContinuousData(values, np.arange(10) * sample_step))
    assert pint_extension.allclose(sig.first_sample, 0 * sample_step)
    
    # testing a __getitem__ (slicing) is mostly copied from the tester of ContinuousData
    t_range = Range(np.array([2.5, 6.5]) * uerg.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousDataEven(values[expected_slice], sample_step, expected_slice[0] * sample_step)
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)
    
test_ContinuousDataEven()
#%%
def fft(contin, n=None):
    # shoult insert a way to enforce "fast", poer of 2 stuff
    if not n:
        n = len(contin.values)
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
    
test_fft()
#%%
def diff(contin, n=1):
    """
    a wrap around numpy.diff
    returns a signal of same length
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
    
test_diff()
#%%
    
"""
def freq_filter(contin, freq_ranges, ?, ?, ?):
    raise NotImplementedError
    
     
    @uerg.wraps(None, (None, uerg.Hz, uerg.Hz, None, None, None, uerg.Hz))    
    def firwin_pint(numtaps, cutoff, width, window, pass_zero, scale, nyq):
        return sp.signal.firwin(numtaps, cutoff, width, window, pass_zero, scale, nyq)
"""

def read_wav(filename, domain_unit=uerg.sec, first_sample=0, value_unit=uerg.milliamp, expected_sample_rate_and_tolerance=None):
    sample_rate, raw_sig = sp.io.wavfile.read(filename)
    sample_rate = 1.0 * sample_rate / domain_unit
    raw_sig = raw_sig * value_unit
    if expected_sample_rate_and_tolerance != None:
        # shold raise a meaningful excepion.
        assert np.abs(sample_rate - expected_sample_rate_and_tolerance[0] < expected_sample_rate_and_tolerance[1])
    
    sig = ContinuousDataEven(raw_sig, 1.0 / sample_rate, first_sample)
    return sig
    #return signal
   
def test_read_wav():
    values = np.arange(10) * uerg.milliamp
    sample_rate = 1.0 * uerg.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)
    
    f_temp = tempfile.TemporaryFile()
    sp.io.wavfile.write(f_temp, sample_rate.magnitude, values.magnitude)
    sig_read = read_wav(f_temp)
    
    assert sig.is_close(sig_read)
    f_temp.close()
    
test_read_wav()
    
    
#%%
    

