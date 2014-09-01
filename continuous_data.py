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

from global_uerg import uerg


from Range import Range
import numpy_extension
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
        
    @property
    def n_samples(self):
        return len(self.values)
        
    @property
    def first_sample(self):
        return self.domain_samples[0]
    
    @property     
    def last_sample(self):
        return self.domain_samples[-1]
        
        
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
        
    def gain(self, factor):
        """
        multiplies the values by the factor
        """
        raise NotImplementedError
        
    def __add__(self, other):
        raise NotImplementedError
    
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
    assert sig.n_samples == 10
    assert sig.first_sample == 0 * uerg.sec
    assert sig.last_sample == 9 * uerg.sec
    
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
    return plot(contin, fig=None, is_abs=is_abs, fmt=fmt)
#%%

def plot(contin, fig=None, is_abs=False, fmt="-"):
    # assert contin type?
    warnings.warn("plot is not tested")
    warnings.warn("plot dosn't rescale the last signal according to axes")
    
    if fig == None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
    
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

#%%    
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
        
    def is_same_domain_samples(self, other):
        return self.n_samples == other.n_samples and \
        pint_extension.allclose(self.first_sample, other.first_sample) and \
        pint_extension.allclose(self.sample_step, other.sample_step)
        
    def __add__(self, other):
        if self.is_same_domain_samples(other):
            return ContinuousDataEven(self.values + other.values, self.sample_step, self.first_sample)
            
        else:
            raise NotImplementedError
            
    def __sub__(self, other):
        if self.is_same_domain_samples(other):
            return ContinuousDataEven(self.values - other.values, self.sample_step, self.first_sample)
            
        else:
            raise NotImplementedError
        
        
    def gain(self, factor):
        """
        see doc of base class
        """
        return ContinuousDataEven(self.values * factor, self.sample_step, self.first_sample)
        
    def down_sample(self, down_factor):
        assert down_factor > 0
        assert int(down_factor) == down_factor
        # maybe there should be another interface, with "new sample rate"
        return ContinuousDataEven(self.values[::down_factor], down_factor * self.sample_step, self.first_sample)

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

def test___add__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    assert (sig + sig).is_close(sig.gain(2))
    
def test___sub__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    sig_2 = ContinuousDataEven(np.ones(10) * uerg.mamp, uerg.sec)
    dif = ContinuousDataEven(np.arange(-1,9) * uerg.mamp, uerg.sec)
    assert (sig - sig_2).is_close(dif)
    
test_ContinuousDataEven()
test_down_sample()
test_gain()
test_is_same_domain_samples()
test___add__()
test___sub__()

#%%
def determine_fft_len(n_samples, mode='accurate'):
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
def fft(contin, n=None, mode='fast'):
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

def band_pass_filter(sig, freq_range, mask_len):
    warnings.warn('not tested well')
    #TODO: test well
    freq_range.edges.ito(sig.sample_rate.units)
    print freq_range.edges
    print sig.sample_rate
    assert freq_range.end.magnitude < 0.5 * sig.sample_rate.magnitude
    # if error rises with firwin with units, wrap it: http://pint.readthedocs.org/en/0.5.1/wrapping.html
    mask_1 = sp.signal.firwin(mask_len, freq_range.edges.magnitude, pass_zero=False, nyq=0.5 * sig.sample_rate.magnitude)
    filterred_values = np.convolve(sig.values.magnitude, mask_1, mode="same") * pint_extension.get_units(sig.values)
    filterred = ContinuousDataEven(filterred_values, sig.sample_step, sig.first_sample)
    return filterred
    
def test_band_pass_filter():
    sample_step = uerg.sec
    np.random.seed(13)
    white_noise = ContinuousDataEven((np.random.rand(2048) - 0.5)* uerg.mamp, sample_step)
    white_noise_spec = fft(white_noise)
    freq_range = Range(np.array([0.3, 0.4]) * uerg.Hz)
    white_noise_filterred = band_pass_filter(white_noise, freq_range, 32)
    white_noise_filterred_spec = fft(white_noise_filterred)
    plot_quick(white_noise_spec, is_abs=True)
    plot_quick(white_noise_filterred_spec, is_abs=True)
    
    
    
    
# test_band_pass_filter()
    
    
#%%



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
    
def hilbert(sig, mode='fast'):
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
    """ based on hilbert transform.
    the pm demodulation at the edges is not accurate.
    TODO: map how much of the edges is a problem
    TODO: maybe it should return only the time without the edges.
    TODO: how to improve the pm demodulation at the edges?    
    TODO: maybe should add a "n_fft" parameter
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
    check_range = Range(np.array([2, 30]) * uerg.ksec)
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
    assert pint_extension.allclose(phase_sig.first_sample, expected_phase_sig.first_sample)
    assert pint_extension.allclose(phase_sig.last_sample, expected_phase_sig.last_sample, atol=min(phase_sig.sample_step, expected_phase_sig.sample_step))
    #assert pint_extension.allclose(phase_sig.sample_step, expected_phase_sig.sample_step)
    #assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)
        
def fm_demodulation(sig, mode='fast'):
    sig_phase = pm_demodulation(sig, mode)
    angular_freq = diff(sig_phase)
    freq = angular_freq.gain(1.0 / (2 * np.pi))
    return freq
    
def test_fm_demodulation():
    # copied from test_pm_demodulation
    check_range = Range(np.array([2, 30]) * uerg.ksec)
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
    #copied from test_pm_demodulation
    check_range = Range(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * uerg.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * uerg.mamp, sample_step)
    expected_sine_am = ContinuousDataEven(np.ones(sine.n_samples) * uerg.mamp, sample_step)
    sine_am = am_demodulation_hilbert(sine)
    """
    plot_quick(sine)
    plot_quick(sine_am)
    plot_quick(expected_sine_am)
    """
    assert sine_am[check_range].is_close(expected_sine_am[check_range], values_rtol=0.01)
    
def am_demodulation_convolution(sig, t_smooth):
    raise NotImplementedError
    n_samples_smooth = np.ceil(t_smooth * sig.sample_rate)
    mask_am = numpy_extension.normalize(np.ones(n_samples_smooth), ord=1)
    sig_am = np.convolve(np.abs(sig_diff), mask_am, mode="same")
    return sig_am
    
def am_demodulation_filter(sig, dt_smooth, mask_len):
    top_freq = 1.0 / dt_smooth
    band = Range([1e-12 * pint_extension.get_units(top_freq), top_freq])
    return band_pass_filter(sig, band, mask_len = mask_len)
    

def test_am_demodulation_filter():
    
    sample_step = 1.0 * uerg.sec
    time = np.arange(2 ** 15) * sample_step
    freq_1 = 0.15 * uerg.Hz
    freq_2 = 0.40 * uerg.Hz
    phase_1 = 2 * np.pi * freq_1 * time
    phase_2 = 2 * np.pi * freq_2 * time
    sine_1 = ContinuousDataEven(np.sin(phase_1) * uerg.mamp, sample_step)
    sig = ContinuousDataEven((np.sin(phase_1) + np.sin(phase_2)) * uerg.mamp, sample_step)
    
    dt = 1.0 / freq_1 * 0.5
    am = am_demodulation_filter(sig, dt, 32)
    fig, junk = plot(sig)
    plot(sine_1, fig)
    plot(am, fig)
    plot_quick(sine_1 - am)
    assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * uerg.mamp)

test_hilbert()    
test_pm_demodulation()
test_fm_demodulation()
test_am_demodulation_hilbert()
test_am_demodulation_filter()
 
