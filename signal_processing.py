# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:31:11 2014

@author: noam
"""

import numpy as np
import scipy as sp
from scipy import  signal
import matplotlib.pyplot as plt

from pulses import Pulses
import continuous_data
from continuous_data import ContinuousDataEven

import pint_extension
from global_uerg import uerg
#from . import ureg, Q_

IS_DEBUG = False
#%%
"""
def mixer(sig, sample_rate, f_shift):
    mixer_sig = np.cos(2 * np.pi * f_shift * 1.0 / sample_rate * np.arange(len(sig)))
    return mixer_sig * sig

    
def lpf(sig, sample_rate, f_cut, mask_len=2 ** 6):
    mask = sp.signal.firwin(mask_len, cutoff=f_cut, nyq=0.5 * sample_rate)
    return np.convolve(sig, mask, mode='same')
    
def mixer_lpf(sig, sample_rate, f_min, f_max, mask_len=2 ** 6):
    assert f_max > f_min
    df = f_max - f_min
    return lpf(mixer(sig, sample_rate, f_min), sample_rate, f_cut=df, mask_len=mask_len)
    
def pm_demodulation(sig, sample_rate):
     # baed on hilbert
    analytic_sig = sp.signal.hilbert(sig)
    phase_wrapped = np.angle(analytic_sig)
    phase = np.unwrap(phase_wrapped)
    return phase
    
def fm_demodulation(sig, sample_rate):
    phase = pm_demodulation(sig, sample_rate)
    angular_freq = np.diff(phase) * sample_rate
    freq = 1.0 / (2 * np.pi) * angular_freq
    return freq
    

def test_pm_and_fm_demodulation():
    sample_rate = 1.0
    sample_step = 1.0 / sample_rate
    time = np.arange(2 ** 10) * sample_step
    freq = 0.15
    sine = np.sin(2 * np.pi * freq * time)
    pm_demo_sine = pm_demodulation(sine, sample_rate)
    plt.figure()
    plt.plot(time, pm_demo_sine)
    plt.figure()
    plt.plot(time[1:], fm_demodulation(sine, sample_rate))
    

#%%
def test_mixer_lpf():
    plt.close("all")
    
    sample_rate = 1 #Hz
    sample_step = 1.0 / sample_rate
    N = 2 ** 10
    time = np.arange(N) * sample_step
    freq = 0.15 # Hz
    freq_shift = 0.10 # Hz
    
    sine = np.sin(2 * np.pi * freq * time)
    mixed = mixer(sine, sample_rate, freq_shift)
    filterred = lpf(mixed, sample_rate, freq)
    
    freqs = np.fft.fftfreq(N, 1.0 / sample_rate)
    
    plt.figure()
    plt.xlabel("time")
    plt.plot(time, sine, label="sine")
    plt.plot(time, mixed, label="mixed")
    plt.plot(time, filterred, label="filterred")
    plt.legend(loc="best")
    
    plt.figure()
    plt.xlabel("freq")
    plt.plot(freqs, np.abs(np.fft.fft(sine, N)), label="sine")
    plt.plot(freqs, np.abs(np.fft.fft(mixed, N)), label="mixed")
    plt.plot(freqs, np.abs(np.fft.fft(filterred)), label="filterred")
    plt.legend(loc="best")
"""
#%%
def fast_convolve(sig, mask, mode):
    """ type of input determine convolution algorithm """
    raise NotImplementedError
    if case_regular:
        return np.convolve(sig, mask, mode)
    elif case_fft:
        return sp.signal.fftconvolve(sig, mask, mode)
        
#%%
def threshold_crosses(sig, threshold, is_above=True):
    """
    returns the location in indexes of crossing up, crossing down
    """
    above = sig.values > threshold
    if not is_above:
        above = np.logical_not(above)
    # the beginning and end count as non pulse
    crossings = np.logical_xor(np.concatenate([above, [False,]]), np.concatenate([[False], above]))
    crossings_indexes = np.where(crossings)[0]
    crossings_times = crossings_indexes * sig.sample_step + sig.first_sample
    starts = crossings_times[::2]
    ends = crossings_times[1::2]
    return Pulses(starts, ends)
#%%
def test_threshold_crosses():
    sig = ContinuousDataEven(np.array([3, 3, 3, 0, 0, 0, 3, 3, 0]) * uerg.mamp, uerg.sec)
    threshold = 2 * uerg.mamp
    starts_expected = np.array([0, 6])
    ends_expected = np.array([3, 8])
    pulses_expected = Pulses(starts_expected, ends_expected)
    pulses = threshold_crosses(sig, threshold)
    assert pulses.is_close(pulses_expected)
    

test_threshold_crosses()

#%%
def cluster1d(vec, resolution, threshold):
    raise NotImplementedError
    bins_num = np.ceil(1.0 * vec.ptp() / resolution)
    hist, edges = pint_extension.histogram(vec, bins_num, density=True)
    clusters = threshold_crosses(vec, 1, threshold)
    return clusters