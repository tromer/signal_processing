# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 18:31:11 2014

@author: noam
"""

import numpy as np
import scipy as sp
from scipy import  signal
import matplotlib.pyplot as plt

IS_DEBUG = False
#%%
def mixer(sig, sample_rate, f_shift):
    mixer_sig = np.cos(2 * np.pi * f_shift * 1.0 / sample_rate * np.arange(len(sig)))
    return mixer_sig * sig

    
def lpf(sig, sample_rate, f_cut, mask_len=2 ** 6):
    mask = sp.signal.firwin(mask_len, cutoff=f_cut, nyq=0.5 * sample_rate)
    return np.convolve(sig, mask, mode='same')
    
def mixer_lpf(sig, sample_rate, f_min, f_max):
    assert f_max > f_min
    df = f_max - f_min
    return lpf(mixer(sig, sample_rate, f_min), sample_rate, f_cut=df)
    
def pm_demodulation(sig, sample_rate):
    """ baed on hilbert """
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
