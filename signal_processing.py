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
    
def mixer_lpf(sig, sample_rate, f_min, f_max, mask_len=2 ** 6):
    assert f_max > f_min
    df = f_max - f_min
    return lpf(mixer(sig, sample_rate, f_min), sample_rate, f_cut=df, mask_len=mask_len)
    
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

#%%
def fast_convolve(sig, mask, mode):
    """ type of input determine convolution algorithm """
    if case_regular:
        return np.convolve(sig, mask, mode)
    elif case_fft:
        return sp.signal.fftconvolve(sig, mask, mode)
        
#%%
def threshold_crosses(sig, sample_rate, threshold, is_above=True):
    """
    returns the location in indexes of crossing up, crossing down
    """
    above = sig > threshold
    if not is_above:
        above = np.logical_not(above)
    # the beginning and end count as non pulse
    crossings = np.logical_xor(np.concatenate([above, [False,]]), np.concatenate([[False], above]))
    crossings_indexes = np.where(crossings)[0]
    crossings_times = 1.0 * crossings_indexes / sample_rate
    crossings_up = crossings_times[::2]
    crossings_down = crossings_times[1::2]
    return crossings_up, crossings_down
#%%
def test_threshold_crosses():
    sig = np.array([3, 3, 3, 0, 0, 0, 3, 3, 0])
    sample_rate = 1.0
    threshold = 2
    crossings_up_expected = np.array([0, 6])
    crossings_down_expected = np.array([3, 8])
    crossings_up, crossings_down = threshold_crosses(sig, sample_rate, threshold)
    assert np.allclose(crossings_up, crossings_up_expected)
    assert np.allclose(crossings_down, crossings_down_expected)

test_threshold_crosses()

#%%
def adjoin_close_pulses(starts, ends, max_distance):
    end_to_start_gaps = starts[1:]  - ends[:-1]
    gaps_justify_separate_pulses = end_to_start_gaps > max_distance
    true_starts_mask = np.concatenate([[True,], gaps_justify_separate_pulses])
    true_ends_mask = np.concatenate([gaps_justify_separate_pulses, [True,]])
    return starts[true_starts_mask], ends[true_ends_mask]
    
    """
    another approach is: raw_signal -> threshold -> convolve with mask=np.ones(n, dtype=bool)
    then xoring with a shift to find ends and starts, then trim the edges
    """
#%%
def test_adjoin_close_pulses():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 11])
    max_distance = 2
    adjoined_starts_expected = np.array([0, 10])
    adjoined_ends_expected = np.array([5, 11])
    adjoined_starts, adjoined_ends = adjoin_close_pulses(starts, ends, max_distance)
    assert np.allclose(adjoined_starts, adjoined_starts_expected)
    assert np.allclose(adjoined_ends, adjoined_ends_expected)
    
test_adjoin_close_pulses()
#%%
def filter_short_pulses(starts, ends, min_duration):
    durations = ends - starts
    long_enough_mask = durations > min_duration
    return starts[long_enough_mask], ends[long_enough_mask]
#%%    
def test_filter_short_pulses():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    min_duration = 0.75
    long_starts_expected = np.array([0, 2, 4])
    long_ends_expected = np.array([1, 3, 5])
    long_starts, long_ends = filter_short_pulses(starts, ends, min_duration)
    assert np.allclose(long_starts, long_starts_expected)
    assert np.allclose(long_ends, long_ends_expected)
    
test_filter_short_pulses()
#%%
def switch_pulses_and_gaps(starts, ends, absolute_start=None, absolute_end=None, epsilon=None):
    if not absolute_start:
        # this trick is used to enable concatanation with pint
        absolute_start = np.ones(1) * starts[0]
    if not absolute_end:
        absolute_end = np.ones(1) * ends[-1]
    
    starts_gaps = np.concatenate([absolute_start, ends])
    ends_gaps = np.concatenate([starts, absolute_end])
    if epsilon:
        starts_gaps, ends_gaps = filter_short_pulses(starts_gaps, ends_gaps, min_duration=epsilon)
    return starts_gaps, ends_gaps
    
    
    starts_gaps = ends[:-1]
    ends_gaps = starts[1:]
    if absolute_start:
        starts_gaps = np.concatenate([np.ones(1) * absolute_start, starts_gaps])
    if absolute_end:
        ends_gaps = np.concatenate([np.ones(1) * absolute_start, starts_gaps])
#%%
def test_switch_pulses_and_gaps():
    starts = np.array([0, 2, 4, 10])
    ends = np.array([1, 3, 5, 10.5])
    expected_starts_gaps = np.array([1, 3, 5])
    expected_ends_gaps = np.array([2, 4, 10])
    starts_gaps, ends_gaps = switch_pulses_and_gaps(starts, ends)
    assert np.allclose(starts_gaps, expected_starts_gaps)
    assert np.allclose(ends_gaps, expected_ends_gaps)

test_switch_pulses_and_gaps()
    