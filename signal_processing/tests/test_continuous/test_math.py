import numpy as np
import scipy as sp
from scipy import signal

from signal_processing import uerg

from signal_processing.extensions import pint_extension

from signal_processing.continuous.continuous_data_obj import ContinuousData
from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.segment import Segment



from signal_processing.continuous import math


def test_diff():
    #copied from other test
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    expected_diffs = np.ones(10) * uerg.amp / uerg.sec
    expected_sig_diff = ContinuousDataEven(expected_diffs, sample_step)
    sig_diff = math.diff(sig)
    assert sig_diff.is_close(expected_sig_diff)

def visual_test_correlate():
    v = np.concatenate([np.arange(10), np.arange(10)[::-1]])
    sig_stable = ContinuousDataEven(v * uerg.mamp, uerg.sec, 10 * uerg.sec,"stable")
    sig_sliding = ContinuousDataEven(v * uerg.mamp, uerg.sec, 20 * uerg.sec, "sliding")
    sig_c = math.correlate(sig_stable, sig_sliding, mode='full')
    sig_c.values_description = 'correlation'
    fig = sig_c.plot()
    sig_stable.plot(fig)
    sig_sliding.plot(fig)

def test_correlate_find_new_location():
    v = np.concatenate([np.arange(10), np.arange(10)[::-1]])
    sig_stable = ContinuousDataEven(v * uerg.mamp, uerg.sec, 10 * uerg.sec)
    sig_sliding = ContinuousDataEven(v * uerg.mamp, uerg.sec)
    new_location, max_val = math.correlate_find_new_location(sig_stable, sig_sliding, 'full', is_return_max=True)
    print new_location
    expected_new_location = 10 * uerg.sec
    expected_max_val = 2 * (np.arange(10) ** 2).sum() * uerg.mamp ** 2 * uerg.sec
    assert pint_extension.allclose(new_location, expected_new_location)
    assert pint_extension.allclose(max_val, expected_max_val)
 
def test_clip():
    v = np.arange(10) * uerg.mamp
    sig = ContinuousDataEven(v, uerg.sec)
    Range = Segment([3, 6], uerg.mamp)
    clipped = math.clip(sig, Range)
    expected_clipped = ContinuousDataEven(np.clip(v, 3 * uerg.mamp, 6 * uerg.mamp), uerg.sec)
    assert clipped.is_close(expected_clipped)
    


   
""" 
fft is going to be out of math module
def test_fft():
    sig = ContinuousDataEven(np.arange(32) * uerg.amp, 1.0 * uerg.sec)
    expected_freqs = np.fft.fftshift(np.fft.fftfreq(32)) / uerg.sec
    expected_freqs_vals = np.fft.fftshift(np.fft.fft(np.arange(32))) * uerg.amp * uerg.sec
    expected_spec = ContinuousData(expected_freqs_vals, expected_freqs)
    spec = math.fft(sig)
    
    assert spec.is_close(expected_spec)
    
    #mostly a copy of the other test
    sig = ContinuousDataEven(np.arange(31) * uerg.amp, 1.0 * uerg.sec)
    expected_freqs_fast = np.fft.fftshift(np.fft.fftfreq(32)) / uerg.sec
    expected_freqs_vals_fast = np.fft.fftshift(np.fft.fft(np.arange(31), 32)) * uerg.amp * uerg.sec
    expected_spec_fast = ContinuousData(expected_freqs_vals_fast, expected_freqs_fast)
    spec_fast = math.fft(sig, mode='fast')
    
    assert spec_fast.is_close(expected_spec_fast)
"""
    

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
    sine_hilbert = math.hilbert(sine)
    assert sine_hilbert.is_close(expected_sine_hilbert)
 