import warnings
import numpy as np
import scipy as sp
from scipy import signal

from continuous_data_even_obj import ContinuousDataEven

from signal_processing import U_

warnings.warn('generators module is deprecated')

"""
some constructors of interesting signals

"""

"""
desing issues
--------------
1. use modulate to create square sine modulated
2. many functions here accept amplitude parameter. maybe I want it to default to 1 * dimensionless. maybe I even want to take care of the amplitude out of the generator



refactor
-----------
this entire module should be squized to ContinuousDataEven.generate function
"""

def generate_const(sample_step, n_samples, value):
    raise NotImplementedError

def generate_sine(sample_step, n_samples, amplitude, sine_freq, phase_at_0=0, first_sample=0):
    """
    returns
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


def white_noise(sample_step, n_samples, amplitude, mean=0):
    warnings.warn("not tested")
    random_values = (np.random.rand(n_samples) - 0.5 + mean) * amplitude
    white_noise = ContinuousDataEven(random_values, sample_step)
    return white_noise

def generate_square(sample_step, n_samples, amplitude, period, duty=0.5, phase_at_0=0, first_sample=0):
    """
    returns
    a ContinuousDataEven which is suqare wave with min at zero and max at amplitude

    TODO: maybe add a parameter of base level.
    """
    if np.abs(phase_at_0) > 2 * np.pi:
        warnings.warn("you are using phase_at_0 not from [-2 pi, 2 pi], weird")
    if sample_step > min(duty * period, (1-duty) * period):
        warnings.warn("the sample step is larger then 'up time' or 'down time', you can miss some wave-fronts")
    t = np.arange(n_samples) * sample_step + first_sample
    phase = 2 * np.pi * 1.0 / period * t + phase_at_0
    # the sp,signal.square returns a shquare with value 1 and -1
    # we return between 1 and 0
    square_values = 0.5 * (1 + sp.signal.square(phase, duty))
    square = ContinuousDataEven(amplitude * square_values, sample_step, first_sample)
    return square


def square_freq_modulated(sample_step, n_samples, amplitude, sine_freq, period, duty=0.5, sine_phase_at_0=0, square_phase_at_t_0=0, first_sample=0):
    """
    returns
    ContinuousDataEven which is a square wave modulated by sine. it's coherentic,
    means that all the "pulses" are taken from the same sine unstopped
    """
    envelope = generate_square(sample_step, n_samples, 1 * U_.dimensionless, period, duty, square_phase_at_t_0, first_sample)
    sine = generate_sine(sample_step, n_samples, amplitude, sine_freq, sine_phase_at_0, first_sample)
    modulated = envelope * sine
    return modulated

