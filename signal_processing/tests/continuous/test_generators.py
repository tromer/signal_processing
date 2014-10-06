import numpy as np
import scipy as sp
from scipy import signal


from signal_processing import uerg

from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.continuous import generators


def test_generate_sine():
    sample_step = 1 * uerg.sec
    n_samples = 128
    sine_freq = 0.15 * uerg.Hz
    amplitude = 1 * uerg.mamp
    expected_sine = ContinuousDataEven(amplitude * np.sin(2 * np.pi * sine_freq * sample_step * np.arange(n_samples)), sample_step)
    sine = generators.generate_sine(sample_step, n_samples, amplitude, sine_freq)
    assert sine.is_close(expected_sine)

def test_generate_square():
    sample_step = 1 * uerg.sec
    n_samples = 128
    period = 10 * uerg.sec
    amplitude = 1 * uerg.mamp
    expected_square = ContinuousDataEven(amplitude * 0.5 * (1 + sp.signal.square(2 * np.pi * 1.0 / period * sample_step * np.arange(n_samples))), sample_step)
    square = generators.generate_square(sample_step, n_samples, amplitude, period)
    assert square.is_close(expected_square)
    #plot_quick(square)
 
def test_generate_square_freq_modulated():
    sample_step = 1 * uerg.sec
    n_samples = 2 ** 12
    sine_freq = 0.15 * uerg.Hz
    amplitude = 1 * uerg.mamp
    period = 100 * uerg.sec
    modulated = generators.generate_square_freq_modulated(sample_step, n_samples, amplitude, sine_freq, period)
    envelope = generators.generate_square(sample_step, n_samples, 1 * uerg.dimensionless, period)
    sine = generators.generate_sine(sample_step, n_samples, amplitude, sine_freq)
    assert modulated.is_close(envelope * sine)

    
test_generate_sine()
test_generate_square()
test_generate_square_freq_modulated()
#%%


