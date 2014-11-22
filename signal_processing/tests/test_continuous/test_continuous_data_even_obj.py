import numpy as np
import scipy as sp
from scipy import signal

from signal_processing import U_

from signal_processing.extensions import pint_extension

from signal_processing.continuous.continuous_data_obj import ContinuousData
from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven
from signal_processing.continuous import modulate, demodulate

from signal_processing.segment import Segment


def test_ContinuousDataEven():
    values = np.arange(10) * U_.amp
    sample_step = 1.0 * U_.sec
    sig = ContinuousDataEven(values, sample_step)
    assert pint_extension.allclose(sig.sample_step, sample_step)
    assert pint_extension.allclose(sig.sample_rate, 1.0 / sample_step)
    assert pint_extension.allclose(sig.values, values)
    assert pint_extension.allclose(sig.total_domain_width, 10 * U_.sec)
    assert pint_extension.allclose(sig.domain_samples, np.arange(10) * sample_step)
    assert sig.is_close(ContinuousData(values, np.arange(10) * sample_step))
    assert pint_extension.allclose(sig.domain_start, 0 * sample_step)

    # testing a __getitem__ (slicing) is mostly copied from the tester of ContinuousData
    t_range = Segment(np.array([2.5, 6.5]) * U_.sec)
    expected_slice = np.arange(3,7)
    expected_sig_middle = ContinuousDataEven(values[expected_slice], sample_step, expected_slice[0] * sample_step)
    sig_middle = sig[t_range]
    assert sig_middle.is_close(expected_sig_middle)

def test_generate():
    sample_step = 1 * U_.sec
    n_samples = 128
    sine_freq = 0.15 * U_.Hz
    amplitude = 1 * U_.mamp
    expected_sine = ContinuousDataEven(amplitude * np.sin(2 * np.pi * sine_freq * sample_step * np.arange(n_samples)), sample_step)
    sine = ContinuousDataEven.generate('sine', sample_step, n_samples, amplitude, freq=sine_freq)
    assert sine.is_close(expected_sine)

    period = 10 * U_.sec
    expected_square = ContinuousDataEven(amplitude * (1 + sp.signal.square(2 * np.pi * 1.0 / period * sample_step * np.arange(n_samples))), sample_step)

    square = ContinuousDataEven.generate('square', sample_step, n_samples, amplitude, period=period)
    assert square.is_close(expected_square)

def test_new_values():
    sig = ContinuousDataEven(np.arange(10) * U_.volt, U_.sec)
    new_vals = 2 * np.arange(10) * U_.mamp
    expected_new_sig = ContinuousDataEven(new_vals, U_.sec)
    new_sig = sig.new_values(new_vals)

    assert new_sig.is_close(expected_new_sig)

def test_down_sample():
    # copied from the test of fft
    sig = ContinuousDataEven(np.arange(32) * U_.amp, 1.0 * U_.sec)
    down_factor = 2
    expected_down = ContinuousDataEven(np.arange(0, 32, 2) * U_.amp, 2.0 * U_.sec)
    down = sig.down_sample(down_factor)
    assert down.is_close(expected_down)


def test_shift():
    sig = ContinuousDataEven(np.arange(32) * U_.amp, 1.0 * U_.sec)
    shift = 3 * U_.sec
    expected_shifted = ContinuousDataEven(np.arange(32) * U_.amp, 1.0 * U_.sec, shift)
    shifted = sig.shift(shift)
    shifted.is_close(expected_shifted)

def test_gain():
    # copied from test_ContinuousDataEven
    values = np.arange(10) * U_.amp
    sample_step = 1.0 * U_.sec
    sig = ContinuousDataEven(values, sample_step)
    factor = 2
    expected_sig_gain = ContinuousDataEven(values * factor, sample_step)
    sig_gain = sig.gain(factor)
    assert sig_gain.is_close(expected_sig_gain)

def test_is_same_domain_samples():
    step_1 = U_.sec
    step_2 = U_.sec * 2
    start_1 = 0
    start_2 = 1 * U_.sec
    vals_1 = np.arange(10) * U_.mamp
    vals_2 = 2 * np.arange(10) * U_.amp
    vals_3 = np.arange(5) * U_.amp
    assert ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_2, step_1))
    assert not ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_1, step_2))
    assert not ContinuousDataEven(vals_1,step_1, start_1).is_same_domain_samples(ContinuousDataEven(vals_1, step_1, start_2))
    assert not ContinuousDataEven(vals_1, step_1).is_same_domain_samples(ContinuousDataEven(vals_3, step_1))

def test__extract_values_from_other_for_continuous_data_arithmetic():
    # copied from test___add__
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    expected_values = sig.values
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(sig)
    assert pint_extension.allclose(values, expected_values)

    num = 2 * U_.mamp
    expected_values = num
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(num)
    assert pint_extension.allclose(values, expected_values)

def test___add__():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    assert (sig + sig).is_close(sig.gain(2))
    num = 2 * U_.mamp
    add_1 = sig + num
    expected_add_1 = ContinuousDataEven((2 + np.arange(10)) * U_.mamp, U_.sec)
    assert add_1.is_close(expected_add_1)

def test___sub__():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    sig_2 = ContinuousDataEven(np.ones(10) * U_.mamp, U_.sec)
    dif = ContinuousDataEven(np.arange(-1,9) * U_.mamp, U_.sec)
    assert (sig - sig_2).is_close(dif)

def test___mul__():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    expected_sig_pow_2 = ContinuousDataEven(np.arange(10) ** 2 * U_.mamp ** 2, U_.sec)
    sig_pow_2 = sig * sig
    assert sig_pow_2.is_close(expected_sig_pow_2)

def test___div__():
        sig = ContinuousDataEven(np.arange(1, 10) * U_.mamp, U_.sec)
        assert (sig / sig).is_close(ContinuousDataEven(1 * U_.dimensionless * np.ones(9), U_.sec))
        assert (sig / 2.0).is_close(ContinuousDataEven(0.5 * np.arange(1, 10) * U_.mamp, U_.sec))

def test_abs():
    sig = ContinuousDataEven((-1) * np.ones(10) * U_.mamp, U_.sec)
    expected_sig_abs = ContinuousDataEven(np.ones(10) * U_.mamp, U_.sec)
    sig_abs = sig.abs()
    assert sig_abs.is_close(expected_sig_abs)

def test_is_power_of_2_samples():
    sig = ContinuousDataEven(np.ones(16) * U_.mamp, U_.sec)
    assert sig.is_power_of_2_samples()

    sig = ContinuousDataEven(np.ones(13) * U_.mamp, U_.sec)
    assert not sig.is_power_of_2_samples()

def test_trim_to_power_of_2_XXX():
    sig = ContinuousDataEven(U_.mamp * np.arange(12), 1 * U_.sec)
    expected_sig_trim = ContinuousDataEven(U_.mamp * np.arange(8), 1 * U_.sec)
    sig_trim = sig.trim_to_power_of_2_XXX()
    assert sig_trim.is_close(expected_sig_trim)

def test__spectrum_parameters():
    sig = ContinuousDataEven(np.arange(32) * U_.amp, 1.0 * U_.sec)
    n_fft = 32
    expected_freq_step = U_.Hz * 1.0 / n_fft
    expected_first_freq = -0.5 * U_.Hz
    expected_spectrum_sample_step_factor = U_.sec
    freq_step, first_freq, spectrum_sample_step_factor = sig._spectrum_parameters(n_fft)

    assert pint_extension.allclose(freq_step, expected_freq_step)
    assert pint_extension.allclose(first_freq, expected_first_freq)
    assert pint_extension.allclose(spectrum_sample_step_factor, expected_spectrum_sample_step_factor)

def test_fft():
    sig = ContinuousDataEven(np.arange(32) * U_.amp, 1.0 * U_.sec)
    expected_freqs = np.fft.fftshift(np.fft.fftfreq(32)) / U_.sec
    expected_freqs_vals = np.fft.fftshift(np.fft.fft(np.arange(32))) * U_.amp * U_.sec
    expected_spec = ContinuousData(expected_freqs_vals, expected_freqs)
    spec = sig.fft()

    assert spec.is_close(expected_spec)

    #mostly a copy of the other test
    sig = ContinuousDataEven(np.arange(31) * U_.amp, 1.0 * U_.sec)
    expected_freqs_fast = np.fft.fftshift(np.fft.fftfreq(32)) / U_.sec
    expected_freqs_vals_fast = np.fft.fftshift(np.fft.fft(np.arange(31), 32)) * U_.amp * U_.sec
    expected_spec_fast = ContinuousData(expected_freqs_vals_fast, expected_freqs_fast)
    spec_fast = sig.fft(mode='fast')

    assert spec_fast.is_close(expected_spec_fast)


def test_modulate():
    square = ContinuousDataEven.generate('square', U_.sec, 2 ** 6, period = 20 * U_.sec)
    square_with_carrier = square.modulate('am', f_carrier = 0.15 * U_.Hz)
    expected_square_with_carrier = modulate.am(square, f_carrier = 0.15 * U_.Hz)

def test_demodulate():
    sig = ContinuousDataEven.generate('sine', U_.sec, 2 ** 6, freq=0.15 * U_.Hz)

    sig_am = sig.demodulate('am')
    sig_fm = sig.demodulate('fm')
    sig_pm = sig.demodulate('pm')

    expected_sig_am = demodulate.am(sig)
    expected_sig_fm = demodulate.fm(sig)
    expected_sig_pm = demodulate.pm(sig)

    assert sig_am.is_close(expected_sig_am)
    assert sig_fm.is_close(expected_sig_fm)
    assert sig_pm.is_close(expected_sig_pm)


def test_get_chunks():
    N = 32
    sig = ContinuousDataEven(np.arange(N) * U_.mamp, U_.sec)
    chunk_duration = 3 * U_.sec
    chunked_odd = sig.get_chunks(chunk_duration, is_overlap=False)
    chunked = sig.get_chunks(chunk_duration, is_overlap=True)
    expected_chunked_odd = [ContinuousDataEven(np.arange(4 * i, 4 * (i + 1)) * U_.mamp, U_.sec, U_.sec * 4 * i) for i in range(8)]
    for i in xrange(len(chunked_odd)):
        assert chunked_odd[i].is_close(expected_chunked_odd[i])

    expected_chunked_even = [ContinuousDataEven(np.arange(2 + 4 * i, 2 + 4 * (i + 1)) * U_.mamp, U_.sec, U_.sec * ( 4 * i + 2)) for i in range(7)]
    expected_chunked = expected_chunked_odd + expected_chunked_even

    for i in xrange(len(chunked)):
        assert chunked[i].is_close(expected_chunked[i])




