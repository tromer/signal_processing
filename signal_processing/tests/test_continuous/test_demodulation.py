import numpy as np

from signal_processing.extensions.plt_extension import plot_few

from signal_processing import U_


from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.segment import Segment

from signal_processing.continuous import demodulation
from signal_processing.continuous import generators

def test_pm():
    check_range = Segment(np.array([2, 30]) * U_.ksec)
    sample_step = 1.0 * U_.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * U_.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * U_.mamp, sample_step)
    expected_phase_sig = ContinuousDataEven(phase, sample_step)
    phase_sig = demodulation.pm(sine)
    assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)

    time = np.arange(2 ** 15 - 100) * sample_step
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * U_.mamp, sample_step)
    expected_phase_sig = ContinuousDataEven(phase, sample_step)
    phase_sig = demodulation.pm(sine)
    assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)
    # weird, it acctually gives phase diff of 0.5 pi from what I expect

    #fig, junk = plot_quick(expected_phase_sig)
    #plot(phase_sig, fig)
    #assert pint_extension.allclose(phase_sig.sample_step, expected_phase_sig.sample_step)
    #assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)

def test_fm():
    # copied from test_pm
    check_range = Segment(np.array([2, 30]) * U_.ksec)
    sample_step = 1.0 * U_.sec
    time = np.arange(2 ** 15) * sample_step
    freq = 0.15 * U_.Hz
    phase = 2 * np.pi * freq * time
    sine = ContinuousDataEven(np.sin(phase) * U_.mamp, sample_step)
    expected_freq_sig = ContinuousDataEven(np.ones(2 ** 15) * freq, sample_step)
    freq_sig = demodulation.fm(sine)
    assert freq_sig[check_range].is_close(expected_freq_sig[check_range], values_rtol=0.01)

def test_am():
    check_range = Segment(np.array([2, 30]) * U_.ksec)
    sample_step = 1.0 * U_.sec
    n_samples = 2 ** 15
    freq = 0.15 * U_.Hz
    amp = U_.mamp
    sine = generators.generate_sine(sample_step, n_samples, amp, sine_freq=freq)
    expected_sine_am = ContinuousDataEven(np.ones(sine.n_samples) * amp, sample_step)
    sine_am = demodulation.am(sine)
    plot_few(sine, sine_am, expected_sine_am)

    assert sine_am[check_range].is_close(expected_sine_am[check_range], values_rtol=0.01)

    """
    this test fails now, it needs is_close_l_1 to work properly
    period = 100 * U_.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_hilbert(sig)
    plot_few(sig, expected_sig_am, sig_am, sig_am - expected_sig_am)
   # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)
    """

#def XXX_test_am_demodulation_convolution():
    #check_range = Segment(np.array([2, 30]) * U_.ksec)
    #sample_step = 1.0 * U_.sec
    #n_samples = 2 ** 15
    #freq_1 = 0.15 * U_.Hz
    #freq_2 = 0.40 * U_.Hz
    #amp = U_.mamp
    #sine_1 = generators.generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    #sine_2 = generators.generate_sine(sample_step, n_samples, amp, freq_2)
    #sig = sine_1 + sine_2
    #"""
    ##copied from test_am_demodulation_filter
    #dt = 1.0 / freq_1 * 0.25
    #am = am_demodulation_convolution(sig, dt)
    #assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * U_.mamp)
    #"""
    #dt = 1.0 / freq_1 * 3
    #period = 100 * U_.sec
    #sig = generators.generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    #expected_sig_am = generators.generate_square(sample_step, n_samples, amp, period)
    #sig_am = demodulation.am_demodulation_convolution(sig, dt)
    #plot_few(sig, expected_sig_am, sig_am, sig_am - expected_sig_am)
   ## the big tolerance is due to gibs effect
    #assert sig_am[check_range].is_close(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)

#def XXX_test_am_demodulation_filter():
    #check_range = Segment(np.array([2, 30]) * U_.ksec)
    #sample_step = 1.0 * U_.sec
    #n_samples = 2 ** 15
    #freq_1 = 0.15 * U_.Hz
    #freq_2 = 0.40 * U_.Hz
    #amp = U_.mamp
    #dt = 1.0 / freq_1 * 0.5
    #"""
    #sine_1 = generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    #sine_2 = generate_sine(sample_step, n_samples, amp, freq_2)
    #sig = sine_1 + sine_2


    #am = am_demodulation_filter(sig, dt, 128)
    #assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * U_.mamp)
    #"""

    #dt = 1.0 / freq_1 * 3
    #period = 100 * U_.sec
    #sig = generators.generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    #expected_sig_am = generators.generate_square(sample_step, n_samples, amp, period)
    #sig_am = demodulation.am_demodulation_filter(sig, dt, 256)
    #plot_few(sig, expected_sig_am, sig_am, sig_am - expected_sig_am)
   ## the big tolerance is due to gibs effect
    #assert sig_am[check_range].is_close(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)

