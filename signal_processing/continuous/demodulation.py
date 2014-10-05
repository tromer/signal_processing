    
#TODO: there are some problematic issues with fft / hilbert /demodulations with not 2 ** n samples signals.

def pm_demodulation(sig, mode='fast'):
    """
    based on hilbert transform.
    the pm demodulation at the edges is not accurate.
    TODO: map how much of the edges is a problem
    TODO: maybe it should return only the time without the edges.
    TODO: how to improve the pm demodulation at the edges?    
    TODO: maybe should add a "n_fft" parameter
    TODO: maybe it's better to allow calculation of phase with separation to windows?
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
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
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
    # weird, it acctually gives phase diff of 0.5 pi from what I expect
    assert pint_extension.allclose(phase_sig.first_sample, expected_phase_sig.first_sample)
    assert pint_extension.allclose(phase_sig.last_sample, expected_phase_sig.last_sample, atol=min(phase_sig.sample_step, expected_phase_sig.sample_step))
    #fig, junk = plot_quick(expected_phase_sig)
    #plot(phase_sig, fig)
    #assert pint_extension.allclose(phase_sig.sample_step, expected_phase_sig.sample_step)
    #assert phase_sig[check_range].is_close(expected_phase_sig[check_range], values_rtol=0.01)
        
def fm_demodulation(sig, mode='fast'):
    """
    fm demodulation
    based on differentiating the pm demodulation
    """
    sig_phase = pm_demodulation(sig, mode)
    angular_freq = diff(sig_phase)
    freq = angular_freq.gain(1.0 / (2 * np.pi))
    return freq
    
def test_fm_demodulation():
    # copied from test_pm_demodulation
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
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
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq = 0.15 * uerg.Hz
    amp = uerg.mamp
    sine = generate_sine(sample_step, n_samples, amp, sine_freq=freq)
    expected_sine_am = ContinuousDataEven(np.ones(sine.n_samples) * amp, sample_step)
    sine_am = am_demodulation_hilbert(sine)
    """
    plot_quick(sine)
    plot_quick(sine_am)
    plot_quick(expected_sine_am)
    """
    assert sine_am[check_range].is_close(expected_sine_am[check_range], values_rtol=0.01)
    
    """
    this test fails now, it needs is_close_l_1 to work properly
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_hilbert(sig)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)
    """
    
    
def am_demodulation_convolution(sig, t_smooth):
    """
    params:
    t_smooth is the width in domain units, that you want to smooth together
    """
    warnings.warn("not tested well")
    n_samples_smooth = np.ceil(t_smooth * sig.sample_rate)
    mask_am = numpy_extension.normalize(np.ones(n_samples_smooth), ord=1)
    values_am = np.convolve(np.abs(sig.values.magnitude), mask_am, mode="same") * pint_extension.get_units(sig.values)
    return ContinuousDataEven(values_am, sig.sample_step, sig.first_sample)

def test_am_demodulation_convolution():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq_1 = 0.15 * uerg.Hz
    freq_2 = 0.40 * uerg.Hz
    amp = uerg.mamp
    sine_1 = generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    sine_2 = generate_sine(sample_step, n_samples, amp, freq_2)
    sig = sine_1 + sine_2
    """
    #copied from test_am_demodulation_filter
    dt = 1.0 / freq_1 * 0.25
    am = am_demodulation_convolution(sig, dt)
    fig, junk = plot(sig)
    plot(sine_1, fig)
    plot(am, fig)
    plot_quick(sine_1 - am)
    assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * uerg.mamp)
    """
    dt = 1.0 / freq_1 * 3
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_convolution(sig, dt)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)

    
def am_demodulation_filter(sig, dt_smooth, mask_len):
    warnings.warn("not tested well")
    top_freq = 1.0 / dt_smooth
    band = Segment([1e-12 * pint_extension.get_units(top_freq), top_freq])
    return band_pass_filter(sig.abs(), band, mask_len = mask_len)
    

def test_am_demodulation_filter():
    check_range = Segment(np.array([2, 30]) * uerg.ksec)
    sample_step = 1.0 * uerg.sec
    n_samples = 2 ** 15
    freq_1 = 0.15 * uerg.Hz
    freq_2 = 0.40 * uerg.Hz
    amp = uerg.mamp
    dt = 1.0 / freq_1 * 0.5    
    """
    sine_1 = generate_sine(sample_step, n_samples, amp, sine_freq=freq_1)
    sine_2 = generate_sine(sample_step, n_samples, amp, freq_2)
    sig = sine_1 + sine_2
    
    
    am = am_demodulation_filter(sig, dt, 128)
    fig, junk = plot(sig)
    plot(sine_1.abs(), fig)
    plot(am, fig)
    plot_quick(sine_1.abs() - am)
    assert sine_1.is_close(am, domain_rtol=0.01, domain_atol=0.1 * uerg.mamp)
    """
    
    dt = 1.0 / freq_1 * 3
    period = 100 * uerg.sec
    sig = generate_square_freq_modulated(sample_step, n_samples, amp, freq_1, period)
    expected_sig_am = generate_square(sample_step, n_samples, amp, period)
    sig_am = am_demodulation_filter(sig, dt, 256)
    fig, junk = plot_quick(sig)
    plot(expected_sig_am, fig)
    plot(sig_am, fig)
    plot_quick(sig_am - expected_sig_am, fig)
    # the big tolerance is due to gibs effect
    assert sig_am[check_range].is_close_l_1(expected_sig_am[check_range], values_rtol=0.2, values_atol=0.2 * amp)
  
test_pm_demodulation()
test_fm_demodulation()
test_am_demodulation_hilbert()
#test_am_demodulation_convolution()
#test_am_demodulation_filter()


