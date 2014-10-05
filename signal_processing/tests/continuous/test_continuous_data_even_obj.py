def test_ContinuousDataEven():
    values = np.arange(10) * uerg.amp
    sample_step = 1.0 * uerg.sec
    sig = ContinuousDataEven(values, sample_step)
    assert pint_extension.allclose(sig.sample_step, sample_step)
    assert pint_extension.allclose(sig.sample_rate, 1.0 / sample_step)
    assert pint_extension.allclose(sig.values, values)
    assert pint_extension.allclose(sig.total_domain_width, 10 * uerg.sec)
    assert pint_extension.allclose(sig.domain_samples, np.arange(10) * sample_step)
    assert sig.is_close(ContinuousData(values, np.arange(10) * sample_step))
    assert pint_extension.allclose(sig.first_sample, 0 * sample_step)
    
    # testing a __getitem__ (slicing) is mostly copied from the tester of ContinuousData
    t_range = Segment(np.array([2.5, 6.5]) * uerg.sec)
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

def test__extract_values_from_other_for_continuous_data_arithmetic():
    # copied from test___add__
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    expected_values = sig.values
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(sig)
    assert pint_extension.allclose(values, expected_values)
    
    num = 2 * uerg.mamp
    expected_values = num
    values = sig._extract_values_from_other_for_continuous_data_arithmetic(num)
    assert pint_extension.allclose(values, expected_values)

def test___add__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    assert (sig + sig).is_close(sig.gain(2))
    num = 2 * uerg.mamp
    add_1 = sig + num
    expected_add_1 = ContinuousDataEven((2 + np.arange(10)) * uerg.mamp, uerg.sec)
    assert add_1.is_close(expected_add_1)
    
def test___sub__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    sig_2 = ContinuousDataEven(np.ones(10) * uerg.mamp, uerg.sec)
    dif = ContinuousDataEven(np.arange(-1,9) * uerg.mamp, uerg.sec)
    assert (sig - sig_2).is_close(dif)
    
def test___mul__():
    sig = ContinuousDataEven(np.arange(10) * uerg.mamp, uerg.sec)
    expected_sig_pow_2 = ContinuousDataEven(np.arange(10) ** 2 * uerg.mamp ** 2, uerg.sec)
    sig_pow_2 = sig * sig
    assert sig_pow_2.is_close(expected_sig_pow_2)
    
def test___div__():
        sig = ContinuousDataEven(np.arange(1, 10) * uerg.mamp, uerg.sec)
        assert (sig / sig).is_close(ContinuousDataEven(1 * uerg.dimensionless * np.ones(9), uerg.sec))
        assert (sig / 2.0).is_close(ContinuousDataEven(0.5 * np.arange(1, 10) * uerg.mamp, uerg.sec))
    
def test_abs():
    sig = ContinuousDataEven((-1) * np.ones(10) * uerg.mamp, uerg.sec)
    expected_sig_abs = ContinuousDataEven(np.ones(10) * uerg.mamp, uerg.sec)
    sig_abs = sig.abs()
    assert sig_abs.is_close(expected_sig_abs)
    
def test_is_power_of_2_samples():
    sig = ContinuousDataEven(np.ones(16) * uerg.mamp, uerg.sec)
    assert sig.is_power_of_2_samples()

    sig = ContinuousDataEven(np.ones(13) * uerg.mamp, uerg.sec)
    assert not sig.is_power_of_2_samples()
    
def test_trim_to_power_of_2_XXX():
    sig = ContinuousDataEven(uerg.mamp * np.arange(12), 1 * uerg.sec)
    expected_sig_trim = ContinuousDataEven(uerg.mamp * np.arange(8), 1 * uerg.sec)
    sig_trim = sig.trim_to_power_of_2_XXX()
    assert sig_trim.is_close(expected_sig_trim)


def test_get_chunks():
    N = 32
    sig = ContinuousDataEven(np.arange(N) * uerg.mamp, uerg.sec)
    chunk_duration = 3 * uerg.sec
    chunked_odd = sig.get_chunks(chunk_duration, is_overlap=False)
    chunked = sig.get_chunks(chunk_duration, is_overlap=True)
    expected_chunked_odd = [ContinuousDataEven(np.arange(4 * i, 4 * (i + 1)) * uerg.mamp, uerg.sec, uerg.sec * 4 * i) for i in range(8)]
    for i in xrange(len(chunked_odd)):
        assert chunked_odd[i].is_close(expected_chunked_odd[i])
        
    expected_chunked_even = [ContinuousDataEven(np.arange(2 + 4 * i, 2 + 4 * (i + 1)) * uerg.mamp, uerg.sec, uerg.sec * ( 4 * i + 2)) for i in range(7)]
    expected_chunked = expected_chunked_odd + expected_chunked_even
    
    for i in xrange(len(chunked)):
        assert chunked[i].is_close(expected_chunked[i])

    
    

test_ContinuousDataEven()
test_down_sample()
test_gain()
test_is_same_domain_samples()
test__extract_values_from_other_for_continuous_data_arithmetic()
test___add__()
test___sub__()
test___mul__()
test___div__()
test_abs()
test_is_power_of_2_samples()
test_trim_to_power_of_2_XXX()
test_get_chunks()
#