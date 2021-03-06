import numpy as np

from signal_processing.extensions import numpy_extension
from signal_processing.extensions.numpy_extension import close_power_of_2, is_power_of_2

def test_is_power_of_2():
    assert is_power_of_2(2)
    assert is_power_of_2(32)
    assert not is_power_of_2(2.1)
    assert not is_power_of_2(3)
 
def test_close_power_of_2():
    print "testing"
    inputs = np.array([1, 1.6, 2, 3, 4, 10, 20])
    expeced_outputs_smaller = np.array([1, 1, 2, 2, 4, 8, 16])
    expeced_outputs_bigger = np.array([1, 2, 2, 4, 4, 16, 32])
    expeced_outputs_closer = np.array([1, 2, 2, 4, 4, 8, 16])
    assert np.allclose(close_power_of_2(inputs, 'smaller'), expeced_outputs_smaller)
    assert np.allclose(close_power_of_2(inputs, 'bigger'), expeced_outputs_bigger)
    assert np.allclose(close_power_of_2(inputs, 'closer'), expeced_outputs_closer)
    assert close_power_of_2(30) == 16

def test_determine_n_fft():
    assert numpy_extension.determine_n_fft(14, 'accurate') == 14
    assert numpy_extension.determine_n_fft(14, 'fast') == 16
    assert numpy_extension.determine_n_fft(7, 'trim') == 4
    assert numpy_extension.determine_n_fft(5, 'zero-pad') == 8
    assert numpy_extension.determine_n_fft(10, n_fft=6) == 6
 
def test_normalize():
    vec = np.array([1, 1])
    vec_n_1 = vec / 2.0
    assert np.allclose(numpy_extension.normalize(vec, ord=1), vec_n_1)
    vec_n_2 = vec / np.sqrt(2)
    assert np.allclose(numpy_extension.normalize(vec), vec_n_2)
    

def test_deviation_from_reference():
    a = np.arange(10)
    assert np.allclose(numpy_extension.deviation_from_reference(a, np.mean(a)), np.std(a))
    


