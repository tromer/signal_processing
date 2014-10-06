import numpy as np

from signal_processing import uerg

from signal_processing.extensions import pint_extension

def test_allclose():
    a = 3 * uerg.meter
    b = 3 * uerg.meter
    c = 3 * uerg.centimeter
    d = 300 * uerg.centimeter
    atol= 1 * uerg.centimeter
    assert pint_extension.allclose(a, b)
    assert pint_extension.allclose(a, d)
    assert not pint_extension.allclose(a, c)
    assert pint_extension.allclose(a, b, atol=atol)
    #TODO: check that using a different unit for atol raises exception.
    #TODO: add assertions. this is a very fondemental function.
    


def test_get_units():
    x = 3 * uerg.meter
    assert pint_extension.allclose(pint_extension.get_units(x), 1 * uerg.meter)
    vec = np.arange(1, 5) * uerg.meter
    assert pint_extension.allclose(pint_extension.get_units(vec), uerg.meter)
    

def test_units_list_to_ndarray():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    assert pint_extension.allclose(pint_extension.units_list_to_ndarray(l), np.array([1, 2, 1]) * uerg.meter)
    

def test_histogram():
    a = (np.arange(10) + 0.5) * uerg.meter
    range_ = np.array([0, 10]) * uerg.meter
    expected_hist = np.ones(10)
    expected_edges = np.arange(11) * uerg.meter
    hist, edges = pint_extension.histogram(a, bins=10, range_=range_)
    assert np.allclose(hist, expected_hist)
    assert pint_extension.allclose(edges, expected_edges)
    

def test_rescale_all():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    rescaled = pint_extension.rescale_all(l)
    expected_results = np.array([1, 2, 1]) * uerg.meter
    for i in range(3):
        assert pint_extension.allclose(rescaled[i], expected_results[i])
 
def test_strip_units():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    mag, unit = pint_extension.strip_units(l)
    assert unit == uerg.meter
    assert np.allclose(mag, np.array([1, 2, 1]))

def test_concatenate():
    a = np.arange(3) * uerg.meter
    b = np.arange(3) * uerg.meter
    c = np.arange(3) * 100 * uerg.cmeter
    expected_concat = np.concatenate([np.arange(3), np.arange(3), np.arange(3)]) * uerg.meter
    concat = pint_extension.concatenate([a, b, c])
    assert pint_extension.allclose(concat, expected_concat)


def test_array():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    expected_v = np.array([1, 2, 1]) * uerg.meter
    v = pint_extension.array(l)
    assert pint_extension.allclose(v, expected_v)

    l_2 = [1 * uerg.sec, 3 * uerg.sec]
    expected_v_2 = np.array([1,3]) * uerg.sec
    v_2 = pint_extension.array(l_2)
    assert pint_extension.allclose(v_2, expected_v_2)

def test_median():
    v = np.arange(10) * uerg.m
    med = pint_extension.median(v)
    expected_median = 4.5 * uerg.m
    assert pint_extension.allclose(med, expected_median)



