import numpy as np

from signal_processing import U_

from signal_processing.extensions import pint_extension

def test_allclose():
    a = 3 * U_.meter
    b = 3 * U_.meter
    c = 3 * U_.centimeter
    d = 300 * U_.centimeter
    atol= 1 * U_.centimeter
    assert pint_extension.allclose(a, b)
    assert pint_extension.allclose(a, d)
    assert not pint_extension.allclose(a, c)
    assert pint_extension.allclose(a, b, atol=atol)
    #TODO: check that using a different unit for atol raises exception.
    #TODO: add assertions. this is a very fondemental function.



def test_get_units():
    x = 3 * U_.meter
    assert pint_extension.allclose(pint_extension.get_units(x), 1 * U_.meter)
    vec = np.arange(1, 5) * U_.meter
    assert pint_extension.allclose(pint_extension.get_units(vec), U_.meter)


def test_units_list_to_ndarray():
    l = [1 * U_.meter, 2 * U_.meter, 100 * U_.cmeter]
    assert pint_extension.allclose(pint_extension.units_list_to_ndarray(l), np.array([1, 2, 1]) * U_.meter)

def test_rescale_all():
    l = [1 * U_.meter, 2 * U_.meter, 100 * U_.cmeter]
    rescaled = pint_extension.rescale_all(l)
    expected_results = np.array([1, 2, 1]) * U_.meter
    for i in range(3):
        assert pint_extension.allclose(rescaled[i], expected_results[i])

def test_strip_units():
    l = [1 * U_.meter, 2 * U_.meter, 100 * U_.cmeter]
    mag, unit = pint_extension.strip_units(l)
    assert unit == U_.meter
    assert np.allclose(mag, np.array([1, 2, 1]))

    l = np.arange(10) * U_.m
    mag, unit = pint_extension.strip_units(l)
    assert unit == U_.meter
    assert np.allclose(mag, np.arange(10))

    l = np.arange(10)
    mag, unit = pint_extension.strip_units(l)
    assert unit == U_.dimensionless
    assert np.allclose(mag, l)


def test_array():
    l = [1 * U_.meter, 2 * U_.meter, 100 * U_.cmeter]
    expected_v = np.array([1, 2, 1]) * U_.meter
    v = pint_extension.array(l)
    assert pint_extension.allclose(v, expected_v)

    l_2 = [1 * U_.sec, 3 * U_.sec]
    expected_v_2 = np.array([1,3]) * U_.sec
    v_2 = pint_extension.array(l_2)
    assert pint_extension.allclose(v_2, expected_v_2)

# non mathematical manipulations
def test_concatenate():
    a = np.arange(3) * U_.meter
    b = np.arange(3) * U_.meter
    c = np.arange(3) * 100 * U_.cmeter
    expected_concat = np.concatenate([np.arange(3), np.arange(3), np.arange(3)]) * U_.meter
    concat = pint_extension.concatenate([a, b, c])
    assert pint_extension.allclose(concat, expected_concat)

# some functions that are connected to plotting and presentations
def test_get_dimensionality_str():
    u_1 = U_.m
    expected_str_1 = str((U_.m).dimensionality)
    str_1 = pint_extension.get_dimensionality_str(u_1)
    assert str_1 == expected_str_1

    u_2 = U_.Hz
    expected_str_2 = "[frequency]"
    str_2 = pint_extension.get_dimensionality_str(u_2)
    assert str_2 == expected_str_2


def test_get_units_beautiful_str():
    u_1 = U_.dimensionless
    expected_str_1 = "[AU]"
    str_1 = pint_extension.get_units_beautiful_str(u_1)
    assert str_1 == expected_str_1

    u_2 = U_.m
    expected_str_2 = "[" + str(u_2.units) + "]"
    str_2 = pint_extension.get_units_beautiful_str(u_2)
    assert str_2 == expected_str_2


def test_prepare_data_and_labels_for_plot():
    x = np.arange(10) * U_.s
    y = np.arange(10) * U_.m
    x_str = "look: x"
    curv_str = "look: curv"

    x_bare, y_bare, x_label, curv_label = \
            pint_extension.prepare_data_and_labels_for_plot(\
            x, y, x_str, curv_str)

    expected_x_bare, expected_y_bare = np.arange(10), np.arange(10)
    expected_x_label = x_str + " " + pint_extension.get_units_beautiful_str(U_.s)
    expected_curv_label = curv_str + " " + pint_extension.get_units_beautiful_str(U_.m)

    assert np.allclose(x_bare, expected_x_bare)
    assert np.allclose(y_bare, expected_y_bare)
    print x_label, expected_x_label
    assert x_label == expected_x_label
    assert curv_label == expected_curv_label


def test_histogram():
    a = (np.arange(10) + 0.5) * U_.meter
    range_ = np.array([0, 10]) * U_.meter
    expected_hist = np.ones(10)
    expected_edges = np.arange(11) * U_.meter
    hist, edges = pint_extension.histogram(a, bins=10, range_=range_)
    assert np.allclose(hist, expected_hist)
    assert pint_extension.allclose(edges, expected_edges)



def test_median():
    v = np.arange(10) * U_.m
    med = pint_extension.median(v)
    expected_median = 4.5 * U_.m
    assert pint_extension.allclose(med, expected_median)


def test_fft():
    v = np.arange(32) * U_.mamp
    spec = pint_extension.fft(v)
    expected_spec = U_.mamp * np.fft.fftshift(np.fft.fft(v))
    assert pint_extension.allclose(spec, expected_spec)

    v = np.arange(20) * U_.mamp
    spec = pint_extension.fft(v)
    expected_spec = U_.mamp * np.fft.fftshift(np.fft.fft(v))
    assert pint_extension.allclose(spec, expected_spec)

    spec = pint_extension.fft(v, mode='zero-pad')
    expected_spec = U_.mamp * np.fft.fftshift(np.fft.fft(v, 32))
    assert pint_extension.allclose(spec, expected_spec)

    spec = pint_extension.fft(v, mode='trim')
    expected_spec = U_.mamp * np.fft.fftshift(np.fft.fft(v, 16))
    assert pint_extension.allclose(spec, expected_spec)


