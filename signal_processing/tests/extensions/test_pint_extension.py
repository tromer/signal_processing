def test_allclose():
    a = 3 * uerg.meter
    b = 3 * uerg.meter
    c = 3 * uerg.centimeter
    d = 300 * uerg.centimeter
    atol= 1 * uerg.centimeter
    assert allclose(a, b)
    assert allclose(a, d)
    assert not allclose(a, c)
    assert allclose(a, b, atol=atol)
    #TODO: check that using a different unit for atol raises exception.
    #TODO: add assertions. this is a very fondemental function.
    
test_allclose()


def test_get_units():
    x = 3 * uerg.meter
    assert allclose(get_units(x), 1 * uerg.meter)
    vec = np.arange(1, 5) * uerg.meter
    assert allclose(get_units(vec), uerg.meter)
    
test_get_units()

def test_units_list_to_ndarray():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    assert allclose(units_list_to_ndarray(l), np.array([1, 2, 1]) * uerg.meter)
    
test_units_list_to_ndarray()
    

def test_histogram():
    a = (np.arange(10) + 0.5) * uerg.meter
    range_ = np.array([0, 10]) * uerg.meter
    expected_hist = np.ones(10)
    expected_edges = np.arange(11) * uerg.meter
    hist, edges = histogram(a, bins=10, range_=range_)
    assert np.allclose(hist, expected_hist)
    assert allclose(edges, expected_edges)
    
test_histogram()

def test_rescale_all():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    rescaled = rescale_all(l)
    expected_results = np.array([1, 2, 1]) * uerg.meter
    for i in range(3):
        assert allclose(rescaled[i], expected_results[i])
 
def test_strip_units():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    mag, unit = strip_units(l)
    assert unit == uerg.meter
    assert np.allclose(mag, np.array([1, 2, 1]))

def test_concatenate():
    a = np.arange(3) * uerg.meter
    b = np.arange(3) * uerg.meter
    c = np.arange(3) * 100 * uerg.cmeter
    expected_concat = np.concatenate([np.arange(3), np.arange(3), np.arange(3)]) * uerg.meter
    concat = concatenate([a, b, c])
    assert allclose(concat, expected_concat)


def test_array():
    l = [1 * uerg.meter, 2 * uerg.meter, 100 * uerg.cmeter]
    expected_v = np.array([1, 2, 1]) * uerg.meter
    v = array(l)
    assert allclose(v, expected_v)

    l_2 = [1 * uerg.sec, 3 * uerg.sec]
    expected_v_2 = np.array([1,3]) * uerg.sec
    v_2 = array(l_2)
    assert allclose(v_2, expected_v_2)

def test_median():
    v = np.arange(10) * uerg.m
    med = median(v)
    expected_median = 4.5 * uerg.m
    assert allclose(med, expected_median)

test_rescale_all()
test_strip_units()
test_concatenate()
test_array()

test_median()


