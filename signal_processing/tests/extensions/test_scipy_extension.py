def test_smart_convolve():
    v = np.arange(2 ** 10)
    m = np.arange(2 ** 5)
    assert np.allclose(np.convolve(v, m), smart_convolve(v,m))
    assert np.allclose(sp.signal.fftconvolve(v, m), smart_convolve(v,m))
    
test_smart_convolve()
