class ContinuousDataEven(ContinuousData):
    """
    read the ContinuousData documentation.
    the domain samples are evenly spaced
    """
    def __init__(self, values, sample_step, first_sample=0):
        self._values = values
        self._sample_step = sample_step
        if not first_sample:
            self._first_sample = 0 * sample_step
        else:
            self._first_sample = first_sample
        
        
    @property
    def sample_step(self):
        return self._sample_step
        
    @property
    def sample_rate(self):
        return 1.0 / self.sample_step
    
    @property
    def first_sample(self):
        return self._first_sample
    
        
    @property
    def total_domain_range(self):
        """
        
        returns:
        ------------
        domain_range : Segment
        """
        raise NotImplementedError
        
    @property
    def total_domain_width(self):
        """
        returns the total length of the signal
        
        Note:
        maybe it's possible to base the implementation on total_domain_range
        at the moment it's done differently
        
        TODO: use this method as a first check in comparison
        """
        return self.n_samples * self.sample_step
        
    """
    @property
    def last_sample(self):
        # TODO: possible to create a better implementation then in the father class
    """
        
    @property
    def domain_samples(self):
        #print "******"
        #print self.values
        """ TODO: mayebe some cashing would be helpful? """
        return np.arange(len(self.values)) * self.sample_step + self.first_sample
        
    def __getitem__(self, key):
        """
        Note: it's coppied from __getitem__ of ContinuousData
        parameters:
        -------------
        domain_range : Segment
            the range, from the domain, of which we want the slice.
            for example: which time range?
        """
        if type(key) in [int, float]:
            raise KeyError("wrong key. key for ContinuousData is Segment or Segments of the same domain")
        

        if type(key) in [Segment,]:
            domain_range = key
            bottom_index = np.ceil(1.0 * domain_range.start / self.sample_step)
            top_index = np.floor(domain_range.end / self.sample_step)
            return ContinuousDataEven(self.values[bottom_index:top_index + 1], self.sample_step, first_sample=bottom_index * self.sample_step)
            
        elif type(key) in [Segments,]:
            return [self[domain_range] for domain_range in key]
        
        
    def move_first_sample(self, new_first_sample=0):
        """
        
        """
        warnings.warn("not tested")
        return ContinuousDataEven(self.values, self.sample_step, new_first_sample)
        
        
    def is_same_domain_samples(self, other):
        return self.n_samples == other.n_samples and \
        pint_extension.allclose(self.first_sample, other.first_sample) and \
        pint_extension.allclose(self.sample_step, other.sample_step)
        
    def _extract_values_from_other_for_continuous_data_arithmetic(self, other):
        """
        core method to help arithmency between methods
        TODO: add more test not tested enough. some bugs
        """
        
        if type(other) in [float, int]:
            values = other
        
        elif type(other) == uerg.Quantity:
            if type(other.magnitude) in [np.ndarray,]:
                raise ValueError("add const value, or other ContinuousData with same domain samples")
            else:
                values = other
        else:
            # TODO: add gaurdian, that other is another ContinuousData
            if not self.is_same_domain_samples(other):
                raise ValueError("diffrent domain samples")
            else:
                values = other.values        
        
        return values
        
    def __add__(self, other):
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)    
        return ContinuousDataEven(self.values + values, self.sample_step, self.first_sample)
        
    def __radd__(self, other):
        raise NotImplementedError
        return self + other
            
    def __sub__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)    
        return ContinuousDataEven(self.values - values, self.sample_step, self.first_sample)           

            
    def __mul__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)    
        return ContinuousDataEven(self.values * values, self.sample_step, self.first_sample)

    def __rmul__(self, other):
        raise NotImplementedError
        return self * other
        
    def __div__(self, other):
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)    
        return ContinuousDataEven(self.values / values, self.sample_step, self.first_sample)            
    
    def abs(self):
        return ContinuousDataEven(np.abs(self.values), self.sample_step, self.first_sample)
        
        
    def gain(self, factor):
        """
        see doc of base class
        """
        return ContinuousDataEven(self.values * factor, self.sample_step, self.first_sample)
        
    def down_sample(self, down_factor):
        assert down_factor > 0
        if int(down_factor) != down_factor:
            raise NotImplementedError
        # maybe there should be another interface, with "new sample rate"
        return ContinuousDataEven(self.values[::down_factor], down_factor * self.sample_step, self.first_sample)
        
    def is_power_of_2_samples(self):
        """
        check for performance issues
        """
        return numpy_extension.is_power_of_2(self.n_samples)
        
    def trim_to_power_of_2_XXX(self):
        """
        trancate data to power of 2 sample points
        loss of data, very dangareous
        """
        warnings.warn("XXX trim_to_power_of_2_looses_data")
        new_n = numpy_extension.close_power_of_2(self.n_samples, mode='smaller')
        trimmed = ContinuousDataEven(self.values[:new_n], self.sample_step, self.first_sample)
        assert trimmed.n_samples == new_n
        return trimmed
        
    def get_chunks(self, domain_duration, is_power_of_2_samples=True, is_overlap=False, mode_last_chunk='throw'):
        """
        get small chunks of the signal
        
        parameters
        -----------------
        domain_duration : uerg.Quantity
            the duration / len / width of the chunk
        is_power_of_2_samples : bool
            whether to return bigger chunks, with 2 ** m samples
            for performence / efficiency issues
        is_overlap : bool
            whether to give chunks with overlap of half the duration
            in order to cope with splits
        mode_last_chunk : str
            what to do with the last chunk, which is smaller?
            'throw' - just don't give it back
            
        returns
        ----------
        a list of signals.
        if performence issues erise, it's better to return a
        generator using yield.
        
        TODO:
        --------
        there is a lot of code duplication with the odd and even
        maybe should return a sig array object
            channels object is another option, but I think that it's a bad idea
            since the chunks do not share the domain samples. (different time)
        maybe should enable passing Segment, or Segments to specify where to chunk
        cope with case the signal is too short compared to chunk asked
        
        """
        n_samples_chunk = np.ceil(domain_duration / self.sample_step)
        if is_power_of_2_samples:
            n_samples_chunk = numpy_extension.close_power_of_2(n_samples_chunk, 'bigger')

        n_samples_tot = self.n_samples   

        chunks_odd = []
        n_of_chunks_odd = np.floor(n_samples_tot / n_samples_chunk)
        chunks_odd_data = self.values[:n_samples_chunk * n_of_chunks_odd].reshape((n_of_chunks_odd, n_samples_chunk)).transpose()
        chunk_odd_first_samples = self.first_sample + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_odd)
        assert chunks_odd_data.shape[1] == len(chunk_odd_first_samples)

        for i in xrange(len(chunk_odd_first_samples)):
            chunk = ContinuousDataEven(chunks_odd_data[:,i], self.sample_step, chunk_odd_first_samples[i])
            chunks_odd.append(chunk)

        if not is_overlap:
            chunks = chunks_odd
        else:
            chunks_even = []
            n_of_chunks_even = np.floor((n_samples_tot - 0.5 * n_samples_chunk) / n_samples_chunk)
            chunks_even_data = self.values[0.5 * n_samples_chunk : 0.5 * n_samples_chunk + n_samples_chunk * n_of_chunks_even].reshape((n_of_chunks_even, n_samples_chunk)).transpose()
            chunk_even_first_samples = self.first_sample + self.sample_step * 0.5 * n_samples_chunk + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_even)
            assert chunks_even_data.shape[1] == len(chunk_even_first_samples)
                
            for i in xrange(len(chunk_even_first_samples)):
                chunk = ContinuousDataEven(chunks_even_data[:,i], self.sample_step, chunk_even_first_samples[i])
                chunks_even.append(chunk)
                
            chunks = chunks_odd + chunks_even
    
        if mode_last_chunk != 'throw':
            raise NotImplementedError
            
        return chunks
            
        

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
