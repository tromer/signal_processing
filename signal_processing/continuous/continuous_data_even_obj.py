import warnings

import numpy as np
from continuous_data_obj import ContinuousData

from signal_processing.extensions import pint_extension
from signal_processing.extensions import numpy_extension
from signal_processing.segment import Segment
from signal_processing.segments.segments_obj import Segments

from signal_processing import uerg

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

    def _spectrum_parameters(self, n_fft):
        """
        parameters
        -------------------------
        contin : ContinuousDataEven
        
        n_fft : int
            len of fft
            
        returns
        --------------
        freq_step : uerg.Quantity
            the correct frequency step of the spectrum
            
        first_freq : uerg.Quantity
            the first frequency. it's determined here to be (-1) * nyquist rate
            thus the spectrum is around 0 (as is should!)
            
        spectrum_amplitude : uerg.Quantity
            the factor that multiplies the mathematical spectrum.
            it's the multiplication of the units of the values of the signal, and of the sample step itself
            (remember the definition of fft: F(freq) = integral(sig * exp(- 2 * pi * j * freq * t) * dt))
        """
        warnings.warn("not tested")
        freq_step = 1.0 * self.sample_rate / n_fft
        first_freq = - 0.5 * self.sample_rate
        spectrum_sample_step_factor =  self.sample_step
        
        return freq_step, first_freq, spectrum_sample_step_factor
    
            
    def fft(self, mode='accurate', n_fft=None):
        """
        fft of a ContinuousDataEven.
        takes the samples step into account.
        a *thin* wrap arround pint_extension.fft
        uses ContinuousDataEven._spectrum_parameters for determining all
        the spectrum frequency parameters
        
        parameters:
        ----------------
        
        mode : str
            if n_fft is None (default) then n_fft is determined according to a mode bevaviour
            copied from determine_fft_len
            'accurate' like n
            'trim' - smaller then n
            'zero-pad' - bigger then n
            'closer' - either trim or zero pad, depends which is closer (logarithmic scale)    
        
        n_fft : int
            number of samples for fft

        returns
        ---------
        spectrum : ContinuousDataEven    
            a ContinuousDataEven object that represents the spectrum
        the frequencies are considerred from -0.5 nyq frequency to 0.5 nyq frequency
    
        """
        n_fft = numpy_extension.determine_fft_len(self.n_samples, n_fft, mode)
        freq_step, first_freq, spectrum_sample_step_factor = \
                self._spectrum_parameters(n_fft)
        spectrum_values = pint_extension.fft(self, n_fft) * spectrum_sample_step_factor 
        spectrum = ContinuousDataEven(spectrum_values_with_units, freq_step, first_freq)
        return spectrum

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
            
        


