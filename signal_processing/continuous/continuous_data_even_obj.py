"""
.. module:: continuous_data_even_obj
"""
from operator import xor
import warnings

import numpy as np
import scipy as sp
from scipy import signal


from signal_processing.extensions import pint_extension
from signal_processing.extensions import numpy_extension

from signal_processing.segment import Segment

from continuous_data_obj import ContinuousData


from signal_processing.segments.segments_obj import Segments

from signal_processing import U_
from signal_processing.continuous import modulate, demodulate


class ContinuousDataEven(ContinuousData):
    """
    read the ContinuousData documentation.
    the domain samples are evenly spaced

    """
    def __init__(self, values, sample_step, domain_start=0,
                 values_description=None, domain_description=None):
        self._values = values
        self._sample_step = sample_step
        if not domain_start:
            self._domain_start = 0 * sample_step
        else:
            self._domain_start = domain_start

        # copied from ContinuousData
        self._domain_description = domain_description
        self._values_description = values_description

    @classmethod
    def generate(cls, waveform, sample_step, n_samples,
                 amplitude=U_.dimensionless, domain_start=0,
                 phase_at_0=0, **kwargs):
        """
        generate signal of certain type

        parameters:
        ---------------
        waveform : str
            'white_noise'
            'const'

            'sine'
            'square'

        sample_step : U_.Quantity

        n_samples : int

        amplitude : U_.Quantity
            all the values would be between -amplitude to +amplitude

        domain_start : U_.Quantity

        phase_at_0 : float
            between -pi and pi, or between 0 and 2 * pi

        kwargs
        ------------
        freq : U_.Quantity

        period : U_.Quantity
            instead of freq

        duty : float
            for square

        mean : U_.Quantity

        TODO: Table of default values for each waveform
        -----------------------------------------------

        returns
        ---------
        sig : ContinuousDataEven

        design issue
        ---------------------
        maybe passing a string to determine the type of waveform is not optimal
        other options is passing a function like np.sin, sp.signal.square,
        utils.white_noise.
        it's also optional to create a flag is_periodic
        maybe it should be possible to pass every function that is defined on
        the segment [0, 2 * pi] as a  waveform for periodic function
        """

        default_mean_value = {'white_noise' : 0, 'sine' : 0, 'square' : amplitude}

        # case 1 : non periodic
        if waveform == 'const':
            raise NotImplementedError

        elif waveform == 'white_noise':
            warnings.warn('not tested')
            vals = 2 * np.random.rand(n_samples) - 1

        # case 2 : periodic
        if not xor('freq' in kwargs, 'period' in kwargs):
            raise ValueError("given both period and freq parameters")
        if 'freq' in kwargs:
            freq = kwargs['freq']
        elif 'period' in kwargs:
            freq = 1.0 / kwargs['period']

        if np.abs(phase_at_0) > 2 * np.pi:
            # encapsulate
            warnings.warn("you are using phase_at_0 not from [-2 pi, 2 pi], weird")

        t = np.arange(n_samples) * sample_step + domain_start
        phase = 2 * np.pi * freq * t + phase_at_0

        if waveform == 'sine':
            # encapsulate
            if freq > 0.5 * 1.0 / sample_step:
                raise("trying to generate undersampled sine signal, abbort! consider the nyquist!")
            vals = np.sin(phase)

        elif waveform == 'square':
            duty = kwargs.get('duty', 0.5)
            vals =  sp.signal.square(phase, duty)

        vals = vals * amplitude
        mean = kwargs.get('mean', default_mean_value[waveform])
        vals = vals + mean
        sig = cls(vals, sample_step, domain_start)
        return sig


    @property
    def sample_step(self):
        return self._sample_step

    @property
    def sample_rate(self):
        return 1.0 / self.sample_step

    @property
    def domain_start(self):
        return self._domain_start


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
        return np.arange(len(self.values)) * self.sample_step + self.domain_start

    def new_values(self, new_vals, assert_same_n_samples=True, new_domain_start=None):
        """
        parameters:
        ---------------
        new_vals : U_.Quantity
            vectors of values of with the same amount of values as the samples

        returns:
        ---------
        new_sig : ContinuousDataEven

        """
        if assert_same_n_samples and len(new_vals) != self.n_samples:
            raise ValueError("the number of new values must be like the number of values of the old signal")
        if new_domain_start == None:
            domain_start = self.domain_start
        else:
            domain_start = new_domain_start

        new_sig = ContinuousDataEven(new_vals, self.sample_step, domain_start)
        return new_sig


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


        if type(key) in [Segment, ]:
            domain_range = key
            bottom_index = np.ceil(1.0 * domain_range.start / self.sample_step)
            top_index = np.floor(domain_range.end / self.sample_step)
            return ContinuousDataEven(self.values[bottom_index:top_index + 1], self.sample_step, domain_start=bottom_index * self.sample_step)

        elif type(key) in [Segments,]:
            return [self[domain_range] for domain_range in key]


    def move_domain_start(self, new_domain_start=0):
        """

        """
        warnings.warn("not tested")
        return ContinuousDataEven(self.values, self.sample_step, new_domain_start)

    def shift(self, delta):
        """
        shifts the domain
        """
        raise NotImplementedError
        new_domain_start = self.domain_start + delta
        return self.move_domain_start(new_domain_start)


    def is_same_domain_samples(self, other):
        return self.n_samples == other.n_samples and \
        pint_extension.allclose(self.domain_start, other.domain_start) and \
        pint_extension.allclose(self.sample_step, other.sample_step)

    def _extract_values_from_other_for_continuous_data_arithmetic(self, other):
        """
        core method to help arithmency between methods
        TODO: add more test not tested enough. some bugs
        """

        if type(other) in [float, int]:
            values = other

        elif type(other) == U_.Quantity:
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
        return ContinuousDataEven(self.values + values, self.sample_step, self.domain_start)

    def __radd__(self, other):
        raise NotImplementedError
        return self + other

    def __sub__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values - values, self.sample_step, self.domain_start)


    def __mul__(self, other):
        # TODO: add test for operation with num
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values * values, self.sample_step, self.domain_start)

    def __rmul__(self, other):
        raise NotImplementedError
        return self * other

    def __div__(self, other):
        values = self._extract_values_from_other_for_continuous_data_arithmetic(other)
        return ContinuousDataEven(self.values / values, self.sample_step, self.domain_start)

    def abs(self):
        return ContinuousDataEven(np.abs(self.values), self.sample_step, self.domain_start, "abs(" + self.values_description + ")")


    def gain(self, factor):
        """
        see doc of base class
        """
        return ContinuousDataEven(self.values * factor, self.sample_step, self.domain_start)

    def down_sample(self, down_factor):
        assert down_factor > 0
        if int(down_factor) != down_factor:
            raise NotImplementedError
        # maybe there should be another interface, with "new sample rate"
        return ContinuousDataEven(self.values[::down_factor], down_factor * self.sample_step, self.domain_start)

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
        trimmed = ContinuousDataEven(self.values[:new_n], self.sample_step, self.domain_start)
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
        freq_step : U_.Quantity
            the correct frequency step of the spectrum

        first_freq : U_.Quantity
            the first frequency. it's determined here to be (-1) * nyquist rate
            thus the spectrum is around 0 (as is should!)

        spectrum_amplitude : U_.Quantity
            the factor that multiplies the mathematical spectrum.
            it's the multiplication of the units of the values of the signal, and of the sample step itself
            (remember the definition of fft: F(freq) = integral(sig * exp(- 2 * pi * j * freq * t) * dt))
        """
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
        n_fft = numpy_extension.determine_n_fft(self.n_samples, mode, n_fft)
        freq_step, first_freq, spectrum_sample_step_factor = \
                self._spectrum_parameters(n_fft)
        spectrum_values = pint_extension.fft(self.values, n_fft = n_fft) * spectrum_sample_step_factor
        spectrum = ContinuousDataEven(spectrum_values, freq_step, first_freq, self.values_description + " specturm")
        return spectrum

    def modulate(self, kind, **kwargs):
        """
        create a modulated signal. self is considerred as the data for the\
            signal. This applies also to amplitude modulation (am).
        in amplitude modulation self is the envelope (and therfore have to be\
            non-negative)

        parameters:
        -------------
        kind : str
            'am'
            'pm'
            'fm'

        kwargs
        -----------
        amp_carrier

        f_carrier

        phase_0_carrier

        returns:
        -------------
        sig_modulated
        """
        modulator = getattr(modulate, kind)
        modulated_sig = modulator(self, **kwargs)
        return modulated_sig

    def demodulate(self, kind, mode='accurate'):
        """
        kind : str
            link to ContinuousDataEven.modulate

        mode : str
            link to numpy_extension. the preparator of fourier??

        returns
        ----------
        sig_demodulated
        """
        demodulator = getattr(demodulate, kind)
        demodulated_sig = demodulator(self, mode)
        return demodulated_sig


    def get_chunks(self, domain_duration, is_power_of_2_samples=True, is_overlap=False, mode_last_chunk='throw'):
        """
        get small chunks of the signal

        parameters
        -----------------
        domain_duration : U_.Quantity
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

        refactor
        -----------
        this is a huge function, I have to cut it

        """
        n_samples_chunk = np.ceil(domain_duration / self.sample_step)
        if is_power_of_2_samples:
            n_samples_chunk = numpy_extension.close_power_of_2(n_samples_chunk, 'bigger')

        n_samples_tot = self.n_samples

        chunks_odd = []
        n_of_chunks_odd = np.floor(n_samples_tot / n_samples_chunk)
        chunks_odd_data = self.values[:n_samples_chunk * n_of_chunks_odd].reshape((n_of_chunks_odd, n_samples_chunk)).transpose()
        chunk_odd_first_samples = self.domain_start + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_odd)
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
            chunk_even_first_samples = self.domain_start + self.sample_step * 0.5 * n_samples_chunk + self.sample_step * n_samples_chunk * np.arange(n_of_chunks_even)
            assert chunks_even_data.shape[1] == len(chunk_even_first_samples)

            for i in xrange(len(chunk_even_first_samples)):
                chunk = ContinuousDataEven(chunks_even_data[:,i], self.sample_step, chunk_even_first_samples[i])
                chunks_even.append(chunk)

            chunks = chunks_odd + chunks_even

        if mode_last_chunk != 'throw':
            raise NotImplementedError

        return chunks




