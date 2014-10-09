import warnings
import numpy as np

from signal_processing.segment import Segment

from continuous_data_even_obj import ContinuousDataEven
from signal_processing.continuous.math import hilbert
from signal_processing.continuous.math import diff
from signal_processing.continuous.filters import band_pass_filter
from signal_processing import uerg
from signal_processing.extensions import pint_extension
from signal_processing.extensions import numpy_extension

# TODO: there are some problematic issues with fft / hilbert /demodulations with not 2 ** n samples signals.


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


def fm_demodulation(sig, mode='fast'):
    """
    fm demodulation
    based on differentiating the pm demodulation
    """
    sig_phase = pm_demodulation(sig, mode)
    angular_freq = diff(sig_phase)
    freq = angular_freq.gain(1.0 / (2 * np.pi))
    return freq
    
   
def am_hilbert(sig, mode='fast'):
    #worning copied from pm_demodulation
    if sig.n_samples < 2 ** 10:
        warnings.warn("this pm-modulation technique doesn't work well on short signals, the mistakes on the edges are big")
    analytic_sig = hilbert(sig, mode)
    envelope = np.abs(analytic_sig.values.magnitude) * pint_extension.get_units(analytic_sig.values)
    sig_am = ContinuousDataEven(envelope, analytic_sig.sample_step, analytic_sig.first_sample)
    return sig_am
    
   
def am_demodulation_convolution(sig, t_smooth):
    """
    params:
    t_smooth is the width in domain units, that you want to smooth together
    """
    warnings.warn("not tested")
    n_samples_smooth = np.ceil(t_smooth * sig.sample_rate)
    mask_am = numpy_extension.normalize(np.ones(n_samples_smooth), ord=1)
    values_am = np.convolve(np.abs(sig.values.magnitude), mask_am, mode="same") * pint_extension.get_units(sig.values)
    return ContinuousDataEven(values_am, sig.sample_step, sig.first_sample)

    
def am_demodulation_filter(sig, dt_smooth, mask_len):
    warnings.warn("not tested")
    top_freq = 1.0 / dt_smooth
    band = Segment([1e-12 * pint_extension.get_units(top_freq), top_freq])
    return band_pass_filter(sig.abs(), band, mask_len=mask_len)
