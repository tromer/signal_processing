import warnings
import numpy as np


from continuous_data_even_obj import ContinuousDataEven
from signal_processing.continuous.math import hilbert
from signal_processing.continuous.math import diff
from signal_processing import uerg
from signal_processing.extensions import pint_extension

# TODO: there are some problematic issues with fft / hilbert /demodulations with not 2 ** n samples signals.


def pm(sig, mode='fast'):
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
    phase_sig = analytic_sig.new_values(phase)
    return phase_sig


def fm(sig, mode='fast'):
    """
    fm demodulation
    based on differentiating the pm demodulation
    """
    sig_phase = pm(sig, mode)
    angular_freq = diff(sig_phase)
    freq = angular_freq.gain(1.0 / (2 * np.pi))
    return freq
    
   
def am(sig, mode='fast'):
    #worning copied from pm
    if sig.n_samples < 2 ** 10:
        warnings.warn("this pm-modulation technique doesn't work well on short signals, the mistakes on the edges are big")
    analytic_sig = hilbert(sig, mode)
    envelope = np.abs(analytic_sig.values.magnitude) * pint_extension.get_units(analytic_sig.values)
    sig_am = ContinuousDataEven(envelope, analytic_sig.sample_step, analytic_sig.first_sample)
    return sig_am
    

