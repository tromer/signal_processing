"""
.. module:: math
    :synopsis: bla
"""

import warnings
import numpy as np


from signal_processing.extensions import pint_extension
from signal_processing.extensions import numpy_extension

from signal_processing.continuous.filters import band_pass
from signal_processing.segment import Segment
"""
mathematical manipulations - except fouriers

"""

def diff(contin, n=1):
    """
    numeric differentiation of a ContinuousData
    a wrap around numpy.diff

    returns
    -------------------
    ContinuousData of the same type, of the same same length

    notes
    --------
    for n == 1:
    all points except the last one are calculated using np.diff,
    the last one is defined to be like the one before it.

    TODO: Design issues:
    --------------
    it's not clean / beautiful definition for the last sample, but it hardly matters.
    I decided that it returns a ContinuousData of the same length, so it
    desn't hurt signals of length 2 ** m, which are easier to fft
    maybe it's better to return a signal that have samples in the middle between each two samples of the original signal
    """
#    if type(contin) != ContinuousDataEven:
        #raise NotImplementedError

    new_vals = np.empty(len(contin.values))
    if n != 1:
        raise NotImplementedError
    elif n == 1:
        new_vals[:-1] = np.diff(contin.values.magnitude, 1)
        new_vals[-1] = new_vals[-2]
        new_vals = new_vals * pint_extension.get_units(contin.values) * contin.sample_rate ** n

        diffed = contin.new_values(new_vals)
        return diffed

def correlate(sig_stable, sig_sliding, mode='valid'):
    """
    a correlation between 2 signals. we try to relocate the sliding sig, to fit the location of the stable sig

    parameters
    --------------------
    sig_stable : ContinuousData

    sig_sliding : ContinuousData

    returns
    -------------
    the correlation as signal. the peak of the correlation should inticate the bast location for the first sample of sig_sliding


    """
    warnings.warn("correlate is not tested")
# commented this assertion out because I want this module not to import ContunuousDataEven
#    if not type(sig_stable) in [ContinuousDataEven,] or not type(sig_sliding) in [ContinuousDataEven,]:
        #raise NotImplementedError("implemented only for ContinuousDataEven")

    if not pint_extension.allclose(sig_stable.sample_step, sig_sliding.sample_step):
        raise NotImplementedError("implemented only for same sample step signals")

    if sig_stable.n_samples < sig_sliding.n_samples:
        warnings.warn("note that sig_stable has less points then sig_sliding, why is that?")

    # values
    a = sig_stable.values.magnitude
    b = sig_sliding.values.magnitude
    c = np.correlate(a, b, mode)
    sig_c_values = c * pint_extension.get_units(sig_stable.values) * pint_extension.get_units(sig_sliding.values) * sig_stable.sample_step

    #times
    if mode == 'full':
        domain_start = (-1) * sig_stable.sample_step * (-1 + 0.5 * (sig_stable.n_samples + sig_sliding.n_samples)) + sig_stable.domain_start
    elif mode == 'same':
        raise NotImplementedError("timing the correlation not implemented for same mode")
    elif mode == 'valid':
        raise NotImplementedError("timing the correlation not implemented for valid mode")

    sig_c = sig_stable.new_values(sig_c_values, assert_same_n_samples=False,
                                  new_domain_start = domain_start)
    # old old old
    # sig_c = ContinuousDataEven(sig_c_values, sig_stable.sample_step, first_sample)

    return sig_c


def correlate_find_shift(sig_stable, sig_sliding, mode='valid'):
    """

    returns
    ---------
    shift : Quantity
        the shift that the sliding signal need to be in max correlation with
        the stable sig
    """
    raise NotImplementedError

def correlate_find_new_location(sig_stable, sig_sliding, mode='valid', is_return_max=False):
    """
    for most of the documentation refer to correlate
    TODO: the signature of this function is not stable, according to user input it returns either 1 or 2 values

    parameters
    --------------------
    is_return_max : bool
        can return also the max value, in order to compare the success of different correlations



    behind the scences:
    ----------------------
    using correlate and np.argmax
    """
    corr = correlate(sig_stable, sig_sliding, mode)
    top_index = np.argmax(corr.values)
    top_domain_sample = corr.domain_start + corr.sample_step * top_index
    max_value = corr.values[top_index]

    if not is_return_max:
        return top_domain_sample
    else:
        return top_domain_sample, max_value


def convonlve(sig, mask, mode):
    """
    .. todo::
        decide how to nake the parameters. it's decided by how I percieve the convolution
        maybe mask should be named impulse_response?
    """
    raise NotImplementedError

def clip(sig, values_range):
    """
    parameters
    ---------------------
    sig : ContinuousData

    values_range : Segment

    """
#    if type(sig) != ContinuousDataEven:
        #raise NotImplementedError

    clipped_vals = np.clip(sig.values, values_range.start, values_range.end)

    clipped = sig.new_values(clipped_vals)
    return clipped






#%%

def hilbert(sig, mode='accurate', n_fft=None):
    """
    returns the analytic signal
    a wrap around sp.signal.hilbert
    """
    analytic_sig_values = pint_extension.hilbert(sig.values, mode, n_fft=n_fft)
    analytic_signal = sig.new_values(analytic_sig_values, assert_same_n_samples=False)
    return analytic_signal


def am_demodulation_convolution(sig, t_smooth):
    """
    params:
    t_smooth is the width in domain units, that you want to smooth together
    """
    warnings.warn("not tested")
    n_samples_smooth = np.ceil(t_smooth * sig.sample_rate)
    mask_am = numpy_extension.normalize(np.ones(n_samples_smooth), ord=1)
    values_am = np.convolve(np.abs(sig.values.magnitude), mask_am, mode="same") * pint_extension.get_units(sig.values)
    smoothed = sig.new_values(values_am)
    return smoothed


def am_demodulation_filter(sig, dt_smooth, mask_len):
    warnings.warn("not tested")
    top_freq = 1.0 / dt_smooth
    band = Segment([1e-12 * pint_extension.get_units(top_freq), top_freq])
    return band_pass(sig.abs(), band, mask_len=mask_len)
