"""
.. module:: filters
    :synopsis: bla
"""



import warnings

import scipy as sp


from signal_processing.extensions import pint_extension
from signal_processing.extensions import scipy_extension


def band_pass(sig, freq_range, mask_len):
    """
    band pass filter of ContinuousDataEven

    parameters
    --------------
    freq_range: a Segment of frequencies

    notes
    ----------

    .. todo::
        design issue: shuold change interface.
        mask_len is a vector interface, not signal interface.
        the correct design would be with mask_duration: with units of the domain.
        maybe internally I want to pad it to 2 ** n for efficiency, maybe a
        parameter should be given to determine this.
        XXX at the moment this function is not stable with downsampling for this reason
    """
    warnings.warn('not tested')
    # TODO: test well
    freq_range.edges.ito(sig.sample_rate.units)
    print freq_range.edges
    print sig.sample_rate
    assert freq_range.end.magnitude < 0.5 * sig.sample_rate.magnitude
    # if error rises with firwin with units, wrap it: http://pint.readthedocs.org/en/0.5.1/wrapping.html
    mask_1 = sp.signal.firwin(mask_len, freq_range.edges.magnitude, pass_zero=False, nyq=0.5 * sig.sample_rate.magnitude)
    filterred_values = scipy_extension.smart_convolve(sig.values.magnitude, mask_1, mode="same") * pint_extension.get_units(sig.values)
    filterred = sig.new_values(filterred_values)
    return filterred
