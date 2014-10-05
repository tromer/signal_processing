import scipy as sp
from scipy import signal
    
"""
def freq_filter(contin, freq_ranges, ?, ?, ?):
    raise NotImplementedError
    
     
    @uerg.wraps(None, (None, uerg.Hz, uerg.Hz, None, None, None, uerg.Hz))    
    def firwin_pint(numtaps, cutoff, width, window, pass_zero, scale, nyq):
        return sp.signal.firwin(numtaps, cutoff, width, window, pass_zero, scale, nyq)
"""

def band_pass_filter(sig, freq_range, mask_len):
    """
    band pass filter of ContinuousDataEven
    
    parameters:
    freq_range: a Segment of frequencies
    
    implemented using np.convolve with a mask. maybe with fft is better
    
    XXX TODO: shuold change interface. maks len is something without units. shoult receive something else
    XXX this function is not stable with downsampling for this reason
    """
    warnings.warn('not tested well')
    #TODO: test well
    freq_range.edges.ito(sig.sample_rate.units)
    print freq_range.edges
    print sig.sample_rate
    assert freq_range.end.magnitude < 0.5 * sig.sample_rate.magnitude
    # if error rises with firwin with units, wrap it: http://pint.readthedocs.org/en/0.5.1/wrapping.html
    mask_1 = sp.signal.firwin(mask_len, freq_range.edges.magnitude, pass_zero=False, nyq=0.5 * sig.sample_rate.magnitude)
    filterred_values = scipy_extension.smart_convolve(sig.values.magnitude, mask_1, mode="same") * pint_extension.get_units(sig.values)
    filterred = ContinuousDataEven(filterred_values, sig.sample_step, sig.first_sample)
    return filterred
    

