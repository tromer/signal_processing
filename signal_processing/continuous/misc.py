"""
.. module:: misc
    :synopsis: bla
"""

import warnings
from filters import band_pass
import demodulate


def filter_downsample_fm_demodulation_base_band_filter(raw_sig, freq_range, freq_mask_len, down_factor, base_band_range, base_band_mask_len):
    """
    preprocessing of the signal.

    returns
    -------------
    signal : ContinuousDataEven
        signal after filtering, de-fm

    """
    warnings.warn("filter_downsample_fm_demodulation_base_band_filter not tested")
    # step_2: filtering the raw_data
    # Note:  getting memory error with long signals (290 M-Byte and more). the problem is with the convolution. need to try fftconvolve
    # maybe we want to filter, only after finding where are the traces, and with the exact band of the real raw_sig (not noise)
    filterred = band_pass(raw_sig, freq_range, freq_mask_len)

    if False:
        filterred[parameters.time_for_quick_plot].plot()

    # step_3: down sampling, (which also aliases, and changed the freq of the signal)
    # XXX: maybe the frequency band is close to the +0.5 nyq, so the fm
    # signal would be split between +0.5nyq and -0.5nyq
    # TODO: cope with the freq wrap problems. acctually np.unwrap could
    # be a fine solution, after the fm-demosulation.
    # another approach - calculate where the signal should be "falling"
    # it's a modular / sawtooth calculation. if it's "splinched", then
    # just add and use modulu (roll it)
    down = filterred.down_sample(down_factor)

    if False:
        down.plot()
        down.fft().abs().plot()

    # step_4: fm demodulation
    de_fm = demodulate.fm(down, mode='fast')

    if False:
        de_fm.plot()

    # step 4.1 filter the base-band signal
    sig = band_pass(de_fm, freq_range=base_band_range, mask_len=base_band_mask_len)
    return sig


