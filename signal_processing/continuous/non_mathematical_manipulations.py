import numpy as np

from continuous_data_even_obj import ContinuousDataEven


from signal_processing.extensions import pint_extension

"""
non mathematical manipulations

"""



def concatenate(sig_list):
    """
    concatenate signals

    parameters:
    -------------------
    sig_list : list of ContinuousData

    returns:
    -----------------
    sig : ContinuousData
    """
    if len(sig_list) == 0:
        raise ValueError("no signals in the list")

    if len(sig_list) == 1:
        return sig_list[0]

    if not np.unique(map(type, sig_list))[0] == ContinuousDataEven:
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type")

    sample_steps = pint_extension.array(map(lambda(s) : s.sample_step, sig_list))
    sample_step = sig_list[0].sample_step
    if not pint_extension.allclose(sample_step, sample_steps):
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type, with same sample rate")



    domain_starts = pint_extension.array(map(lambda(s) : s.domain_start, sig_list))
    domain_ends = pint_extension.array(map(lambda(s) : s.domain_end, sig_list))

    gaps = domain_starts[1:] - domain_ends[:-1]
    if not pint_extension.allclose(sample_step, gaps):
        print gaps
        print sample_step
        raise NotImplementedError("concatenate implemented only for ContinuousDataEven type, with same sample rate, and right one after the other, at diffrence of sample_step")

    values = pint_extension.concatenate(map(lambda s : s.values, sig_list))
    sig = ContinuousDataEven(values, sample_step, sig_list[0].domain_start)
    return sig


def resample(sig, new_sample_points):
    """
    create a new sig object, that represents the same signal, on different sample points.
    algorithm: linear intrapulation
    """
    raise NotImplementedError

