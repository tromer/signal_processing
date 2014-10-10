import numpy as np

from continuous_data_even_obj import ContinuousDataEven
import generators
from signal_processing import uerg


def am(amp_t, f_carrier, phase_0_carrier=0):
    """
    parameters:
    -------------------
    amp_t : ContinuousDataEven

    f_carrier : uerg.Quantity

    phase_0_carrier : float or radians?
    """
    if not (amp_t.values >= 0).all():
        raise ValueError("am amp_t which is not positive")
    carrier = generators.generate_sine(amp_t.sample_step, amp_t.n_samples, uerg.dimensionless, f_carrier, phase_0_carrier)
    am_modulated = amp_t * carrier
    return am_modulated


def pm(phase_t, amp_carrier):
    """
    parameters:
    ------------
    phase_t : ContinuousDataEven with dimensionless values

    amp_carrier

    """
    raise NotImplementedError
    if phase_t.values_unit != uerg.dimensionless:
        raise ValueError("phase phase_t with units")
    pm_modulated_values = ContinuousDataEven(np.sin(phase_t.values), phase_t.sample_step, phase_t.first_sample) * amp_carrier
    return pm_modulated_values


def fm(freq_t, amp_carrier):
    """

    """
    raise NotImplementedError
    phase_values = np.cumsum(2 * np.pi * freq_t.values * freq_t.sample_step)
    phase_t = ContinuousDataEven(phase_values, freq_t.sample_step, freq_t.first_sample)
    fm_modulated = pm(phase_t, amp_carrier)
    return fm_modulated
