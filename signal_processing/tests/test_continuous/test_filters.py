import numpy as np

from signal_processing import uerg

from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.segment import Segment


from signal_processing.continuous import filters


def visual_test_band_pass_filter():
    sample_step = uerg.sec
    np.random.seed(13)
    # TODO: use generators.white_noise instead
    white_noise = ContinuousDataEven((np.random.rand(2048) - 0.5)* uerg.mamp, sample_step)
    white_noise_spec = white_noise.fft()
    freq_range = Segment(np.array([0.3, 0.4]) * uerg.Hz)
    white_noise_filterred = filters.band_pass_filter(white_noise, freq_range, 32)
    white_noise_filterred_spec = white_noise_filterred.fft()
    white_noise_spec.abs().plot()
    white_noise_filterred_spec.abs().plot()
