from signal_processing import Segment
from signal_processing import ContinuousDataEven
from signal_processing import continuous as cont
from signal_processing import U_

white_noise = ContinuousDataEven.generate('white_noise', U_.sec,
                                          n_samples=2**16, amplitude=U_.mamp)
white_noise.values_description = "white noise"

freq_range = Segment([0.3, 0.4], U_.Hz)
white_noise_filterred = cont.filters.band_pass(white_noise, freq_range,
                                                      mask_len=32)
white_noise_filterred.values_description = "filterred"

cont.plot_few(white_noise, white_noise_filterred)
cont.plot_few(white_noise.fft().abs(), white_noise_filterred.fft().abs())
