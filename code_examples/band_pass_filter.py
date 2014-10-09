from signal_processing import Segment
from signal_processing import continuous as cont
from signal_processing import uerg

white_noise = cont.generators.white_noise(uerg.sec, n_samples=2**16, amplitude=uerg.mamp)
white_noise.values_description = "white noise"

freq_range = Segment([0.3, 0.4], uerg.Hz)
white_noise_filterred = cont.filters.band_pass_filter(white_noise, freq_range, mask_len=32)
white_noise_filterred.values_description = "filterred"

fig_time = white_noise.plot()
white_noise_filterred.plot(fig_time)

fig_spectrum = white_noise.fft().abs().plot()
white_noise_filterred.fft().abs().plot(fig_spectrum)

