from signal_processing import Segment
from signal_processing import continuous as cont
from signal_processing import uerg

white_noise = cont.generators.white_noise(uerg.sec, n_samples=2**16, amplitude=uerg.mamp)

freq_range = Segment([0.3, 0.4], uerg.Hz)
white_noise_filterred = cont.filters.band_pass_filter(white_noise, freq_range, mask_len=32)

white_noise_spec = cont.fft(white_noise)
white_noise_filterred_spec = cont.fft(white_noise_filterred)
fig = cont.plot(white_noise_spec, is_abs=True)[0]
cont.plot(white_noise_filterred_spec, fig, is_abs=True)

