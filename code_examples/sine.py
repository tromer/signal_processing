from signal_processing import Segment
from signal_processing import continuous as cont
from signal_processing import uerg

sig_line = cont.generators.generate_sine(uerg.msec, 2 ** 7, 270 * uerg.volt, sine_freq=50 * uerg.Hz) + \
cont.generators.white_noise(uerg.msec, 2 ** 7, 270 * 0.6 * uerg.volt)
freq_range = Segment([48.0, 52.0], uerg.Hz)

sig_line_clean = cont.filters.band_pass_filter(sig_line, freq_range, mask_len=32)

sig_line.values_description = 'sine unclean'
sig_line_clean.values_description = 'clean'
cont.plot_few(sig_line, sig_line_clean)
