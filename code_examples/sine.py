from signal_processing import Segment
from signal_processing import ContinuousDataEven
from signal_processing import continuous as cont
from signal_processing import U_

sig_line_0 = ContinuousDataEven.generate('sine', U_.msec, 2 ** 7,
                                    270 * U_.volt, sine_freq=50 * U_.Hz)
noise = ContinuousDataEven.generate('white_noise', U_.msec, 2 ** 7,
                                    270 * 0.6 * U_.volt)
sig_line = sig_line_0 + noise

freq_range = Segment([48.0, 52.0], U_.Hz)
sig_line_clean = cont.filters.band_pass(sig_line, freq_range,
                                               mask_len=32)

sig_line.values_description = 'sine unclean'
sig_line_clean.values_description = 'clean'
cont.plot_few(sig_line, sig_line_clean)
