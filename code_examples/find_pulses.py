from signal_processing import Segment
from signal_processing import continuous as cont
from signal_processing.extensions.plt_extension import mark_horizontal_lines
from signal_processing import threshold
from signal_processing import uerg

sig = cont.generators.square_freq_modulated(uerg.sec, 2**9, uerg.volt, sine_freq=0.15 * uerg.Hz, period=100 * uerg.sec, duty=0.3)
pulses = threshold.crosses(cont.demodulation.am_hilbert(sig), threshold=0.5 * uerg.volt)
fig = sig.plot()
pulses.mark_edges(fig)
mark_horizontal_lines(0.5, fig, "thrshold")

