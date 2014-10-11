from signal_processing import continuous as cont
from signal_processing.extensions.plt_extension import mark_horizontal_lines
from signal_processing import threshold
from signal_processing import U_

sig = cont.generators.square_freq_modulated(U_.sec, 2**9, U_.volt, sine_freq=0.15 * U_.Hz, period=100 * U_.sec, duty=0.3)
pulses = threshold.crosses(cont.demodulation.am_hilbert(sig), threshold=0.5 * U_.volt)
fig = sig.plot()
pulses.mark_edges(fig)
mark_horizontal_lines(0.5, fig, "thrshold")

