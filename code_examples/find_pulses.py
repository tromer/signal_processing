from signal_processing import ContinuousDataEven
from signal_processing import threshold
from signal_processing import U_

envelope = ContinuousDataEven.generate('square', U_.sec, 2**9, U_.volt,
                                       period=100 * U_.sec, duty=0.3)
sig = envelope.modulate('am', f_carrier=0.15 * U_.Hz)
pulses = threshold.crosses(sig.demodulate('am'), threshold=0.5 * U_.volt)
fig = sig.plot()
pulses.mark_edges(fig)
