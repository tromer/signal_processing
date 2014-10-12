#XXX
#XXX
#this example uses old interface
import signal_processing as sigpro
from sigpro.global_ureg import ureg, Q_

oscillator, motor =  sigpro.continuous_data.read_wav("~/harmonic.wav", ureg.rad, channels=[0,1])
oscillator, motor = oscillator.down_sample(2), motor.down_sample(2)
fig, junk = sigpro.continuous_data.plot_quick(oscillator)
sigpro.continuous_data.plot(motor, fig)

oscillator_am = sigpro.continuous_data.am_hilbert(oscillator)
threshold = sigpro.threshold.auto_threshold(mode='mean', factor=2)
locations_of_oscillations = sigpro.threshold.thrshold_crosses(oscillator_am, threshold)

oscillations = sigpro.SegmentsOfContinuous(locations_of_oscillations, oscillator)
motors = sigpro.SegmentsOfContinuous(locations_of_oscillations, motor)

oscillations_amp = locations_of_oscillations.each_mean_value
motors_amp = motors.each_mean_value
oscillations_freq = oscillations.each_carrier_frequency
response_curve = sigpro.ContinuousData(oscillations_amp / motors_amp, oscillations_freq)
sigpro.continuous_data.plot_quick(response_curve)


