from signal_processing import uerg
from signal_processing.continuous import generators, modulate


def test_am():
    sig = generators.generate_square(uerg.sec, 2 ** 8, amplitude=uerg.volt, period=10 * uerg.sec)
    am_modulated = modulate.am(sig, 0.15 * uerg.Hz)
    expected_am_modulated = sig * generators.generate_sine(uerg.sec, 2 ** 8, amplitude=uerg.dimensionless, sine_freq=0.15 * uerg.Hz)

    assert am_modulated.is_close(expected_am_modulated)
