from signal_processing import U_
from signal_processing.continuous import generators, modulate


def test_am():
    sig = generators.generate_square(U_.sec, 2 ** 8, amplitude=U_.volt, period=10 * U_.sec)
    am_modulated = modulate.am(sig, 0.15 * U_.Hz)
    expected_am_modulated = sig * generators.generate_sine(U_.sec, 2 ** 8, amplitude=U_.dimensionless, sine_freq=0.15 * U_.Hz)

    assert am_modulated.is_close(expected_am_modulated)
