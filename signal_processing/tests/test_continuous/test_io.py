import tempfile
import os

import numpy as np
import scipy as sp


from signal_processing import U_

from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.continuous import io



def test_read_wav():
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    f_temp = tempfile.TemporaryFile()
    sp.io.wavfile.write(f_temp, sample_rate.magnitude, values.magnitude)
    sig_read = io.read_wav(f_temp)

    assert sig.is_close(sig_read)
    f_temp.close()


def test_read_wav_many():
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    dir_temp = tempfile.gettempdir() + 'signal_processing/tests/'
    io.write_wav(sig, dir_temp + "1.wav")
    io.write_wav(sig, dir_temp + "2.wav")

    sig_list = io.read_wav_many(dir_temp)
    expected_sig_list = [sig, sig]

    for i in xrange(len(sig_list)):
        assert sig_list[i].is_close(expected_sig_list[i])

    os.removedirs(dir_temp)





def test_write_wav():
    # copied from test_read_wav
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    f_temp = tempfile.TemporaryFile()
    io.write_wav(sig, f_temp)
    sig_read = io.read_wav(f_temp)

    assert sig.is_close(sig_read)
    f_temp.close()



