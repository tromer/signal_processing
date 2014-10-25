import tempfile
import shutil
import os
import glob

import numpy as np
import scipy as sp


from signal_processing import U_, utils

from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.continuous import io



def test_read_wav():
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    _, f_temp = tempfile.mkstemp()
    sp.io.wavfile.write(f_temp, sample_rate.magnitude, values.magnitude)
    sig_read = io.read_wav(f_temp)

    assert sig.is_close(sig_read)


def test_read_wav_many():
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    dir_temp = tempfile.mkdtemp()
    io.write_wav(sig, os.path.join(dir_temp, "1.wav"))
    io.write_wav(sig, os.path.join(dir_temp, "2.wav"))

    sig_list = io.read_wav_many(dir_temp)
    expected_sig_list = [sig, sig]

    assert utils.is_close_many(sig_list, expected_sig_list)

    shutil.rmtree(dir_temp)





def test_write_wav():
    # copied from test_read_wav
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    _, f_temp = tempfile.mkstemp()
    io.write_wav(sig, f_temp)
    sig_read = io.read_wav(f_temp)

    assert sig.is_close(sig_read)


def test_write_wav_many():
    values = np.arange(10) * U_.milliamp
    sample_rate = 1.0 * U_.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)

    sig_list = [sig, sig]

    dir_temp = tempfile.mkdtemp()

    io.write_wav_many(sig_list, dir_temp)

    print os.listdir(dir_temp)

    sig_list_read = io.read_wav_many(dir_temp)
    print len(sig_list)
    print len(sig_list_read)

    print glob.glob(dir_temp + "*.wav")

    assert utils.is_close_many(sig_list, sig_list_read)

    shutil.rmtree(dir_temp)
