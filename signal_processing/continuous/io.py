"""
.. module:: io
    :synopsis: bla
"""

"""
io

"""
import os
import glob
import warnings

import numpy as np
import scipy as sp
from scipy.io import wavfile

from continuous_data_even_obj import ContinuousDataEven

from signal_processing import U_

def read_wav(filename, domain_unit=U_.sec, first_sample=0,
             value_unit=U_.milliamp, expected_sample_rate_and_tolerance=None,
             channels=None):
    """
    read wav file to ContinuousDataEven.
    implemented only for one channal
    for multiple channels we probably want to return a list of ContinuousDataEven

    parameters
    ------------
    domain_unit
        the unit of the domain. usually sec

    first_sample
        in case it's not 0

    value_unit
        the unit of the values

    channels
        if it's list, it says which channels to return

    XXX TODO: understand whether it reads 16pcm correctly. it's not sure
    some source:
    http://nbviewer.ipython.org/github/mgeier/python-audio/blob/master/audio-files/audio-files-with-scipy-io.ipynb#Reading

    returns
    ------------
    if channels == None, returns signal
    if channels != None, returns a list of signals

    rename
    ----------
    first_sample to domain_start
    """
    sample_rate, raw_sig = sp.io.wavfile.read(filename)
    sample_rate = 1.0 * sample_rate / domain_unit
    raw_sig = raw_sig * value_unit
    if expected_sample_rate_and_tolerance != None:
        # shold raise a meaningful excepion.
        is_sample_rate_as_expected = np.abs(sample_rate - expected_sample_rate_and_tolerance[0]) < expected_sample_rate_and_tolerance[1]
        if not is_sample_rate_as_expected:
            warnings.warn("sample rate is not as expected")

    if channels == None:
        sig = ContinuousDataEven(raw_sig, 1.0 / sample_rate, first_sample)
        return sig

    else:
        warnings.warn("reading multiple channels is not tested")
        sig_list = []
        for c in channels:
            sig_c = ContinuousDataEven(raw_sig[:, c], 1.0 / sample_rate, first_sample)
            sig_list.append(sig_c)

        return sig_list


    #return signal

def read_wav_many(directory, domain_unit=U_.sec, first_sample=0,
                  value_unit=U_.milliamp,
                  expected_sample_rate_and_tolerance=None, channels=None):
    """
    reads all the wavs in a directory

    add docs
    parameters
    ------------
    directory : str
    """
    sig_list = []
    print "directory: , ", directory
    print "path:, ", os.path.join(directory, "*.wav")
    files = glob.glob(os.path.join(directory, "*.wav"))
    print files
    for f in files:
        curr_sig = read_wav(f, domain_unit, first_sample, value_unit,
                                 expected_sample_rate_and_tolerance, channels)
        sig_list.append(curr_sig)

    return sig_list

def write_wav(contin, filename):
    """
    write contin to wav file, and return the units of the axis, and the first sample

    Note: I didn't think deeply about the signature of this function
    TODO: add way to rescale between the domain unit and sec

    example
    --------------
    s = continuous_data.read_wav("/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus.wav")
    s_cut = s[Segment([5.720, 6.610], U_.sec)]
    continuous_data.write_wav(s_cut, "/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus_cut.wav")

    known issues:
    ---------------
    usually the data is floats. when written to a file it is read well by
    read_wav, by the program audacity, but not by python module wave.
    wave.open doesn't recognize the format

    possible improvement
    ---------------------
    if it's possible to write some parameters as the wav metadata, it's great.
    so far I didn't find a way
    """
    print filename
    if contin.domain_samples.dimensionality != U_.sec.dimensionality:
        raise NotImplementedError

    else:
        with open(filename, 'wb') as wav_file:
            sp.io.wavfile.write(
                wav_file,
                rate=contin.sample_rate.to(U_.Hz).magnitude,
                data=contin.values.magnitude)


def write_wav_many(contin_list, directory):
    """
    write a list of signals to a directory
    """
    for i in xrange(len(contin_list)):
        curr_sig = contin_list[i]
        name = str(i)
        write_wav(curr_sig, os.path.join(directory, '.'.join([name, "wav"])))

def fromfile(f):
    """
    read ContinuousData / ContinuousDataEven from file
    TODO:
    ---------
    decide about file format.
    probably a folder with wav file, and txt/csv/xml file for units
    etc
    """
    raise NotImplementedError
