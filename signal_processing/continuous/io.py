"""
io

"""

import numpy as np
from scipy.io import wavfile

from signal_processing import uerg

def read_wav(filename, domain_unit=uerg.sec, first_sample=0, value_unit=uerg.milliamp, expected_sample_rate_and_tolerance=None, channels=None):
    """
    read wav file to ContinuousDataEven.
    implemented only for one channal
    for multiple channels we probably want to return a list of ContinuousDataEven
    
    parameters:
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
    
    returns:
    ------------
    if channels == None, returns signal
    if channels != None, returns a list of signals
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
    
def write_wav(contin, filename):
    """
    write contin to wav file, and return the units of the axis, and the first sample
    
    Note: I didn't think deeply about the signature of this function
    TODO: add way to rescale between the domain unit and sec
    
    example
    --------------
    s = continuous_data.read_wav("/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus.wav")
    s_cut = s[Segment([5.720, 6.610], uerg.sec)]
    continuous_data.write_wav(s_cut, "/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus_cut.wav")

    """
    if contin.domain_samples.dimensionality != uerg.sec.dimensionality:
        raise NotImplementedError
    else:
        sp.io.wavfile.write(filename, rate=contin.sample_rate.to(uerg.Hz).magnitude, data=contin.values.magnitude)


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
    
   

