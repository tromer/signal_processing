# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 00:37:39 2014

@author: noam
"""

import numpy as np
from global_uerg import uerg
from segment import Segment
import continuous_data

def slice_wav(f_in, f_out, time):
    """
    slices a wav file on given times
    """
    s = continuous_data.read_wav(f_in)
    s_cut = s[time]
    continuous_data.write_wav(s_cut, f_out)
    
f_in = "/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus.wav"
f_out = "/home/noam/lab_project/Dropbox/Noam/Periodic recordings for Noam/fast-evo1-chassis-10100-C3000-N200_ettus_cut_2.wav"
#time = [5.720, 6.610]
time = Segment([10.3, 11.5], uerg.sec)
slice_wav(f_in, f_out, time)