# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:46:20 2014

@author: noam
"""

import pandas as pd
import numpy as np
from matplotlib import mlab
from matplotlib import mlab


import signal_processing
from signal_processing import continuous as cont
from signal_processing import U_
#%%
names_1 = ["headers_1", "params_1", "some_units_1", "time_1", "channel_1", "break"] + ["headers_2", "params_2", "some_units_2", "time_2", "channel_2"]
#%%
add = "/home/noam/noam_personal/studies/physics/lab_b/magnetism/histeresis/1.csv"

#%%

t = mlab.csv2rec(add, names=names_1)

dt = float(t["params_1"][1]) * U_(t["some_units_1"][1])
unit_channel_1 = U_(t["params_1"][7].lower())
unit_channel_2 = U_(t["params_2"][7].lower())
#%%
ch_1 = cont.ContinuousDataEven(t["channel_1"] * unit_channel_1, dt)
ch_2 = cont.ContinuousDataEven(t["channel_2"] * unit_channel_2, dt)
#%%
ch_1.plot()