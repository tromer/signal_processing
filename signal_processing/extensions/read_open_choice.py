# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:46:20 2014

@author: noam
"""

from matplotlib import mlab


from signal_processing import ContinuousDataEven
from signal_processing import U_
#%%
def read_open_choice_csv(path, description_ch_1=None, description_ch_2=None):
    names_1 = ["headers_1", "params_1", "some_units_1", "time_1", "channel_1", "break"] + ["headers_2", "params_2", "some_units_2", "time_2", "channel_2"]
 
    table = mlab.csv2rec(path, names=names_1)
    
    dt = float(table["params_1"][1]) * U_(table["some_units_1"][1])
    unit_channel_1 = U_(table["params_1"][7].lower())
    unit_channel_2 = U_(table["params_2"][7].lower())
    #%%
    ch_1 = ContinuousDataEven(table["channel_1"] * unit_channel_1, dt, values_description=description_ch_1)
    ch_2 = ContinuousDataEven(table["channel_2"] * unit_channel_2, dt, values_description=description_ch_2)
    
    
    return ch_1, ch_2
#%%

def visual_test_read_open_choice_csv():
    add = "/home/noam/noam_personal/studies/physics/lab_b/magnetism/histeresis/1.csv"
    ch_1, ch_2 = read_open_choice_csv(add, "H", "B")
    print ch_1
    ch_1.plot()