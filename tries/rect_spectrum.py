# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 12:15:34 2014

@author: noam
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
sample_rate = 1
sample_step = 1.0 / sample_rate
time = np.arange(-2 ** 9, 2 ** 9) * sample_step

T = 30.0
rect = np.zeros_like(time)
rect[np.abs(time) < T / 2] = 1

freqs = np.fft.fftfreq(len(time), sample_step)
rect_spec = np.fft.fft(rect)

trim = np.zeros_like(freqs)
filter_fine = 7
trim[np.abs(freqs) < filter_fine * 1.0 / T] = 1

rect_spec_trim = rect_spec * trim
rect_filterred = np.fft.ifft(rect_spec_trim)

#%%
plt.figure()
plt.xlabel("time")
plt.plot(time, rect)
plt.plot(time, rect_filterred)
plt.figure()
plt.xlabel("freq")
plt.plot(freqs, np.abs(rect_spec))
plt.plot(freqs, np.abs(rect_spec_trim))
