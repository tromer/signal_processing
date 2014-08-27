# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 18:48:21 2014

@author: noam
"""

import time
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
#%%
N = 2 ** 24
M = 2 ** 7

t = np.arange(N)
sig = np.random.rand(N)
mask = np.random.rand(M)
#%%
t_0 = time.time()
c_fir = sp.signal.lfilter(mask, 1.0, sig)
t_1 = time.time()
dt_fir = t_1 - t_0
print dt_fir
#%%
t_0 = time.time()
c_conv = np.convolve(sig, mask, mode="same")
t_1 = time.time()
dt_conv = t_1 - t_0
print dt_conv
#%%
t_0 = time.time()
c_fftconv = sp.signal.fftconvolve(sig, mask, mode="same")
t_1 = time.time()
dt_fftconvolve = t_1 - t_0
print dt_fftconvolve


#%%

plt.figure()
plt.plot(t, sig, label="sig")
plt.plot(t, c_fir, label="FIR")
plt.plot(t, c_conv, label="conv")
plt.legend(loc="best")

#%%
sample_rate = 1.0
nyq = sample_rate * 0.5
f_1 = 0.17
mask_len = 2 ** 6
mask_2 = sp.signal.firwin(mask_len, cutoff=f_1, nyq=nyq)
plt.figure()
plt.plot(mask_2)
plt.figure()
plt.plot(np.fft.fftfreq(mask_len), np.abs(np.fft.fft(mask_2)))

#%%
M = 2 ** 6
sample_rate = 1.0
f_min = 0.15
f_max = 0.20
mask_1 = sp.signal.firwin(M, np.array([f_min, f_max]), pass_zero=False, nyq=0.5 * sample_rate)
plt.figure()
plt.xlabel("time")
plt.plot(mask_1)
plt.figure()
plt.xlabel("freq")
plt.plot(np.fft.fftfreq(M, 1.0 / sample_rate), np.abs(np.fft.fft(mask_1)))



