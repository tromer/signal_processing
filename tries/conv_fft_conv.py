# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 18:31:49 2014

@author: noam
"""
import time
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt

def compare(min_pow_vec, max_pow_vec, pow_mask):
    times_conv = []
    times_fft = []
    
    m = np.arange(2 ** pow_mask)
    for i in range(min_pow_vec, max_pow_vec + 1):
        print i
        v = np.arange(2 ** i)
        
        t_0 = time.time()
        a = np.convolve(v, m, mode='same')
        t_1 = time.time()
        dt = t_1 - t_0
        times_conv.append(dt)
        
        t_0 = time.time()
        b = sp.signal.fftconvolve(v, m, mode='same')
        t_1 = time.time()
        dt = t_1 - t_0
        times_fft.append(dt)
        
    del(v)
    del(a)
    del(b)
    
    times_conv_1 = np.array(times_conv)
    times_fft_1 = np.array(times_fft)
    return times_conv_1, times_fft_1

min_pow = 1
max_pow = 25
x = np.arange(min_pow, max_pow + 1)
t_conv, t_fft = compare(min_pow, max_pow, 8)
plt.plot(x, t_conv, label="conv")
plt.plot(x, t_fft, label="fft")
plt.plot(x, t_fft / t_conv, label='ratio')
plt.legend(loc='best')
    
"""    
np.save("~/temp/conv.npy", times_conv_1)
np.save("~/temo/fftconv.npy", times_fftconve_1)
""" 
    