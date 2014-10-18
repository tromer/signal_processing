# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 18:00:21 2014

@author: noam
"""

import time
import numpy as np

x = np.arange(2 ** 15)
t_0 = time.time()
y = np.fft.rfft(x)
t_1 = time.time()
dt = t_1 - t_0
print dt



x = np.arange(2 ** 15 - 10)
t_0 = time.time()
y = np.fft.rfft(x)
t_1 = time.time()
dt = t_1 - t_0
print dt