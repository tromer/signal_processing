
"""
.. module:: continuous
    :synopsis: signals.

continuous
================
.. automodule:: signal_processing.continuous.continuous_data_obj
.. autoclass:: ContinuousData
    :members:

.. automodule:: signal_processing.continuous.continuous_data_even_obj
.. autoclass:: ContinuousDataEven
    :members:

io
========
.. automodule:: signal_processing.continuous.io
    :members:
"""
from continuous_data_obj import ContinuousData
from continuous_data_even_obj import ContinuousDataEven

import non_mathematical_manipulations
import math
import generators
import io
import filters
import demodulate
import modulate
import misc

from io import read_wav
from non_mathematical_manipulations import concatenate

from signal_processing.extensions.plt_extension import plot_few
