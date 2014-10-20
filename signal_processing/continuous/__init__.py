
"""

Data types
=================
.. module:: continuous
    :synopsis: signals.

.. automodule:: signal_processing.continuous.continuous_data_obj
    :members:

.. automodule:: signal_processing.continuous.continuous_data_even_obj
    :members:

filters
===========
.. automodule:: signal_processing.continuous.filters
    :members:

demodulate
===============
.. automodule:: signal_processing.continuous.demodulate
    :members:

modulate
=============
.. automodule:: signal_processing.continuous.modulate
    :members:

misc
==========
.. automodule:: signal_processing.continuous.misc
    :members:

math
=======
.. automodule:: signal_processing.continuous.math
    :members:

io
========
.. automodule:: signal_processing.continuous.io
    :members:

non_mathematical_manipulations
==================================
.. automodule:: signal_processing.continuous.non_mathematical_manipulations
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
