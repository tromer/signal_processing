
from global_uerg import uerg, Q_

from .extensions import numpy_extension, scipy_extension, pint_extension

import segment
from .segment import Segment
import segments
from .segments import Segments, SegmentsOfContinuous
import continuous_data
from .continuous_data import ContinuousData, ContinuousDataEven

from .histograms import *
from .threshold import *

# __all__ is not included here, since I do not expect from signal_processing import *, a statement that will create great confusion

#TODO
"""
* use a refactoring tool. vim rope?
refactoring issues:
    name - first_sample, last_sample > sample_start, sample_end
    separate tests
    separate big files to classes, sub modules (generators, fft, demodulations, plot, io...)


"""
