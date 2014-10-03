"""
this is a signal processing package.
it's heavily based on numpy, scipy, and uses pint for units support.

Data types
---------------
the package introduces a few data types to represent different datas
easily and efficiently:
1. ContinuousData - every continuous data, like signals.
2. Segments - a quantisation / clustering / finding areas of interest
within signals

Processes
-------------------
various processes to manipulate ContinuousData and Segments:
    frequency filtering, adjoining segments together, filtering by some properties.
    automatic parameters finding for various processes (thresholds)

some conventions
---------------------------------
1. XXX is a marker of highly problematic process.
2. is_each_..... a name for a boolian array which specifies whether
each element in another array fullfils some condition. it's a mask.

"""

from global_uerg import uerg, Q_

from .extensions import numpy_extension, scipy_extension, pint_extension

import .segment
from .segment import Segment
import .segments
from .segments import Segments, SegmentsOfContinuous
import .continuous_data
from .continuous_data import ContinuousData, ContinuousDataEven



