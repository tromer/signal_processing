"""
from global_uerg import uerg, Q_
"""
from pint import UnitRegistry
uerg = UnitRegistry()
Q_ = uerg.Quantity

from .extensions import numpy_extension, scipy_extension, pint_extension

from .segment import Segment
import segments
from .segments.segments_obj import Segments
from .segments.segments_of_continuous_obj import SegmentsOfContinuous

import continuous
from .continuous.continuous_data_obj import ContinuousData
from .continuous.continuous_data_even_obj import ContinuousDataEven

import histograms
import threshold

# __all__ is not included here, since I do not expect from signal_processing import *, a statement that will create great confusion


