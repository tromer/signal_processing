# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 18:22:29 2014

@author: noam
"""

"""
http://pint.readthedocs.org/en/0.5.1/tutorial.html#using-pint-in-your-projects
"""

"""
TODO: it should acctualy be ureg, should rename this in the entire project
"""

from pint import UnitRegistry
uerg = UnitRegistry()
Q_ = uerg.Quantity
