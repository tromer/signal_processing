Background
----------------
This module introduces a natural interface for signal processing.
It also introduces some signal processing functions.
it's heavily based on numpy, scipy, and uses pint for units support.

Motivation
-----------------------
Scipy has great mathematical tools, but doesn't have a natural interface for signal processing.
One want to think of a signal as a continuous data (the interface), not vector of samples (internal implementation). One also want to think with physical units of the data.

Data types - the interface for signal processing
----------------------------------------------------
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

naming conventions
---------------------------------
0. As a rule of thumb, scipy and numpy naming conventions are preferable.
1. XXX is a marker of highly problematic process.
2. is_each_..... a name for a boolian array which specifies whether
each element in another array fullfils some condition. it's a mask.

TODO
---------------------------------
1. refactoring issues:
    name - first_sample, last_sample > sample_start, sample_end
    separate tests
    separate big files to classes, sub modules (generators, fft, demodulations, plot, io...)
    *  use a refactoring tool. vim rope?
2. testing:
   choose a testing package (probably nose or pytest)
   seperate tests and create a testing script
   add tests


