Background
----------------
This module introduces a natural interface for signal processing,
as well as various signal processing functions.
It's based on numpy, scipy, and uses pint for units support.

Motivation
-----------------------
Scipy has great mathematical tools, but it lacks a * natural interface * for signal processing.

    * A signal as a continuous data (the interface), not vector of samples (internal implementation).
    * A signal should support physical unit. it's a measurement!

Data types - the interface for signal processing
----------------------------------------------------
the package introduces a few data types to represent different datas
easily and efficiently:

1) Object that represent continuous data such as voice recording (a signal).
other examples:

    - continuous spatial measurement. Such as mountain's profile. 
    - continuous kinematic measurement: position of something as a function of time.
    - a distribution (continuous histogram)
    - a spectrum of another signal (FFT).

2) Object that represent segments of interest within a signal, such as:
    
    - times and amplitudes of pulses of energy in an electromagnetic signal.
    - locations of seismic bursts in a continuous seismic measurement.
    - cluster: ranges that contain most of the value in a continuous distribution.

The use of these objects eliminates irrelevant information and thus simplifies the code.



Processes
-------------------
various processes to manipulate ContinuousData and Segments:

    - frequency filtering, modulations, demodulatioins
    - Auto noise level detection, threshold operations, algorithms to cope with noise.

How to proceed
-------------------

1. look at the code examples.
2. read the documentation of ContinuousData, ContinuousDataEven and Segments. it explains the they way this package percieves signals.

naming conventions
---------------------------------
0. As a rule of thumb, scipy and numpy naming conventions are preferable.
1. XXX is a marker of highly problematic process.
2. is_each_..... a name for a boolian array which specifies whether
each element in another array fullfils some condition. it's a mask.

Main issues before first release
---------------------------------
1. refactoring issues:

   1. name - first_sample, last_sample > domain_start, domain_end.
   2. shorten the name of some functions in the sub-modules (adjoin, filter. threshold etc.) since they are in a module, they can have a less indicative name
   3. change cotinuous/plots.py to continuous/visualisation.py and fix imports
   4. in tests and examples, use continuos/generators.py instead if creating a signal manualy
   5. use verbs instead of nouns as names of modules. generate, modulate, demodulate

    * use a refactoring tool. vim rope?
    * python-mode !
       
2. testing:
   
   1. so far using nose. maybe should use pytest?
   2. use test classes.
   3. add tests
   4. grep all the "not tested" signs (there are warnings that some functions are not tested

3. choose the right license.
4. choose a way to manage the issues and TODO's.
   grepable text that indicates issue in the code:

   * TODO
   * XXX
   * design issus
   * not tested
   * rename
   * refactor
   * encapsulate
   * old old old
   * link to
   * deprecated
   * NotImpmentedError

#. make sure to remove the use of old interface (like module generators)

#. arrange the imports according to a certain order even within the package, for example:
   import warnings
   import numpy as np
   from signal_processing import uerg
   from signal_processing import extension
   from signal_processing import Segment
   from signal_processing import continuous
#. find how to let some functions share documentation of parameters with same behaviour. very important when I wrap functions. maybe links?
#. improve use of exceptions. design the package exceptions, and use pint exceptions for units errors
#. use sphinx to compile documentation.
#. tag a commit as release 0.1

Design principles
---------------------
1. The API of the ContinuousData object have several distinguished layers and they have to be repected.

   a. the layer that accesses the internals. and returns the values, and sample times of the signal.
   b. mathematical operations, or mildly complex operations such as addition, absolute value etc. this methods to no use the internals, but instead use the first layer.
   c. there are some operations that are very common such doing fft to your signal, or plotting it to gain some intuition. They are methods, instead of external functions, because they are used all the time. However, they are percieved as a this connection to what actually does the logic (numpy.fft, or plt.plot). They must not contain any logic of there own. If they need any logic, it should be implemented as a second layer method.

2. This package handles only signal_processing.\n
   in some cases it needs a service that logically lies in the responsability of some other package (numpy, scipy, pint, matplotlib).\n
   in this cases the service (function in most cases) is put on the coresponding file in the extensions/ sub-package. \n
   the core sub-package: signal_processing, uses this services, but should never implement them on it's own, as it's not it's reponsability.\n
   Ideally, in the future all this extension files would be incorporated to the cpresponding modules.
