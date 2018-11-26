#!/bin/env python
from __future__ import division
import sys
import os
import time

import healpy as hp
import numpy as np
import matplotlib.pyplot as mp

import qubic
from pysimulators import FitsArray

import SpectroImLib as si

tol = 1e-3

### Instrument ###
d = qubic.qubicdict.qubicDict()
d.read_from_file("parameters.dict")

### Sky ###
skypars = {'dust_coeff':1.39e-2, 'r':0}
x0 = si.create_input_sky(d, skypars) #shape is (num of sub-bands, npix, IQU)

### TOD ###
p = qubic.get_pointing(d)
TOD = si.create_TOD(d, p, x0)



