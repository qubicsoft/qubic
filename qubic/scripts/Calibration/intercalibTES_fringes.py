from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from qubicpack.utilities import Qubic_DataDir

import qubic
# import qubic.fibtools as ft
# import qubic.sb_fitting as sbfit
import qubic.selfcal_lib as sc


# Use a tool from qubicpack to get a path
basedir = Qubic_DataDir(datafile='instrument.py', )
print('basedir : ', basedir)
dictfilename = basedir + '/dicts/global_source_oneDet.dict'

# Get a dictionary
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
print(d['detarray'])

# Create an object
baseline = [25, 57]
q = qubic.QubicInstrument(d)
ca = sc.SelfCalibration(baseline, d)

S, Cminus_i, Cminus_j, Sminus_ij, Ci, Cj, Sij = ca.get_power_combinations(q)

plt.figure()
plt.imshow(Sij[:, :, 0], origin='lower')

full_real_fp, quart_fp = sc.get_real_fp(Sij[:, :, 0], quadrant=3)

plt.imshow(quart_fp, origin='lower')

FP_amp_intercalib = np.random.random((17, 17))

plt.imshow(FP_amp_intercalib, origin='lower')

fake_measurement = quart_fp * FP_amp_intercalib

plt.imshow(fake_measurement, origin='lower')