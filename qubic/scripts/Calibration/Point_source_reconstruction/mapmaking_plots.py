"""$Id: mapmaking_plots.py
$auth: Martin Gamboa <mgamboa@fcaglp.unlp.edu.ar> & Jean-Christophe Hamilton & James Murphy
$created: Fri 11 Feb 2022

Inspired and using the methods created by Jean-Christophe and James Murphy 
		to do map-making with real data + demodulated scripts done by James. 
 
This file contains functions to make consistency plots to check all is working well 

"""

import os
import sys 

import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits as pyfits

import scipy.ndimage.filters as f
import qubic.sb_fitting as sbfit
import qubic.demodulation_lib as dl

def plot_scan(time_axis, t_src, data_src, az, el):
	"""
	Plot the scan in azimuth and elevation. Data + Calibration source
	"""

	plt.rc('figure',figsize = (12,12))

	tinit = time_axis[0]

	plt.subplot(2,2,1)
	plt.plot((time_axis - tinit)/3600, az,',')
	plt.xlabel('Time [h]')
	plt.ylabel('Az')
	plt.subplot(2,2,2)
	plt.plot((time_axis - tinit)/3600, el,',')
	plt.xlabel('Time [h]')
	plt.ylabel('El')
	plt.ylim(30,70)

	plt.subplot(2,2,3)
	plt.plot(az * np.cos(np.radians(el)), el,',')
	plt.xlabel('Az')
	plt.ylabel('El')

	plt.subplot(2,2,4)
	plt.plot((t_src - tinit)/3600, data_src,',')
	plt.xlabel('Time [h]')
	plt.ylabel('Src Data')


	return

def plot_raw_data(tod_time, tod_data, calsrc_time, calsrc_data,
	TESNum = None, asic = None):

	"""
	Plot calibration source and raw data in hours
	"""
	plt.plot(calsrc_time / 3600, dl.renorm(calsrc_data), 
		label='Calsource', color='tab:orange')
	plt.plot(tod_time / 3600, dl.renorm(tod_data), 
        label = 'Data TES {} ASIC {}'.format(TESNum,asic), 
        color = 'tab:blue')
	plt.xlabel('Unix Epoch (s)')

	plt.legend(loc = 'upper left')

	plt.show()

	return

def plot_data_and_src(tod_time, tod_data, tod_data_filtered,
	calsrc_time, calsrc_data, **kwargs_plot):#ylim = [-5,5]):
	"""
	Plot calibration source, raw and filtered data
	"""

	plt.figure(figsize = (16,8))
	plt.plot(calsrc_time, (calsrc_data - np.mean(calsrc_data)) / np.std(calsrc_data), 
	         color = 'tab:orange', label = 'Calibration Source', alpha = 0.5)
	plt.plot(tod_time, (tod_data_filtered - np.mean(tod_data_filtered)) / np.std(tod_data_filtered), 
	         color = 'tab:green', label = 'Filtered Data', alpha = 0.5)
	plt.plot(tod_time, (tod_data - np.mean(tod_data)) / np.std(tod_data), 
	         label = 'Raw Data', color = 'tab:blue', alpha = 0.99)
	plt.xlabel('Unix Epoch (s)')
	
	#plt.xlim(kwargs_plot["xmin"], kwargs_plot["xmax"])
	#plt.ylim(kwargs_plot["ymin"], kwargs_plot["ymax"])
	plt.ylim(-5,5)
	plt.legend()

	return



def plot_spectra_comparisson(frequency_raw, spectra_raw, frequency_filtered, spectra_filtered,
	period, lowcut, highcut, notch, nharm = 10,
	TESNum = None, asic = None,
	xlim = [0.01, 90], 
	ylim = [1e1, 1e17]):
	"""
	This method compare raw spectra vs filtered spectra
	"""

	plt.rc('figure', figsize = (13,8))

	#xmin, xmax, ymin, ymax = 0.01, 90, 1e1, 1e17

	############ Power spectrum
	plt.plot(frequency_raw, f.gaussian_filter1d(spectra_raw, 1), 
	         label = 'Raw Data')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Power Spectrum')
	plt.xlim(xlim[0], xlim[1])
	plt.ylim(ylim[0], ylim[1])
	plt.title('TES {} ASIC {}'.format(TESNum, asic))

	for i in range(10):
	    plt.plot([1. / period * i, 1. / period * i], [ylim[0], ylim[1]],
	             'k--', alpha = 0.3)

	plt.plot([lowcut, lowcut], [ylim[0], ylim[1]], 'k')
	plt.plot([highcut, highcut], [ylim[0], ylim[1]], 'k')
	plt.legend()

	########## New Power spectrum
	plt.plot(frequency_filtered, f.gaussian_filter1d(spectra_filtered, 1), label = 'Filtered data')
	for i in range(nharm):
	    plt.plot([notch[0,0] * (i + 1), notch[0,0] * (i + 1)], 
	             [ylim[0], ylim[1]], 'm:')
	    
	plt.legend(loc = 'upper left')

	plt.tight_layout()

	plt.show()

	return