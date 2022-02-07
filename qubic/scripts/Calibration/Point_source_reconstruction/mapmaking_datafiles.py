"""$Id: mapmaking_datafiles.py
$auth: Martin Gamboa <mgamboa@fcaglp.unlp.edu.ar> & Jean-Christophe Hamilton
$created: Fri 5 Feb 2021

Inspired and using the methods created by Jean-Christophe 
		to do map-making with real data. 
 
This file aims to have the functions to do map making using data files

"""

import os
import sys 
from warnings import warn
import warnings
import pickle 

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt 
from matplotlib import *
from astropy.io import fits as pyfits
import glob
import time

from pysimulators import FitsArray

import qubic 
import qubic.selfcal_lib as sc
from qubicpack.utilities import Qubic_DataDir
import qubic.sb_fitting as sbfit
import jchinstrument as jcinst
import qubic.demodulation_lib as dl
import toolfit_hpmap as fh
import qubic.fibtools as ft
import qubic.SpectroImLib as si
from qubicpack.qubicfp import qubicfp

from qubicpack.pixel_translation import make_id_focalplane, plot_id_focalplane, tes2pix, tes2index

def read_calsource_data(qubic_fp, date = None, 
						keyword = None, datadir = None,
						datacal = None, verbose = False):

	"""
	This method read the calibration source data using qubicpack. The data was stored in a different way before 
	Nov 2019 and after Nov 2019 that's why you will find there are two ways to find and read the data.

	qubic_fp: 
			qubicfp class. Focal plane of QUBIC build using qubicpack
	date: 
			string. YYYY-MM-DD of the scan
	keyword:
			string. keyword used in the scan
	datadir: 
			string. Root directory where RAW TOD is placed (the one where dirs for each scan are).
	datacal:
			string. Root directory where calibration data is placed. This argument is only need it
			when you are reading data before Nov 2019 (arround 10th) and this data is usually saved in
			"calsource" directory. The format of the files are "calsource_YYYYMMDDTHHMMSS.fits"
	===============
	return:
			calsource_time and calsource_data. Time and data of the calibration source
	"""

	if not isinstance(date, str) or not isinstance(keyword, str) or not isinstance(datadir, str):
		raise ValueError("date, key or datadir is not {} class".format(str))
	if not isinstance(qubic_fp, qubicfp):
		raise ValueError("qubic_fp is not the right class: {}".format(qubicfp().__class__))

	try:
		os.path.isdir(datadir)
	except:
		raise ValueError("[path problem] datadir does not exist in your machine: {}".format(datadir))

	dirs = np.sort(glob.glob(datadir + keyword))

	if int(date.replace("-", "")) > 20191110:
		if verbose: print("Reading calibration source. Date: {}".format(date))
		calsource_time = qubic_fp.calsource()[0]
		calsource_data = qubic_fp.calsource()[1]
	elif int(date.replace("-", "")) < 20191110:
		if verbose: print("Reading calibration source. Date: {}".format(date))
		warnings.warn("The format of this kind of files has some tricks to read and plot, keep that in mind.",
						UserWarning, stacklevel=2)
		if datacal == None: 
			raise ValueError("[path problem] You have to provide a directory where the calsource data is.")
		try:
			os.path.isdir(datacal)
		except:
			raise ValueError("[path problem] datacal does not exist in your machine: {}".format(datacal))

		filesname = glob.glob(datacal + "*{}*.fits".format(date.replace("-", "")))
		calsource_time, calsource_data = [], []
		for eachfile in filesname:
			hdufile = pyfits.open(eachfile)
			hdu = qubic_fp.read_calsource_fits(hdufile[1])
			calsource_time.append(qubic_fp.hk['CALSOURCE']['timestamp'])
			calsource_data.append(qubic_fp.hk['CALSOURCE']['Value'])
	else:
		raise ValueError("The day argument is not in the correct format 'YYYY-MM-DD'.")

	return calsource_time, calsource_data

def pipe_demodulation(QubicFocPlane, 
					time_calsource, data_calsource,
					path_demod_data_save, 
					demodulate = False,
					demod_method = "demod_quad",
					remove_noise = True,
					lowcut = 0.5,
					highcut = 70,
					nharm = 10,
					verbose = False):

	"""
	This method computes the demodulation of the signal from the raw tod. It returns the amplitud
	and time demodulated using demodulation_lib.py

	====================
	Arguments:
		QubicFocPlane:
			qubicpack.qubicfp.qubicfp instantiation.
		time_calsource, data_calsource: 
			time and data from the calibration source files. 
		path_demod_data_save:
			path where to save or read the demodulated time and data. 
		demodulate:
			If True, it demodulates the signal. Otherwise, if False, it reads the demodulated data 
			from directory
		demod_method: 
			mthod to use to demodulate data (see options in demodulated_lib.demodulated_methods)
		remove_noise:
		lowcut:
		highcut:
		nharm:
	====================
	Return:
		demodulated_time:
			time demodulated
		demodulated_data:
			data demodulated
		"""
	thefreqmod = 1.

	period = 1./ thefreqmod
	notch = np.array([[1.724, 0.005, nharm]])
	fourier_cuts = [lowcut, highcut, notch]

	t0 = time.time()
	if demodulate: 
		if os.path.isfile(path_demod_data_save + "tod_data_demodulated_TESNum_120.fits"):
			print("In {} there EXISTS files so probably your data is already \
				demodulated,".format(path_demod_data_save))
			confirm_demodulate = input("Would you like to demodulate the data anyway? \
				(1 if True, 0 if False)")

			confirm_demodulate = bool(confirm_demodulate)
			
			if (confirm_demodulate == 1) or (confirm_demodulate == True): 
			
				warn("(re)Demodulation confirmed! ")
			
			else:
			
				sys.exit("Stopped!")

		try: 
			if not os.path.isdir(path_demod_data_save): 
				os.mkdir(path_demod_data_save) 
			else: 
				print("Saving demodulated data and time in existing directory: {}".format(
					path_demod_data_save))
				False 
		except: 
			raise ValueError("[path problem] I couldn't create the root directory \
				where save the demodulated data: {}".format(path_demod_data_save))
		# Do one TES to just pick the shape 
		tod_temp = QubicFocPlane.timeaxis(axistype = 'pps', asic = 1)
		temp_amps = dl.demodulate_methods([QubicFocPlane.timeaxis(
			axistype = 'pps', asic = 1), QubicFocPlane.timeline(TES = 1, asic = 1)], 
			1./period, 
			src_data_in = [tod_temp, np.interp(tod_temp, time_calsource, data_calsource)],
			method = demod_method, 
			remove_noise = remove_noise,
			fourier_cuts = fourier_cuts)[1]

		demodulated_amps = np.zeros((256, len(temp_amps)))

		for asic in [1,2]:
			tod_time = QubicFocPlane.timeaxis(axistype = 'pps', asic = asic)
			source = [tod_time, np.interp(tod_time, time_calsource, data_calsource)]
			for i in range(128):
				print('Mapmaking for Asic {} TES {}'.format(asic, i + 1))    
				tod_data = QubicFocPlane.timeline(TES = i + 1, asic = asic)

				print('- Demodulation')
				demodulated_time, demodulated_amps[i+128*(asic-1),:], errors_demod = dl.demodulate_methods(
																			[tod_time, tod_data],
																			1./period, 
																			src_data_in = source,
																			method = demod_method, 
																			remove_noise = remove_noise,
																			fourier_cuts = fourier_cuts)
		if verbose: print("Time spent in demodulate raw data: {:.2f} minutes".format((time.time()-t0)/60) )
		for i in range(256):
			FitsArray(demodulated_amps[i,:]).save(path_demod_data_save + \
											'tod_data_demodulated_TESNum_{}.fits'.format(i+1))    
		FitsArray(demodulated_time).save(path_demod_data_save + 'tod_time_demodulated.fits')
		
	elif not demodulate:
		print("Reading demodulated data")
		demodulated_amps = []
		for i in range(256):
			demodulated_amps.append(np.array(FitsArray(path_demod_data_save + \
												 'tod_data_demodulated_TESNum_{}.fits'.format(i+1))))
		demodulated_time = np.array(FitsArray(path_demod_data_save + 'tod_time_demodulated.fits'))
		if verbose: print("Time spent in read demodulated data: {:.2f} minutes".format((time.time()-t0)/60))

	return demodulated_time, demodulated_amps

def hall_pointing(az, el, angspeed_psi, maxpsi,
				 date_obs=None, latitude=None, longitude=None,fix_azimuth=None,random_hwp=True):
	"""
	This method will reproduce the pointing that is used in the hall to take the data. Will start from bottom
	left and will go up at fixed elevation.
	"""

	nsamples = len(az)*len(el)
	pp = qubic.QubicSampling(nsamples,date_obs=date_obs, period=0.1, latitude=latitude,longitude=longitude)
	
	#Comented because we do not go and back in simulations.. 
	#mult_el = []
	#for eachEl in el:
	#    mult_el.append(np.tile(eachEl, 2*len(az)))
	# Azimuth go and back and same elevation. 
	#az_back = az[::-1]
	#az = list(az)
	#az.extend(az_back)
	#mult_az = np.tile(az, len(el))
	#print(i,np.asarray(mult_el).ravel().shape)
	#pp.elevation = np.asarray(mult_el).ravel()
	#pp.azimuth = np.asarray(mult_az).ravel()
	
	mult_el = []
	for eachEl in el:
		mult_el.extend(np.tile(eachEl, len(az)))
	mult_az = []
	
	mult_az.append(np.tile(az, len(el)))
	
	
	pp.elevation = np.asarray(mult_el)#az2d.ravel()
	pp.azimuth = np.asarray(mult_az[0])#el2d.ravel()
	
	### scan psi as well,
	pitch = pp.time * angspeed_psi
	pitch = pitch % (4 * maxpsi)
	mask = pitch > (2 * maxpsi)
	pitch[mask] = -pitch[mask] + 4 * maxpsi
	pitch -= maxpsi
	
	pp.pitch = pitch
	
	if random_hwp:
		pp.angle_hwp = np.random.random_integers(0, 7, nsamples) * 11.25
		
	if fix_azimuth['apply']:
		pp.fix_az=True
		if fix_azimuth['fix_hwp']:
			pp.angle_hwp=pp.pitch*0+ 11.25
		if fix_azimuth['fix_pitch']:
			pp.pitch= 0
	else:
		pp.fix_az=False

	return pp







def select_det(q,idqp):
	"""
	Returns a sub-instrument with detectors index given by idqp. These indices are to be understood
	in the qubicpack() numbering starting from 1
	"""
	if len(q)==1:
		x, y, FP_index, index_q = sc.get_TES_Instru_coords(q, frame='ONAFP', verbose=False)
		q.detector = q.detector[index_q[np.array(idqp)-1]]
	else:
		x, y, FP_index, index_q = sc.get_TES_Instru_coords(q[0], frame='ONAFP', verbose=False)
		for i in range(len(q)):
			q[i].detector = q[i].detector[index_q[np.array(idqp)-1]]
	return(q)

def readmaps(directory, tes, asic, az, el, p, proj_name, nside = None,  
		azmin = None, azmax = None, remove = None, verbose = False):
	"""
	directory: path to flat maps (get_flatmap method used to read maps)
	npix: array of number of pixels
	el, az: coordinates from data
	proj_name: projection you want. It could be 'flat' or 'healpix'
	nside: healpix parameter needed
	p: pointing"""
	if proj_name == 'flat':
		filemaps = np.zeros ( (len(tes), len(el), len(az)) )
		for i in range(len(tes)):
			filename =directory + 'flat_ASIC{}TES{}.fits'.format(asic[i], tes[i])
			if verbose: print("[FlatMapsFile]: Reading {} file".format(filename))
			#filemaps[i], _, _ = sbfit.get_flatmap(npix[i], directory, 
			#	azmin=azmin, azmax=azmax, remove=remove)
			filemaps[i] = FitsArray(filename)

	elif proj_name == 'healpix':
		warn("HEALPix projections: At the moment, HEALPix projection comes from the readout of flat maps \
			In future this should be change and read directly from TOD")

		if nside == None:
			nside = hp.get_nside(FitsArray(directory + 'hp_ASIC{}TES{}.fits'.format(asic[0], tes[0])))			

		filemaps = np.zeros ( (len(tes), 12 * nside ** 2) )
		for i in range(len(tes)):
			filename = directory + 'hp_ASIC{}TES{}.fits'.format(asic[i], tes[i])
			if verbose: print("[FlatMapsFile]: Reading {} file".format(filename))
			#auxmap, _, _ = sbfit.get_flatmap(npix[i], directory, 
			#		azmin=azmin, azmax=azmax, remove=remove)

			filemaps[i] = FitsArray(filename)
	else: 
		raise ValueError("You didn't specified the projection you want: flat or healpix.")

	return filemaps

def _plot_onemap(filemaps, az, el, phth, proj_name, makerotation = False, angs = None):
	indxmap = 0
	if proj_name == "flat":
		plt.title("Checking coord. of peaks and display of map.")
		plt.imshow(filemaps[indxmap, :, :], extent = [np.max(az), np.min(az) ,
								   np.min(el), np.max(el)]	)
		plt.plot(np.degrees(phth[0][indxmap, :]), np.degrees(phth[1][indxmap, :]), color = 'r', marker = '.', linestyle = '')
	
	elif proj_name == "healpix":
		hp.gnomview(filemaps[indxmap], min = 0, cbar= False, #rot = (azcen_fov, elcen_fov),
				title = 'Checking coord. of peaks and display of map', 
				reso = 10, rot = (0,50))
		hp.projscatter(phth[1][indxmap, :], phth[0][indxmap, :], color = 'r', marker = '.')
		hp.graticule(verbose = 0, alpha = 0.4)
	
	plt.show()

	return

def get_data_Mrefsyst(detnums, filemaps, az, el, fitted_directory, fittedpeakfile, proj_name,
	resample = None, newsize = None, azmin = None, azmax =None, remove = None,
	sbfitmodel = None, refit = False, verbose = False):

	"""
	Compute theta, phi for all TES in the measured reference system
	"""

	thecos = np.cos(np.radians(np.mean(el)))
	if refit:
		if verbose: print('We refit the peak locations')
		### We call the fitting function for the Synthesized beam
		xypeaks = []
		for i in range(len(detnums)):
			if fitted_directory is None:
		
				if sbfitmodel is None:
					sbfitmodel = sbfit.SbModelIndepPeaks(nrings=2, common_fwhm=True, 
													 no_xy_shift=False, distortion=False)
				if verbose: 
					print('Using Fit Model {} for TES #{}'.format(sbfitmodel.name,detnums[i]))
				
				figure()
				fit, thexypeaks = sbfit.fit_sb(filemaps, az, el, sbfitmodel, resample=resample, newsize=newsize,
											   verbose=verbose, doplot=True, 
											   extra_title='TES #{}'.format(detnums[i]))
				print('FITING')
				show()
			else:
				filemaps, az, el, fitmap, thexypeaks = sbfit.get_flatmap(detnums[i], directory, 
																		azmin = azmin, azmax=azmax, remove=remove,
																	   fitted_directory=fitted_directory)
			xypeaks.append(thexypeaks)
			
		### Convert to measurement coordinate system
		xypeaks = np.array(xypeaks)
		allthetas_M = np.radians(90 - (xypeaks[:,1,:] - elcen_fov))
		allphis_M = np.radians( -xypeaks[:,0,:])#*thecos)
		allvals_M = xypeaks[:,2,:]
	else:           
		if verbose: 
			print('No refitting of the peak locations')
			print("[FitFiles]: Reading {} file".format(fittedpeakfile))
		### We just read them from the old peak file
		peaks = np.array(FitsArray(fittedpeakfile))
		#Thph indexes
		if proj_name == "flat":
			thphidx = [0,1] 
			peaks[:,1,:] = peaks[:,1,:] / thecos
		elif proj_name == "healpix":
			thphidx = [0,1]
			peaks[:,1,:] = peaks[:,1,:] 
		### An put them in the expected format. Save TES of interest
		mypeaks = peaks[np.array(detnums)-1,:,:]
		#Peaks in degrees
		allphis_M = mypeaks[:,1,:]
		allthetas_M = mypeaks[:,0,:]
		allvals_M = mypeaks[:,2,:]
	return allphis_M, allthetas_M, allvals_M 

def convert_M2Q(detnums, allphis_M, allthetas_M, allvals_M, elcen_fov, solid_angle,
				nu, horn, solid_angle_s, angs = None, verbose = False):

	if angs is None:
		angs = np.radians(np.array([0, 90 - elcen_fov, 0]))
	allthetas_Q = np.zeros_like(allthetas_M)
	allphis_Q = np.zeros_like(allthetas_M)
	allvals_Q = np.zeros_like(allthetas_M)
	numpeak = np.zeros(len(detnums), dtype=int)

	for idet in range(len(detnums)):
		allthetas_Q[idet,:], allphis_Q[idet,:] = sbfit.rotate_q2m(allthetas_M[idet,:], 
																  allphis_M[idet,:], 
																  angs=angs, inverse=True)
		allvals_Q[idet,:] = allvals_M[idet,:]/np.max(allvals_M[idet,:])*solid_angle * (150e9 / nu)**2 / solid_angle_s * len(horn)
		
		if verbose: 
			print('For TES {}'.format(idet))
			print('Thetas: {}'.format(np.degrees(allthetas_Q[idet,:])))
			print('Phis: {}'.format(np.degrees(allphis_Q[idet,:])))
		
		#### Louise mentions a pi rotation of the measured SB w.r.t. simulations => we apply it here
		allphis_Q[idet,:] += np.pi

	return allphis_Q, allthetas_Q, allvals_Q, numpeak

def _plot_grfRF(detnums, xgrf, ygrf, qcut, numpeak):
	### get TES position in the GRF
	for idet in range(len(detnums)):
		plt.figure(figsize = (16,8))
		plt.clf()
		ax = plt.subplot(131)
		ax.set_aspect('equal')
		plt.plot(xgrf, ygrf, 'k+')
		plt.xlim(-0.053, 0)
		plt.ylim(0, 0.053)
		
		position = np.ravel(qcut[0].detector[idet].center)
		if verbose: print(position)
		plt.plot(position[0], position[1], 'ro', label='TES#{}'.format(detnums[idet]))
		plt.legend()
		
		
		position = -position / np.sqrt(np.sum(position**2))
		theta_center = np.arcsin(np.sqrt(position[0]**2 + position[1]**2))
		phi_center = np.arctan2(position[1], position[0])
		if verbose: 
			print('==== Position ==')
			print(position)
			print(theta_center, phi_center)
			print('=================')
	
		ax = plt.subplot(132, projection='polar')
		plt.title('Initial')
		rav_phQ = np.ravel(allphis_Q[idet,:])
		rav_thQ = np.ravel(allthetas_Q[idet,:])
		plt.scatter(rav_phQ, rav_thQ, s=np.ravel(allvals_Q[idet,:])/np.max(allvals_Q[idet,:])*300)
		for k in range(len(rav_phQ)):
			plt.text(rav_phQ[k], rav_thQ[k], k)
		ax.set_rmax(0.5)
		plt.plot(phi_center, theta_center,'r+', ms=10, markeredgewidth=3, label = 'Th. Line of sight')
	
		## Now we identify the nearest peak to the theoretical Line Of Sight
		angdist = np.zeros(len(rav_phQ))
		for k in range(len(rav_phQ)):
			angdist[k] = sbfit.ang_dist([theta_center, phi_center], [rav_thQ[k], rav_phQ[k]])
			if verbose: print(k,np.degrees(angdist[k]))
		idxmin = np.argmin(angdist)

		numpeak[idet]=idxmin
		throt = allthetas_Q[idet,numpeak[idet]]
		phrot = allphis_Q[idet,numpeak[idet]]
		## Rotate around the position predicted from the TES location
		#throt = theta_center
		#phrot = phi_center
		if verbose: 
			print('+++++++++++')
			print(throt, phrot)
			print('+++++++++++')

		plt.plot(phrot, throt, 'gx', ms=15, markeredgewidth=2, label='Measured Line of sight')
		ax.set_rmax(0.5)
		plt.legend()

		myangs = np.array([phrot,throt, phrot])
		newth, newph = sbfit.rotate_q2m(allthetas_Q[idet,:], allphis_Q[idet,:], angs=myangs, inverse=True)
	
		ax = plt.subplot(133, projection='polar')
		plt.scatter(np.ravel(allphis_Q[idet,:]), np.ravel(allthetas_Q[idet,:]), s=np.ravel(allvals_Q[idet,:])/np.max(allvals_Q[idet,:])*300)
		plt.title('Back')
		for k in range(len(qcut)):
			factor = 150e9/qcut[k].filter.nu
			newthfinal, newphfinal = sbfit.rotate_q2m(newth*factor, newph, angs=myangs, inverse=False)
			#plt.scatter(np.ravel(newphfinal), np.ravel(newthfinal), s=np.ravel(allvals_Q)/np.max(allvals_Q)*300)
		ax.set_rmax(0.5)
		plt.legend()
		plt.show()

def do_some_dets(detnums, d, p, directory, fittedpeakfile, az, el, proj_name, custom=False, 
				 q = None, nside=None, tol=5e-3, refit=False, resample=False, newsize=70, 
				 doplot=True, verbose=True, sbfitmodel=None, angs=None, usepeaks=None,
				 azmin=None, azmax=None, remove=None, fitted_directory=None, weighted=False,
				nf_sub_rec=1, lowcut=1e-3, highcut=0.3):


	if nside is not None:
		d['nside'] = nside
	s = qubic.QubicScene(d)
	
	if q == None:
		q = qubic.QubicMultibandInstrument(d)
	
	if len(q) == 1:
		xgrf, ygrf, FP_index, index_q = sc.get_TES_Instru_coords(q, frame='GRF', verbose=False)
	else:
		xgrf, ygrf, FP_index, index_q = sc.get_TES_Instru_coords(q[0], frame='GRF', verbose=False)

	# Create TES, index and ASIC numbers assuming detnums is continuos TES number (1-248).
	tes, asic = np.zeros((len(detnums), ), dtype = int), np.zeros((len(detnums), ), dtype = int)
	qpix = np.zeros((len(detnums), ), dtype = int)

	for j, npix in enumerate(detnums):
		tes[j], asic[j] = (npix, 1) if (npix < 128) else (npix - 128, 2)
		qpix[j] = tes2pix(tes[j], asic[j]) - 1
		if verbose: print("DETNUM{} TES{} ASIC{} QPIX{}".format(npix,tes[j],asic[j],qpix[j]))

	#Center of FOV
	azcen_fov = np.mean(az)
	elcen_fov = np.mean(el)

	#Directory where the maps are:
	mapsdir = directory
	#File where the fitted peaks are:
	peaksdir= fittedpeakfile

	if not custom:
		if verbose:
			print('')
			print('Normal Reconstruction')
		qcut = select_det(qubic.QubicMultibandInstrument(d),detnums)
		#qcut = select_det(qubic.QubicMultibandInstrument(d),[145])
	else:
		if verbose:
			print('')
			print('Custom Reconstruction')
		### Refit or not the locations of the peaks 
		### from the synthesized beam images      
		### First instantiate a jchinstrument (modified from instrument 
		### to be able to read peaks from a file)

		print("Generating jchinstrument")
		qcut = select_det(jcinst.QubicMultibandInstrument(d, peakfile = fittedpeakfile, 
			tes = tes, asic = asic), qpix)
		print("LEN(qcut) and detector", len(qcut), np.shape(qcut[0].detector.center))
		print("Generating jchinstrument"[::-1])
		
		### In the present case, we use the peak measurements at 150 GHz
		### So we assume its index is len(qcut)//2
		id150 = len(qcut)//2
		nu = qcut[id150].filter.nu
		synthbeam = qcut[id150].synthbeam
		horn = getattr(qcut[id150], 'horn', None)
		primary_beam = getattr(qcut[id150], 'primary_beam', None)
		# Cosine projection with elevation center of the FOV considering symetric scan in azimuth for each elevation step
		thecos = np.cos(np.radians(elcen_fov))

		#Read map (flat or healpy) for each detector
		print("TES ASIC", tes, asic)
		filemaps = readmaps(directory, tes, asic, az, el, p, 
							proj_name = proj_name, nside = d['nside'], verbose = verbose)

		# Compute measured coordenates
		allphis_M, allthetas_M, allvals_M = get_data_Mrefsyst(detnums, filemaps, az, el, fitted_directory, fittedpeakfile, proj_name,
										resample = resample, newsize = newsize, azmin = azmin, azmax = azmax, remove = remove,
										sbfitmodel = sbfitmodel, refit = refit, verbose = verbose)
		if doplot: _plot_onemap(filemaps,az,el,[allphis_M, allthetas_M], proj_name, makerotation = False)

		### Now we want to perform the rotation to go to boresight 
		### reference frame (used internally by QubicSoft)
		if verbose: 
			print("Solid angles synthbeam = {:.3e}, QubicScene {:.3e}".format(synthbeam.peak150.solid_angle, s.solid_angle))

		allphis_Q, allthetas_Q, allvals_Q, numpeak = convert_M2Q(detnums, allphis_M, allthetas_M, allvals_M, 
														elcen_fov, solid_angle = synthbeam.peak150.solid_angle,
														nu = nu, horn = horn, solid_angle_s = s.solid_angle,
														angs = angs, verbose = verbose)
		print("allphis_Q , allvals_Q shapes", np.shape(allphis_Q), np.shape(allvals_Q))
		
		if doplot:
			plt.title("xy coord. of peaks in M and Q ref system")
			plt.plot(np.cos(allphis_M[0])* np.sin(allthetas_M[0]), 
				np.sin(allphis_M[0])* np.sin(allthetas_M[0]), 'r+', ms = 10,
				label = "M ref syst")
			plt.plot(np.cos(allphis_Q[0])*np.sin(allthetas_Q[0]), 
				np.sin(allphis_Q[0])*np.sin(allthetas_Q[0]), 'b+', ms = 10,
				label = "Q ref syst")
			plt.legend()
			plt.show()

		if doplot:
			# Plot peaks in measured reference frame. 
			_plot_grfRF(detnums, xgrf, ygrf, qcut, numpeak)

		for idet in range(len(detnums)):
			position = np.ravel(qcut[0].detector[idet].center)
			position = - position / np.sqrt(np.sum(position ** 2))
			theta_center = np.arcsin(np.sqrt(position[0] ** 2 + position[1] ** 2 ))
			phi_center = np.arctan2(position[1], position[0])
			rav_thQ = np.ravel(allthetas_Q[idet,:])
			rav_phQ = np.ravel(allphis_Q[idet,:])
			## Now we identify the nearest peak to the theoretical Line Of Sight
			angdist = np.zeros(len(rav_phQ))
			for k in range(len(rav_phQ)):
				angdist[k] = sbfit.ang_dist([theta_center, phi_center], [rav_thQ[k], rav_phQ[k]])
				print(k,np.degrees(angdist[k]))
			idxmin = np.argmin(angdist)
			numpeak[idet] = idxmin


		### We nowwrite the temporary file that contains the peaks locations to be used
		if usepeaks is None:
			peaknums = np.arange(9)
		else:
			peaknums = usepeaks
		data = [allthetas_Q[:,peaknums], allphis_Q[:,peaknums]-np.pi, allvals_Q[:,peaknums], numpeak]
		file = open(os.environ['QUBIC_PEAKS']+'peaks.pk', 'wb')
		pickle.dump(data, file)
		file.close()

	### Make the TODs from the measured synthesized beams
	realTOD = np.zeros((len(detnums),len(p)))
	sigmaTOD = np.zeros(len(detnums))
	if weighted:
		sumweight = 0.
	
	allimg = []

	for i in range(len(detnums)):
		if proj_name == "flat":
			filename = directory + 'flat_ASIC{}TES{}.fits'.format(asic[i], tes[i])
			img = FitsArray(filename)
		elif proj_name == "healpix":
			filename = directory + '../../Flat/synth_beam/flat_ASIC{}TES{}.fits'.format(asic[i], tes[i])
			img = FitsArray(filename)
		allimg.append(img)
		fact = 1 #5e-28
		realTOD[i,:] = np.ravel(img) * fact


		if weighted:   ## Not to be used - old test...
			realTOD[i,:] *= 1./ss**2
			sumweight += 1./ss**2
	print("All img", np.shape(allimg))
	### new code multiband
	plt.figure()
	for i in range(len(detnums)):
		plt.plot(realTOD[i,:], label='TES#{0:}'.format(detnums[i]))
	plt.legend()
	plt.xlabel('Samples')
	plt.ylabel('TOD')
	plt.show()
	
	plt.figure()
	for i in range(len(detnums)):
		spectrum_f, freq_f = ft.power_spectrum(np.arange(len(realTOD[i,:])), realTOD[i,:])
		pl = plt.plot(freq_f, spectrum_f, label='TES#{0:}'.format(detnums[i]), alpha=0.5)
		plt.xscale('log')
		plt.yscale('log')
	plt.legend()
	plt.xlabel('Fourier mode')
	plt.ylabel('Power Spectrum')
	plt.title('Before Filtering')
	plt.show()
	
	for i in range(len(detnums)):
		realTOD[i,:] = ft.filter_data(np.arange(len(realTOD[i,:])), realTOD[i,:], lowcut, highcut)
		mm,ss = ft.meancut(realTOD[i,:],3)
		sigmaTOD[i] = ss 
	if doplot:
		for i in range(len(detnums)):
			plt.figure()
			plt.subplot(1,2,1)
			plt.imshow(allimg[i] * fact, vmin=mm-3*ss, vmax=mm+3*ss,
				   extent=[np.max(az)*thecos, np.min(az)*thecos, np.min(el), np.max(el)], aspect='equal')
			plt.colorbar()
			plt.title('Init - TOD {0:} RMS={1:5.2g}'.format(detnums[i],sigmaTOD[i]))
			plt.subplot(1,2,2)
			plt.imshow(np.reshape(realTOD[i,:], np.shape(img)), vmin=mm-3*ss, vmax=mm+3*ss,
				   extent=[np.max(az)*thecos, np.min(az)*thecos, np.min(el), np.max(el)], aspect='equal')
			plt.colorbar()
			plt.title('Filtered - TOD {0:} RMS={1:5.2g}'.format(detnums[i],sigmaTOD[i]))

	
	plt.figure()
	for i in range(len(detnums)):
		spectrum_f, freq_f = ft.power_spectrum(np.arange(len(realTOD[i,:])), realTOD[i,:])
		pl = plt.plot(freq_f, spectrum_f, label='TES#{0:} Var*2pi={1:5.2g}'.format(detnums[i],sigmaTOD[i]**2*2*np.pi), alpha=0.5)
		plt.plot(freq_f, freq_f*0+sigmaTOD[i]**2*2*np.pi, color=pl[0].get_color())
		plt.xscale('log')
		plt.yscale('log')
	plt.ylim(np.min(sigmaTOD**2*2*np.pi), np.max(sigmaTOD**2*2*np.pi)*1e9)
	plt.legend()
	plt.xlabel('Fourier mode')
	plt.ylabel('Power Spectrum')
	plt.title('After Filtering')
	if lowcut:
		plt.axvline(x=lowcut, color='k')
	if highcut:
		plt.axvline(x=highcut, color='k')    
	plt.show()

	plt.figure()
	plt.clf()
	print('%%%%%%%%%%%%%%%%%%%%%%')
	ax=plt.subplot(111, projection='polar')
	print("len qcut", len(qcut[0].detector.center))
	print("=?===============================")
	print(np.shape(realTOD), np.shape(p), nf_sub_rec, np.shape(qcut))
	print("=?===============================")
	
	maps_recon, cov, nus, nus_edge = si.reconstruct_maps(realTOD, d, p,
														nf_sub_rec, x0 = None, instrument = qcut, #verbose=True,
														forced_tes_sigma = sigmaTOD)
	ax.set_rmax(0.5)
	#legend(fontsize=8)
	if weighted:
		maps_recon /= sumweight/len(detnums)
	return maps_recon, qcut, np.mean(cov, axis=0), nus, nus_edge


