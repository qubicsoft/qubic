import os
import sys
import glob

# Specific science modules
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from importlib import reload
import time
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

# Specific qubic modules
from astropy.io import fits
import qubic
from pysimulators import FitsArray
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic.polyacquisition import compute_freq
from qubic import ReadMC as rmc
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import scipy.constants
from qubic import mcmc
import ForegroundsSED as fsed

plt.rc('text',usetex=False)
plt.rc('font', size=16)

import qubic.AnalysisMC as amc

savefigs = True
nreals = 43
nside_new = 32
from_coords = True
if from_coords:
	DeltaThetaQ, DeltaPhiQ = 10, -6
	DeltaThetaG, DeltaPhiG = -8, 5
else: 
	pixQ_ud = 10759
	pixG_ud = 7235


print("## ===== PREPARE DICTIONARIES CONSIDERING 2 REGIONS + 2 BANDS =====")
# Dictionary saved during the simulation
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
#dictfilename = global_dir + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config)
dictfilename = global_dir + '/dicts/spectroimaging_article.dict'
d150Q = qubic.qubicdict.qubicDict()
d150Q.read_from_file(dictfilename)
d150Q['nf_recon'] = 5
d150Q['nf_sub'] = 5 
d150Q['photon_noise']=True
d150Q['RA_center'] = 0.0
d150Q['DEC_center'] = -57.0
d150Q['effective_duration'] = 2
d150Q['npointings']=10000
d150Q['effective_duration'] = 3

#150 galactic center patch (thereafter GC patch)
d150G = d150Q.copy()
d150G['effective_duration'] = 1

#
# 220 Qubic patch
d220Q = d150Q.copy()
d220Q['filter_nu'] = 220e9
# 220 GC patch
d220G = d220Q.copy()
d150G['effective_duration'] = 1

# Qubic patch - galactic coordinates
centerQ = qubic.equ2gal(d150Q['RA_center'], d150Q['DEC_center'])
# Galactic center patch - galactic coordinates 
centerG = np.array([0,0])
d150G['RA_center'], d150G['DEC_center'] = qubic.gal2equ(centerG[0], centerG[1])
d220G['RA_center'], d220G['DEC_center'] = d150G['RA_center'], d150G['DEC_center']

centers = [centerQ, centerQ, centerG, centerG]
dictionaries = [d150Q, d220Q, d150G, d220G]

regions = ['Qubic_field', 'GalCen_field']
bands = ['150', '220']

nf_recon = d150Q['nf_recon']

print(" =========== DONE ======== ")

print("# Read coverage maps")
coveragesmaps = fsed.coverage(dictionaries, regions, bands)

print("# Read FullMap (noise+foreground with nside = 256)")
maps = FitsArray("DataSED/SED_FullMaps_nside{}_nreals{}.fits".format(nside_new,nreals))

print("# Read sky (dust + synch) down graded to nside {}".format(nside_new))
fgr_map_ud = FitsArray("DataSED/SED_ForegMaps_nside{}_nreals{}.fits".format(nside_new,nreals))

print("# Read noise down graded to nside {}".format(nside_new))
noise_ud_i = FitsArray("DataSED/SED_NoiseMaps_nside{}_nreals{}.fits".format(nside_new,nreals))

print("# Read covariance matrix")
bandreg = ["FI150Q", "FI220Q", "FI150G", "FI220G"]
Cp_prime = []
for j, idict in enumerate(bandreg):
	Cp_prime.append(FitsArray("DataSED/SED_CovarMatrix_nside{}_nreals{}_{}.fits".format(nside_new,nreals, bandreg[j])))
#Cp_prime = FitsArray("SED_CovarMatrix_nside{}_nreals{}.fits".format(nside_new,nreals))

print("# Compute mask for coverage map and compute downgraded coverage map")
_, covmask = fsed._mask_maps(maps, coveragesmaps, nf_recon)
cov_ud = hp.ud_grade(covmask, nside_new)

print("# Compute downgraded maps (foreground + noise) using dgraded noise and foreground maps")
maps_ud_i = np.zeros((len(dictionaries), nreals, nf_recon, 12 * nside_new ** 2, 3))
for idict in range(len(dictionaries)):
	for iNU in range(nf_recon):
		for ireal in range(nreals):
			maps_ud_i[idict, ireal, iNU, ...] = noise_ud_i[idict, ireal, iNU, ...] + fgr_map_ud[idict, iNU]
			
maps_ud = np.mean(maps_ud_i, axis = 1)
for idict in range(len(dictionaries)):
		for iNU in range(nf_recon):
			maps_ud[idict, iNU,~cov_ud[idict],:] = hp.UNSEEN

print("# Done.")

PixPix = lambda p: hp.ang2pix(nside_new, hp.pix2ang(dictionaries[0]['nside'], p)[0], 
							  hp.pix2ang(dictionaries[0]['nside'], p)[1] )


if from_coords:
	print("# Choosing pixels from sky coords")
	pixQ = [hp.ang2pix(dictionaries[0]['nside'], 
				   np.pi / 2 - np.deg2rad(centers[0][1] + DeltaThetaQ), np.deg2rad(centers[0][0] + DeltaPhiQ ) ), ]
	pixG = [hp.ang2pix(dictionaries[0]['nside'], 
				   np.pi / 2 - np.deg2rad(centers[2][1] + DeltaThetaG), np.deg2rad(centers[2][0] - DeltaPhiG  ) ), ]

	pixQ_ud = PixPix(pixQ[0])
	pixG_ud = PixPix(pixG[0])

else:
	print("# Choosing pixels from pixel number")

# Where the sky pixel is in the reduce format (pixels seen array and not full map)
pixQ_red = np.where(np.where(cov_ud[0] == True)[0] == pixQ_ud)[0][0]
pixG_red = np.where(np.where(cov_ud[2] == True)[0] == pixG_ud)[0][0]
print("# Done. Pixel in QUBIC FOV: {} \n \t Pixel in galactic center FOV: {}".format(pixQ_ud, pixG_ud))
print("# Reduce array. Pixel in QUBIC FOV: {} \n \t Pixel in galactic center FOV: {}".format(pixQ_red, pixG_red))

plt.figure(figsize = (10,4))
hp.gnomview(maps_ud[2,-1,:,0], reso = 15,#hold = True, 
			notext = False, title = 'G patch ', sub = (121),
			max = 0.4*np.max(maps_ud[2,-1,:,0]), 
			unit = r'$\mu$K',
			rot = centers[2])
hp.projscatter(hp.pix2ang(nside_new, pixG_ud), marker = '*', color = 'r', s = 200)
hp.gnomview(maps_ud[1,-1,:,0], reso = 15, title = 'Q patch ',
			unit = r'$\mu$K', sub = (122),
			rot = centerQ)
hp.projscatter(hp.pix2ang(nside_new, pixQ_ud), marker = '*', color = 'r', s = 200)
hp.graticule(dpar = 10, dmer = 20, alpha = 0.6, verbose = False)
plt.show()

print("# Preparing for run MCMC ")
_, nus150, nus_out150, _, _, _ = qubic.compute_freq(dictionaries[0]['filter_nu'] / 1e9,  
							dictionaries[0]['nf_recon'],
							dictionaries[0]['filter_relative_bandwidth'])
_, nus220, nus_out220, _, _, _ = qubic.compute_freq(dictionaries[1]['filter_nu'] / 1e9,  
							dictionaries[1]['nf_recon'],
							dictionaries[1]['filter_relative_bandwidth'])
nus_out = [nus_out150, nus_out220, nus_out150, nus_out220]
pixs_ud = [pixQ_ud, pixQ_ud, pixG_ud, pixG_ud]
pixs_red = [pixQ_red, pixQ_red, pixG_red, pixG_red]
nus_edge = [nus150, nus220, nus150, nus220]

study = "dust+synch"
if study == "dust":
	FuncModel = fsed.ThermDust_Planck353
	p0 = np.array([1e3,3])
elif study == "synch":
	FuncModel = fsed.Synchrotron_storja
	#p0 = np.array([1e1,20,-3]) #Planch
	p0 = np.array([1e1,-3])
elif study == "dust+synch":
	FuncModel = fsed.DustSynch_model
	p0 = np.array([1e8, 3, 1e6, 3])
Chi2Model = None#"Chi2Implement"
initP0_fit = np.array([1e5, 3, 1e3, 3])

print("# = = = = RUNNING MCMC for {} study = = = =".format(study))
print("\t with initial guess (MCMC sim): {}".format(p0))
print("\t with initial guess (fit SED): {}".format(initP0_fit))
print("\t with chi2 model {} in fibtools".format("MyChi2" if Chi2Model == None else Chi2Model, "fibtools"))

Imvals, Isvals, xarr, flat_samples = fsed.foregrounds_run_mcmc(dictionaries, fgr_map_ud, Cp_prime, FuncModel,
												nus_out, nus_edge, pixs_ud, 
												pixs_red = pixs_red, chi2=Chi2Model, 
												samples = 5000, verbose = False, initP0 = p0)
print(" = = = = = = = = = = = Done.")

print("# Fitting SED ")
xSED = [nus_out150, nus_out220, nus_out150, nus_out220]

if study == "dust":
	FuncPoint = fsed.ThermDust_Planck353_pointer
	#FuncPoint = fsed.ThermDust_Planck545_pointer
		
elif study == "synch":
	FuncPoint = fsed.Synchrotron_storja_pointer
	#FuncPoint = fsed.Synchrotron_Planck_pointer

elif study == "dust+synch":
	FuncPoint = fsed.DustSynch_model_pointer

ySED_fit, Pmean, Perr = fsed.make_fit_SED(xSED, xarr, Imvals, Isvals,
										  FuncPoint, fgr_map_ud, pixs_ud, nf_recon, 
										  initP0 = initP0_fit, 
										  maxfev = 15000)
print("# = = = = Done.")

print("# _______ Plotting SED Intensity_______ ")
intensity_out_plot = 'TEST4JCH_{}_nrec{}_nside{}_pixQ{}_pixG{}_Intensity_grat'.format(
								FuncModel.__name__, d150Q['nf_recon'],nside_new, pixQ_ud, pixG_ud)

print(" \t \t in {} [.svg, .png and .pdf]".format(intensity_out_plot))

RESO = 15
capsize = 3
plt.rc('font', size = 14)

fig,ax = plt.subplots(nrows = 1, ncols = 4,figsize = (19,4.5), gridspec_kw = {'wspace': 0.4})
ax = ax.ravel()
plt.subplots_adjust(wspace = 0.1)
# Plotting
p1, = ax[0].plot(nus_out150, fgr_map_ud[2, :, pixs_ud[2], 0], 'ro', label = 'Input sky')
p2, = ax[0].plot(nus_out220, fgr_map_ud[3, :, pixs_ud[2], 0], 'bo')
e1 = ax[0].fill_between(xarr[2,:], y1 = ySED_fit[2,:,0] - Isvals[2, :, 0], 
								y2 = ySED_fit[2, :, 0] + Isvals[2, :, 0], 
				 color = 'r', alpha = 0.3, label = '68% C.L.')
e2 = ax[0].fill_between(xarr[3, :], y1 = ySED_fit[3, :, 0] - Isvals[3, :, 0], 
						y2 = ySED_fit[3, :, 0] + Isvals[3, :, 0], 
				   color = 'b', alpha = 0.3)

ax[2].plot(nus_out150, fgr_map_ud[0, :, pixs_ud[0], 0], 'ro')
ax[2].plot(nus_out220, fgr_map_ud[1, :, pixs_ud[0], 0], 'bo')
ax[2].fill_between(xarr[0, :], y1 = ySED_fit[0, :, 0] - Isvals[0, :, 0], 
				   y2 = ySED_fit[0, :, 0] + Isvals[0, :, 0], 
				   color = 'r', alpha = 0.3)
ax[2].fill_between(xarr[1, :], y1 = ySED_fit[1, :, 0] - Isvals[1, :, 0], 
				   y2 = ySED_fit[1, :, 0] + Isvals[1, :, 0], 
				   color = 'b', alpha = 0.3)

# Settings
greyscale = 0.1
ax[2].axvspan(nus150[-1], nus220[0],color='k',alpha = greyscale)
ax[0].axvspan(nus150[-1], nus220[0],color='k',alpha = greyscale)
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
xlim2 = ax[2].get_xlim()
ylim2 = ax[2].get_ylim()
ax[0].axvspan(xlim[0], nus150[0], color = 'k', alpha = greyscale)
ax[0].axvspan(nus220[-1], xlim[-1], color = 'k', alpha = greyscale)

ax[2].axvspan(xlim2[0], nus150[0], color = 'k', alpha = greyscale)
ax[2].axvspan(nus220[-1], xlim2[-1], color = 'k', alpha = greyscale)

ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)
ax[0].text(xlim[0]+(xlim[1]-xlim[0])*0.1, ylim[-1]*0.8, '150 GHz \n band', fontsize = 10)
ax[0].text(xlim[0]+(xlim[1]-xlim[0])*0.6, ylim[-1]*0.8, '220 GHz \n band', fontsize = 10)
ax[2].set_xlim(xlim2)
ax[2].set_ylim(ylim2)
ax[2].text(xlim2[0]+(xlim2[1]-xlim2[0])*0.1, ylim2[-1]*0.8, '150 GHz \n band', fontsize = 10)
ax[2].text(xlim2[0]+(xlim2[1]-xlim2[0])*0.6, ylim2[-1]*0.8, '220 GHz \n band', fontsize = 10)

ax[2].grid(which='both')
l = ax[0].legend([(p1, p2), (e1, e2)], ['Input sky', '68% C.L.'], numpoints=1, loc = 4, fontsize = 12,
			   handler_map={tuple: HandlerTuple(ndivide=None)})

ax[0].grid()
ax[0].set_title('GC patch - {} year'.format(dictionaries[0]['effective_duration']),fontsize=16)
ax[0].set_ylabel(r'$I(\nu)$ [$\mu$K]',fontsize=16)
ax[0].set_xlabel(r'$\nu$[GHz]',fontsize=16)

ax[2].set_title('QUBIC patch - {} years'.format(dictionaries[0]['effective_duration']),fontsize=16)
ax[2].set_ylabel(r'$I(\nu)$ [$\mu$K]',fontsize=16)
ax[2].set_xlabel(r'$\nu$[GHz]',fontsize=16)

# Displaying maps
ax[1].cla()
plt.axes(ax[1])
hp.gnomview(maps_ud[2, -1, :, 0], reso = 15,hold = True, 
			notext = True, title = ' ',
			min = 0,
			max = 0.4*np.max(maps_ud[2, -1, :, 0]), 
			unit = r'$\mu$K',
			rot = centers[2])
hp.projscatter(hp.pix2ang(nside_new, pixs_ud[2]), marker = '*', color = 'r', s = 180)
dpar = 10
dmer = 20
#Watch out, the names are wrong (change it)
mer_coordsG = [centers[2][0] - dmer, centers[2][0], centers[2][0] + dmer]
long_coordsG = [centers[2][1] - 2*dpar, centers[2][1] - dpar, 
				centers[2][1], centers[2][1] + dpar, centers[2][1] + 2 * dpar]
#paralels
for ilong in long_coordsG:
	plt.text(np.deg2rad(mer_coordsG[0] - 12), 1.1*np.deg2rad(ilong), 
			 r'{}$\degree$'.format(ilong))
#meridians
for imer in mer_coordsG:
	if imer < 0:
		jmer = imer + 360
		ip, dp = divmod(jmer/15,1)
	else:
		ip, dp = divmod(imer/15,1)
	if imer == 0:
		plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
			 r'{}$\degree$'.format(int(ip) ))
	else:
		plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
			 r'{}$\degree$'.format(imer))
			 #r'{}h{}m'.format(int(ip), int(round(dp*60))))
hp.projtext(mer_coordsG[1] + 2, long_coordsG[0] - 6, '$l$',  color = 'k', lonlat=True)
hp.projtext(mer_coordsG[2] + 12.5, long_coordsG[2] - 1, '$b$', rotation = 90, color = 'k', lonlat=True)

ax[3].cla()
plt.axes(ax[3])
hp.gnomview(maps_ud[1, -1, :, 0], reso = 15, hold = True, 
			notext = True, title = ' ',
			unit = r'$\mu$K',
			min = 0,
			max = 0.4*np.max(maps_ud[1, -1, :, 0]), 
			rot = centerQ)
hp.projscatter(hp.pix2ang(nside_new, pixQ_ud),marker = '*', color = 'r', s = 180)

mer_coordsQ = [centers[1][0] - dmer, centers[0][0]+0, centers[0][0] + dmer]
long_coordsQ = [centers[0][1] - 2*dpar, centers[0][1] - dpar, centers[0][1], 
				centers[0][1] + dpar, centers[0][1] + 2 * dpar]
#paralels
for ilong in long_coordsQ:
	plt.text(np.deg2rad(mer_coordsQ[0]-360+31), 1.1*np.deg2rad(ilong+58), r'{:.0f}$\degree$'.format(ilong),)
#meridians
for imer in mer_coordsQ:
	ip, dp = divmod(imer/15,1)
	plt.text(-np.deg2rad(imer-360+48), np.deg2rad(long_coordsQ[-1]+58+7), 
		 r'{:.1f}$\degree$'.format(imer))
		 
hp.graticule(dpar = dpar, dmer = dmer, alpha = 0.6, verbose = False)

plt.tight_layout()
if savefigs:
	try:
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+intensity_out_plot+".svg", 
			format = 'svg', bbox_inches='tight')
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+intensity_out_plot+".pdf", 
			format = 'pdf', bbox_inches='tight')
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+intensity_out_plot, bbox_inches='tight')
	except:
		plt.savefig("./"+intensity_out_plot+".svg", 
			format = 'svg', bbox_inches='tight')
		plt.savefig("./"+intensity_out_plot+".pdf", 
			format = 'pdf', bbox_inches='tight')
		plt.savefig("./"+intensity_out_plot, bbox_inches='tight')
else:
	plt.show()


print("# _______ Plotting SED Polarization_______ ")
polarization_out_plot = 'TEST4JCH_{}_nrec{}_nside{}_pixQ{}_pixG{}_Polarization_grat'.format(
								FuncModel.__name__, d150Q['nf_recon'],nside_new, pixQ_ud, pixG_ud)

print(" \t \t in {} [.svg, .png and .pdf]".format(polarization_out_plot))
# Plotting
p1, = ax[0].plot(nus_out150, 
			   np.sqrt(fgr_map_ud[2,:,pixs_ud[2],1] ** 2 + fgr_map_ud[2,:,pixs_ud[2],2] ** 2),
			   'ro', lw = 3, label = 'Input sky')
p2, = ax[0].plot(nus_out220, 
			   np.sqrt(fgr_map_ud[3][:,pixs_ud[3],1] ** 2 + fgr_map_ud[3][:,pixs_ud[3],2] ** 2),
			   'bo', lw = 3)

e1 = ax[0].fill_between(xarr[2], y1 = ySED_fit[2, :, 1] - Perr[2], 
						y2 = ySED_fit[2, :, 1] + Perr[2], 
				   color = 'r', alpha = 0.3, label = '68% C.L. ')
e2 = ax[0].fill_between(xarr[3], y1 = ySED_fit[3, :, 1] - Perr[3], 
						y2 = ySED_fit[3, :, 1] + Perr[3], 
				   color = 'b', alpha = 0.3)
ax[0].axvspan(nus150[-1], nus220[0], color = 'k', alpha = greyscale)
ax[2].plot(nus_out150, 
			   np.sqrt(fgr_map_ud[0,:,pixs_ud[0],1] ** 2 + fgr_map_ud[0, :, pixs_ud[0], 2] ** 2),
			   'ro', lw = 3)
ax[2].plot(nus_out220, 
			   np.sqrt(fgr_map_ud[1, :, pixs_ud[1], 1] ** 2 + fgr_map_ud[1, :, pixs_ud[1], 2] ** 2),
			   'bo', lw = 3)
ax[2].fill_between(xarr[0], y1 = ySED_fit[0, :, 1] - Perr[0], 
				   y2 = ySED_fit[0, :, 1] + Perr[0], 
				   color = 'r', alpha = 0.3)
ax[2].fill_between(xarr[1], y1 = ySED_fit[1, :, 1] - Perr[1], 
				   y2 = ySED_fit[1, :, 1] + Perr[1], 
				   color = 'b', alpha = 0.3)

# Setting
ax[0].set_title('GC patch - {} year'.format(dictionaries[0]['effective_duration']), fontsize = 14)
ax[0].set_ylabel(r'$P(\nu)~[\mu$K]', fontsize = 14)
ax[0].set_xlabel(r'$\nu~[GHz]$', fontsize = 14)
ax[0].legend(loc = 2, fontsize = 12)
ax[0].grid()
ax[2].set_xlabel(r'$\nu~[GHz]$', fontsize = 14)
ax[2].axvspan(nus150[-1], nus220[0], color = 'k', alpha = greyscale)
ax[2].set_ylabel(r'$P(\nu)~[\mu$K]', fontsize = 14)
ax[2].set_title('QUBIC patch - {} years'.format(dictionaries[0]['effective_duration']),fontsize=14)
ax[2].grid()

xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
xlim2 = ax[2].get_xlim()
ylim2 = ax[2].get_ylim()

ax[0].axvspan(xlim2[0], nus150[0], color = 'k', alpha = greyscale)
ax[0].axvspan(nus220[-1], xlim2[-1], color = 'k', alpha = greyscale)
ax[2].axvspan(xlim2[0], nus150[0], color = 'k', alpha = greyscale)
ax[2].axvspan(nus220[-1], xlim2[-1], color = 'k', alpha = greyscale)

ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)
ax[0].text(xlim[0]+(xlim[1]-xlim[0])*0.1, ylim[-1]*0.8, '150 GHz \n band', fontsize = 10)
ax[0].text(xlim[0]+(xlim[1]-xlim[0])*0.6, ylim[-1]*0.8, '220 GHz \n band', fontsize = 10)
ax[2].set_xlim(xlim2)
ax[2].set_ylim(ylim2)
ax[2].text(xlim2[0]+(xlim2[1]-xlim2[0])*0.1, ylim2[-1]*0.8, '150 GHz \n band', fontsize = 10)
ax[2].text(xlim2[0]+(xlim2[1]-xlim2[0])*0.6, ylim2[-1]*0.8, '220 GHz \n band', fontsize = 10)

# Displaying maps    
plt.axes(ax[1])
auxmapG = np.sqrt(maps_ud[2, 0, :, 1] ** 2 + maps_ud[2, 0, :, 2] ** 2)
auxmapG[~cov_ud[2]] = hp.UNSEEN
hp.gnomview(auxmapG,
			reso = 15, hold = True, notext = True, 
			title = ' ',
			min = 0,
			cbar = True,
			unit = r'$\mu$K',
			rot = centers[2])
hp.projscatter(hp.pix2ang(nside_new, pixs_ud[2]),marker = '*',color = 'r', s = 180)
dpar = 10
dmer = 20
#Watch out, the names are wrong (change it)
mer_coordsG = [centers[2][0] - dmer, centers[2][0], centers[2][0] + dmer]
long_coordsG = [centers[2][1] - 2*dpar, centers[2][1] - dpar, 
				centers[2][1], centers[2][1] + dpar, centers[2][1] + 2 * dpar]
#paralels
for ilong in long_coordsG:
	plt.text(np.deg2rad(mer_coordsG[0] - 12), 1.1*np.deg2rad(ilong), 
			 r'{}$\degree$'.format(ilong))
#meridians
for imer in mer_coordsG:
	if imer < 0:
		jmer = imer + 360
		ip, dp = divmod(jmer/15,1)
	else:
		ip, dp = divmod(imer/15,1)
	if imer == 0:
		plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
			 r'{}$\degree$'.format(int(ip) ))
	else:
		plt.text(-np.deg2rad(imer + 3), np.deg2rad(long_coordsG[-1] + 6), 
			 r'{}$\degree$'.format(imer))
			 #r'{}h{}m'.format(int(ip), int(round(dp*60))))
hp.projtext(mer_coordsG[1] + 2, long_coordsG[0] - 6, '$l$',  color = 'k', lonlat=True)
hp.projtext(mer_coordsG[2] + 12.5, long_coordsG[2] - 1, '$b$', rotation = 90, color = 'k', lonlat=True)

plt.axes(ax[3])
auxmapQ = np.sqrt(maps_ud[0, 0, :, 1] ** 2 + maps_ud[0, 0, :, 2] ** 2)
auxmapQ[~cov_ud[0]] = hp.UNSEEN
hp.gnomview(auxmapQ,
			reso = 15, hold = True, notext = True, 
			max = 7,
			min = 0,
			title = ' ',
			cbar = True,
			unit = r'$\mu$K',
			rot = centers[0])
hp.projscatter(hp.pix2ang(nside_new,pixs_ud[0]), marker = '*', color = 'r', s = 180)
mer_coordsQ = [centers[0][0] - dmer, centers[0][0]+0, centers[0][0] + dmer]
long_coordsQ = [centers[0][1] - 2*dpar, centers[0][1] - dpar, 
				centers[0][1], centers[0][1] + dpar, centers[0][1] + 2 * dpar]
#paralels
for ilong in long_coordsQ:
	plt.text(np.deg2rad(mer_coordsQ[0]-360+31), 1.1*np.deg2rad(ilong+58), r'{:.0f}$\degree$'.format(ilong),)
#meridians
for imer in mer_coordsQ:
	ip, dp = divmod(imer/15,1)
	plt.text( - np.deg2rad(imer - 360 + 48), np.deg2rad(long_coordsQ[-1] + 58 + 7), 
		 r'{:.1f}$\degree$'.format(imer))

hp.graticule(dpar = dpar, dmer = dmer, alpha = 0.6, verbose = False)
l = ax[0].legend([(p1, p2), (e1, e2)], ['Input sky', '68% C.L.'], numpoints=1, loc = 4, fontsize = 12,
			   handler_map={tuple: HandlerTuple(ndivide=None)})
plt.tight_layout()#plt.tight_layout()

if savefigs: 
	try:
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+polarization_out_plot+".svg", 
			format = 'svg', bbox_inches='tight')
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+polarization_out_plot+".pdf", 
			format = 'pdf', bbox_inches='tight')
		plt.savefig("Figs-FI-SED/March2021/NSIDE{}/".format(nside_new)+polarization_out_plot, bbox_inches='tight')
	except:
		plt.savefig("./"+polarization_out_plot+".svg", 
			format = 'svg', bbox_inches='tight')
		plt.savefig("./"+polarization_out_plot+".pdf", 
			format = 'pdf', bbox_inches='tight')
		plt.savefig("./"+polarization_out_plot, bbox_inches='tight')
else:
	plt.show()

print("# _-_-_-_All done.")