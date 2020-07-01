#!/usr/bin/python
from pylab import *
import os
import sys
import time
import pickle

# Specific science modules
import healpy as hp
import numpy as np

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
#from mpi4py import MPI
# from pyoperators import MPI
import pysm
import qubic
from qubic import QubicSkySim as qss
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import NamasterLib as nam
from qubic import mcmc

########################################################################################################
#### Now in a function to loop over ell binning, lmin, coverage
def run_mc_spectroim(nsub, nbmc, Namaster, d, signoise, cov, effective_variance_invcov, verbose=False, clnoise=None):
    ell_bins, b = Namaster.get_binning(d['nside'])
    mask_apo = Namaster.mask_apo
    okpix = cov > (np.max(cov)*0.1)

    myd = d.copy()
    myd['nf_sub']=nsub
    seed = np.random.randint(1,100000)
    sky_config = {'cmb': seed}
    Qubic_sky = qss.Qubic_sky(sky_config, myd)

    w=None
    cl_noise_qubic = np.zeros((nbmc, nsub, len(ell_bins), 4))
    print('        Starting MC')
    for imc in range(nbmc):
        t0 = time.time()
        qubicnoiseA = Qubic_sky.create_noise_maps(signoise, cov, nsub=nsub,
                                                  effective_variance_invcov=effective_variance_invcov,
                                                 clnoise=clnoise)
        qubicnoiseB = Qubic_sky.create_noise_maps(signoise, cov, nsub=nsub,
                                                  effective_variance_invcov=effective_variance_invcov,
                                                 clnoise=clnoise)
        

        for isub in range(nsub):
            ### Compute Spectra:
            # Noise Only
            print('   - QUBIC Noise maps sub {}'.format(isub))
            leff, cl_noise_qubic[imc, isub, :,:], w = Namaster.get_spectra(qubicnoiseA[isub,:,:].T, 
                                                                     map2 = qubicnoiseB[isub,:,:].T,
                                                                     purify_e=False, purify_b=True, w=w, verbose=False,
                                                                     beam_correction=True)
        t1 = time.time()
        print('             Monte-Carlo: Iteration {0:} over {1:} done in {2:5.2f} sec'.format(imc, nbmc,t1-t0))
        
    
    # average MC results
    return leff, cl_noise_qubic
########################################################################################################



### Decode arguments
outname = str(sys.argv[1])
file_noise_profile = str(sys.argv[2])
cov_file = str(sys.argv[3])
clnoise_file = str(sys.argv[4])
nbmc = int(sys.argv[5])
signoise = float(sys.argv[6])
lmin = int(sys.argv[7])
delta_ell = int(sys.argv[8])
covcut = float(sys.argv[9])
is_spatial_corr = int(sys.argv[10])
nsub = int(sys.argv[11])





### Initialize
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
dictfilename = global_dir + '/dicts/BmodesNoDustNoSystPaper0_2020.dict'
# Read dictionary chosen
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
d['nside'] = 256
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])

### Open Coverage File
cov = np.array(FitsArray(cov_file))

### Open Noise Profile
fit_n200k = pickle.load( open( file_noise_profile, "rb" ) )

### Open Cl for spatially  correlated noise
if is_spatial_corr == 0:
    print('No Spatial Correlations')
    clth = None
else:
    print('Spatial correlations from '+clnoise_file)
    clth = pickle.load( open( clnoise_file, "rb" ) )
    alpha = 4.5 ### See notebook called "2pt-Correlation Function" for an empirical explanation of alpha
    clth = (clth -1 ) * alpha + 1

print('Spatial correlations:')
print(clth)


################################### Flat Weighting #################################################################
### Create a Namaster object
lmax = 2 * d['nside'] - 1
okpix = cov > np.max(cov) * covcut

### We use Flat weighting
maskpix = np.zeros(12*d['nside']**2)
maskpix[okpix] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell)

### Run the MC
leff, mcl_noise_qubic, scl_noise_qubic = run_mc(nbmc, Namaster, d, signoise, cov, fit_n200k, clnoise=clth)

### Save Output
outfile = outname + 'MCFastNoise_n_{}_sig_{}_lmin_{}_dl_{}_cc_{}_Flat.pk'.format(nbmc, signoise, lmin, delta_ell, covcut)
pickle.dump([leff, mcl_noise_qubic, scl_noise_qubic, sys.argv], open(outfile, "wb"))
print('Leff:')
print(leff)
print('Errors BB:')
print(scl_noise_qubic[:,2])
####################################################################################################################

# ################################### Cov Weighting #################################################################
# ### Create a Namaster object
# lmax = 2 * d['nside'] - 1
# okpix = cov > np.max(cov) * covcut
#
# ### We use Flat weighting
# maskpix = np.zeros(12*d['nside']**2)
# maskpix[okpix] = cov[okpix]/np.max(cov[okpix])
# NamasterCov = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
#
# ### Run the MC
# leff, mcl_noise_qubic, scl_noise_qubic = run_mc(nbmc, NamasterCov, d, signoise, cov, fit_n200k, clnoise=clth)
#
# ### Save Output
# outfile = outname + 'MCFastNoise_n_{}_sig_{}_lmin_{}_dl_{}_cc_{}_Cov.pk'.format(nbmc, signoise, lmin, delta_ell, covcut)
# pickle.dump([leff, mcl_noise_qubic, scl_noise_qubic, sys.argv], open(outfile, "wb"))
# ####################################################################################################################




   








