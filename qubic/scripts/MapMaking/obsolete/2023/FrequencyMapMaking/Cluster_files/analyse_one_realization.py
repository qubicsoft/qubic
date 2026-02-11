# This script allow to analyse one realization of frequency map-making from computer cluster.

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import pickle
import sys
sys.path.append('/home/regnier/work/regnier/mypackages')
import frequency_acquisition as Acq
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

data = 'MM_band220_bandpasscorrectionFalse_Nrec1_Nsub15_Ntod100_correction_conv0.0deg_noiseFalse.pkl'
istk = 2
res = 15

def cl2dl(ell, cl):

    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl

path = os.getcwd() + '/data/' + data

### Loading data
with open(path, 'rb') as f:
    x = pickle.load(f)

myinputs = x['input'].copy()      # Input
myoutputs = x['output'].copy()    # Output
Nf_recon = myoutputs.shape[0]     # Nrec
seenpix = x['seenpix'].copy()     # pixels seen by QUBIC
center = x['center']              # Location of the patch
allfwhm = x['allfwhm']
leff = x['leff']
DlBB = x['Dl_BB']
fact_sub = 2

stk = ['I', 'Q', 'U']



plt.figure(figsize=(10, 8))

k=0
for i in range(Nf_recon):

    if istk == 0:
        v = 300
    else:
        v = 8
    
    ### Inputs
    target = 0#np.min(allfwhm[i*fact_sub:(i+1)*fact_sub])
    print(target)
    C = HealpixConvolutionGaussianOperator(fwhm=target)#allfwhm_ref[i])
    myinputs[i] = C(myinputs[i])
    
    myinputs[:, ~seenpix, :] = hp.UNSEEN
    myoutputs[:, ~seenpix, :] = hp.UNSEEN
    
    hp.gnomview(myinputs[i, :, istk], cmap='jet', rot=x['center'], reso=res, min=-v, max=v, sub=(Nf_recon, 3, k+1), title='Input')
    hp.gnomview(myoutputs[i, :, istk], cmap='jet', rot=x['center'], reso=res, min=-v, max=v, sub=(Nf_recon, 3, k+2), title='Output')

    resi = (myoutputs[i, :, istk]-myinputs[i, :, istk])
    resi[~seenpix] = hp.UNSEEN
    r = 8#1.5 * np.std(resi[seenpix])

    hp.gnomview(resi, cmap='jet', rot=x['center'], reso=res, title=f'Residual - RMS : {np.std(resi[seenpix]):.3e}', min=-r, max=r, sub=(Nf_recon, 3, k+3))

    k+=3

plt.savefig(os.getcwd() + '/data/' + f'Recons_Nrec{Nf_recon}.png')
plt.close()


plt.figure(figsize=(10, 10))

ell = np.arange(2, 4000, 1)
mycls = cl2dl(ell, Acq.give_cl_cmb(r=0, Alens=1.)[2])

plt.plot(ell, mycls)
plt.plot(leff, DlBB, '-or')

plt.yscale('log')
plt.xlim(20, 500)
plt.ylim(4e-5, 5e-1)
plt.savefig(os.getcwd() + '/data/' + f'DlBB15_Nrec{Nf_recon}.png')
plt.close()