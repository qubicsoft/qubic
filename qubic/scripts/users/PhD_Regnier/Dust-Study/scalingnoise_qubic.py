import numpy as np
import qubic
import os
import matplotlib.pyplot as plt
from qubic import progress_bar
import fgbuster
import pickle
import warnings
import pysm3.units as u
import cmbdb
from pysm3 import utils
import pysm3
from qubic import NamasterLib as nam
from qubic import QubicSkySim as qss
import sys
import pickle
import healpy as hp
import warnings
warnings.filterwarnings("ignore")
import forecast_def as fd
#import myMCMC
#import emcee


###################################
#            Arguments            #
###################################

nside=256
lmin=21
lmax=355
dl=35
exp='S4'
nside_fgb=int(sys.argv[1])
r=float(sys.argv[2])
corr=int(sys.argv[3])
nsub=int(sys.argv[4])
Alens=0.1
n_ite=int(sys.argv[5])
nbands=1


###################################
#            Experiences          #
###################################

myconf=fd.get_list_config(name=exp, nsub=nbands)
config=fd.combine_config(myconf)
instr=fd.get_instr_simple(config)
fsky=config['fsky']
real_nus=config['frequency']
nus_edge=config['edges']
covmap=fd.get_coverage(fsky, nside)
pixok=covmap>0
Alens=0.1
      
###################################
#            Namaster             #
###################################

maskpix = np.zeros(12*nside**2)
pixok = covmap > 0
maskpix[pixok] = 1
aposize=4
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl, aposize=aposize)
Namaster.fsky = fsky
Namaster.ell_binned, _ = Namaster.get_binning(nside)


###################################
#            Monte-Carlo          #
###################################

if corr == 1000:
    dust='d0'
    sync='s0'
    extra=None
elif corr==2000:
    dust='d1'
    sync='s1'
    extra=None
elif corr==2002:
    dust='d2'
    sync='s1'
    extra=None
elif corr==2003:
    dust='d3'
    sync='s1'
    extra=None
elif corr==2004:
    dust='d4'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2005:
    dust='d5'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2007:
    dust='d7'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2008:
    dust='d8'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2009:
    dust='d9'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2010:
    dust='d10'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2011:
    dust='d11'
    sync='s1'
    extra=None
    nus_edge=None
elif corr==2012:
    dust='d12'
    sync='s1'
    extra=None
    nus_edge=None
else:
    dust='d6'
    sync='s1'
    extra={'correlation_length':corr}

name_typ='fullpipeline'
name_typn='_'

allfreqs=np.linspace(1, 320, 100000)
allseed=np.linspace(1, 320, len(allfreqs))*n_ite#np.random.randint(1, 100000, len(allfreqs))

if extra is not None:
    for i in extra:
        print(i, extra[i])
        pysm3.sky.PRESET_MODELS[dust][str(i)]=extra[i]
                
print('FOREGROUNDS GENERATION WITH DUST MODEL {} AND SYNC MODEL {}'.format(dust, sync))
sky=pysm3.Sky(nside=nside, preset_strings=[dust, sync])

if corr != 1000:
    print('FIXING T = 20K')
    sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit
#if dust != 'd4' and dust != 'd5' and dust != 'd7' and dust != 'd8' and dust != 'd9':
#    print('FIX TEMPERATURE AT 20°K')
#    sky.components[0].mbb_temperature=20*sky.components[0].mbb_temperature.unit


#allfore, index = fd.foregrounds_all_freqs(allfreqs, nside=nside, dust=dust, NSIDE_PATCH=nside_fgb, extra=extra, bandpass=False)
s = 20
nsub=[nsub]
clBB=np.zeros((s, len(nsub), len(Namaster.ell_binned)))
#total_residuals=np.zeros((len(nsub), 2, 12*nside**2))
#total_residuals_fore=np.zeros((len(nsub), 2, 12*nside**2))
if nside_fgb == 0:
    beta=np.zeros((s, len(nsub), 2))
else:
    beta=np.zeros((s, len(nsub), 2, 12*nside_fgb**2))


for ii, i in enumerate(nsub):
    
        myconf=fd.get_list_config(name=exp, nsub=i)
        config=fd.combine_config(myconf)
        
        real_nus=config['frequency']
        instr=fd.get_instr(real_nus, config, N_SAMPLE_BAND=100)
        fsky=config['fsky']
        nus_edge=config['edges']
        for j in range(s):
            print(ii, j)
            leff, clBB[j, ii], res_maps, beta[j, ii], cmb = fd.Forecast(real_nus, Namaster, dust, instr, nside, extra, Alens, r, pixok, sky, config).RUN_MC(allnus=allfreqs, allseed=allseed, NSIDE_PATCH=nside_fgb)    


###################################
#          Saving results         #
###################################

mypath='/home/regnier/work/regnier/forecast_deco/results/new_data/'

mydict = {'mycl': clBB, 'leff': leff, 'beta':beta, 'pixok':pixok}#, 'index_true':index}
output = open(mypath+'cl_nsub{}_r{:.3f}_{}_{}_nsidefgb{}_{}{}corr{:.0f}_{}reals_{}.pkl'.format(nsub[0], r, exp, dust, nside_fgb, name_typ, name_typn, corr, s, n_ite), 'wb')
pickle.dump(mydict, output)
output.close()
