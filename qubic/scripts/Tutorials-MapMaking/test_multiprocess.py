# QUBIC stuff
import qubic
from qubicpack.utilities import Qubic_DataDir
from qubic.data import PATH
from qubic.io import read_map
from qubic.scene import QubicScene
from qubic.samplings import create_random_pointings, get_pointing

# General stuff
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pysm3
import os
import warnings
warnings.filterwarnings("ignore")
import pysm3.units as u
from importlib import reload
from pysm3 import utils
import Acquisition as Acq
# FG-Buster packages
import component_model as c
import mixing_matrix as mm
import pickle
from scipy.optimize import minimize
import ComponentsMapMakingTools as CMMTools
import multiprocess as mp
import time
# PyOperators stuff
from pysimulators import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

def get_dictionary(nsub, nside, pointing, band):
    dictfilename = 'dicts/pipeline_demo.dict'
    
    # Read dictionary chosen
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    d['nf_recon'] = nsub
    d['nf_sub'] = nsub
    d['nside'] = nside
    d['RA_center'] = 100
    d['DEC_center'] = -157
    center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
    d['effective_duration'] = 3
    d['npointings'] = pointing
    d['filter_nu'] = int(band*1e9)
    d['photon_noise'] = False
    d['config'] = 'FI'
    d['MultiBand'] = True
    
    return d, center

Nsub = 2
nside = 256
pointing = 1000

d150, center = get_dictionary(Nsub, nside, pointing, 150)
d220, center = get_dictionary(Nsub, nside, pointing, 220)


qubic150 = Acq.QubicIntegratedComponentsMapMaking(d150, Nsub=Nsub, comp=[c.Dust(nu0=150, temp=20)])
qubic220 = Acq.QubicIntegratedComponentsMapMaking(d220, Nsub=Nsub, comp=[c.Dust(nu0=150, temp=20)])

qu = Acq.QubicTwoBandsComponentsMapMaking(qubic150, qubic220, comp=[c.Dust(nu0=150, temp=20)])
allexp = Acq.QubicOtherIntegratedComponentsMapMaking(qu, [143, 353], comp=[c.Dust(nu0=150, temp=20)], nintegr=1)

#beta=1.54+0.1*np.random.randn(12*8**2)#np.array([1.54])
components = qubic150.get_PySM_maps({'dust':'d1'})
sky = pysm3.Sky(nside, preset_strings=['d1'])
sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
#beta = np.ones(12*8**2)*1.54#np.array(sky.components[0].mbb_index)
beta = np.array(sky.components[0].mbb_index)


cov = qu.get_coverage()
pixok = cov > 0
pixok = hp.ud_grade(pixok, 1)
beta = hp.ud_grade(beta, 1)

hh = allexp.get_operator(beta, False)
pip = Acq.PipelineReconstruction(qu, [143, 353], comp=[c.Dust(nu0=150, temp=20)], nintegr=1, H=hh)
tod = hh(components.T)

def chi2(x, betamap, sol, data, patch_id):
    return pip.myChi2(x, betamap, sol, data, patch_id)



t = time.time()
fitted = pip.fit_beta(chi2, beta, components.T, tod, x0=np.array([1.5]), mask=np.ones(len(beta), dtype=bool), 
processes=1, N=1, options={'eps':1e-6}, tol=1e-2, method='L-BFGS-B')
print('Execution time : ', time.time() - t)
print(np.mean(fitted - beta))

t = time.time()
fitted = pip.fit_beta(chi2, beta, components.T, tod, x0=np.array([1.5]), mask=np.ones(len(beta), dtype=bool), 
processes=os.cpu_count(), N=os.cpu_count(), options={'eps':1e-6}, tol=1e-2, method='L-BFGS-B')
print('Execution time : ', time.time() - t)
print(np.mean(fitted - beta))