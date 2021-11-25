import qubicplus
import CMBS4
import pysm3
import pysm3.units as u
from pysm3 import utils
import numpy as np
from qubic import camb_interface as qc
import healpy as hp
import matplotlib.pyplot as plt
import os
import random as rd
import string
import qubic
from importlib import reload
import pickle
import sys

center = qubic.equ2gal(0, -57)
# If there is not this command, the kernel shut down every time..
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

nside=256
def get_coverage(fsky, nside, center_radec=[0., -57.]):
    center = qubic.equ2gal(center_radec[0], center_radec[1])
    uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
    uvpix = np.array(hp.pix2vec(nside, np.arange(12*nside**2)))
    ang = np.arccos(np.dot(uvcenter, uvpix))
    indices = np.argsort(ang)
    okpix = ang < -1
    okpix[indices[0:int(fsky * 12*nside**2)]] = True
    mask = np.zeros(12*nside**2)
    mask[okpix] = 1
    return mask

seed=int(sys.argv[1])
N=int(sys.argv[2])
nbmc=int(sys.argv[3])


covmap = get_coverage(0.03, nside)
skyconfig = {'cmb':seed, 'dust':'d0'}

thr = 0.1
mymask = (covmap > (np.max(covmap)*thr)).astype(int)
pixok = mymask > 0

pkl_file = open('/pbs/home/m/mregnier/sps1/QUBIC+/S4_dict.pkl', 'rb')
S4_dict = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('/pbs/home/m/mregnier/sps1/QUBIC+/BI_dict.pkl', 'rb')
BI_dict = pickle.load(pkl_file)
pkl_file.close()

from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)
import fgbuster as fgb
from fgbuster import basic_comp_sep, get_instrument


def separate(comp, instr, maps_to_separate, tol=1e-5, print_option=False):
    solver_options = {}
    solver_options['disp'] = False
    fg_args = comp, instr, maps_to_separate
    fg_kwargs = {'method': 'L-BFGS-B', 'tol': tol, 'options': solver_options}
    try:
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    except KeyError:
        fg_kwargs['options']['disp'] = False
        res = fgb.basic_comp_sep(*fg_args, **fg_kwargs)
    if print_option:
        print()
        print("message:", res.message)
        print("success:", res.success)
        print("result:", res.x)

    return res


def reconvolve(maps, fwhms, ref_fwhm, verbose=False):
    if verbose:
        print('Reconvolution to common FWHM')
    sig_conv = np.sqrt(ref_fwhm**2 - fwhms**2)
    maps_out = np.zeros_like(maps)
    for i in range(len(fwhms)):
        if sig_conv[i] == 0:
            if verbose:
                print('Map {0:} fwhmin={1:6.3f} fwhmout={2:6.3f} => We do not reconvolve'.format(i,
                                                                                             fwhms[i], ref_fwhm))
            maps_out[i,:] = maps[i,:]
        else:
            if verbose:
                print('Map {0:} fwhmin={1:6.3f} fwhmout={2:6.3f} => We reconvolve with {3:6.3f}'.format(i,
                                                                                                    fwhms[i],
                                                                                                    ref_fwhm,
                                                                                                    sig_conv[i]))
            maps_out[i,:] = hp.smoothing(maps[i,:], fwhm=np.deg2rad(sig_conv[i]), pol=True, verbose=False)
    return maps_out

ref_fwhm=0.5

def give_me_truemaps(config, nu0, ref_fwhm, pixok):
    sky = qubicplus.QUBICplus(config, BI_dict).get_sky()
    maps = sky.get_emission(nu0*u.GHz, None)*utils.bandpass_unit_conversion(nu0*u.GHz,None, u.uK_CMB)
    maps = hp.sphtfunc.smoothing(maps, fwhm=np.deg2rad(ref_fwhm),verbose=False)
    maps[:, ~pixok] = hp.UNSEEN
    return maps

truecmb = give_me_truemaps({'cmb':seed}, 150, ref_fwhm, pixok)
truedust = give_me_truemaps({'dust':'d0'}, 150, ref_fwhm, pixok)



def run_mc(N, seed):
    rx_s4 = np.zeros((2, N))
    rs_s4 = np.zeros((((N, 2, 3, 12*256**2))))

    rx_qp = np.zeros((2, N))
    rs_qp = np.zeros((((N, 2, 3, 12*256**2))))
    for i in range(N):
        reload(CMBS4)
        s4=CMBS4.S4({'cmb':seed, 'dust':'d0'}, S4_dict)
        qp=qubicplus.QUBICplus({'cmb':seed, 'dust':'d0'}, BI_dict)
        mapS4_noisy_withoutiib, _, _ = s4.getskymaps(same_resol=ref_fwhm,
                                                         iib=False,
                                                         verbose=True,
                                                         coverage=covmap,
                                                         noise=True,
                                                         signoise=1.)

        mapqp_noisy_withoutiib, _, _ = qp.getskymaps(same_resol=ref_fwhm,
                                                         iib=False,
                                                         verbose=True,
                                                         coverage=covmap,
                                                         noise=True,
                                                         signoise=1.)

        thr = 0.01
        mymask = (covmap > (np.max(covmap)*thr)).astype(int)
        pixok = mymask > 0

        comp = [Dust(nu0=150.), CMB()]

        # CMB-S4
        instr = get_instrument('CMBS4')
        instr.fwhm = np.ones(9)*ref_fwhm*60
        r_s4_withoutiib=separate(comp, instr, mapS4_noisy_withoutiib[:, :, pixok], tol=1e-6)
        rx_s4[:, i] = r_s4_withoutiib.x
        rs_s4[i, :, :, pixok] = np.transpose(r_s4_withoutiib.s, (2, 0, 1))

        # QUBIC+
        instr = get_instrument('Qubic+')
        #instr.frequency = BI_dict['frequency']
        instr.fwhm = np.ones(45)*ref_fwhm*60
        r_qp_withoutiib=separate(comp, instr, mapqp_noisy_withoutiib[:, :, pixok], tol=1e-6)
        rx_qp[:, i] = r_qp_withoutiib.x
        rs_qp[i, :, :, pixok] = np.transpose(r_qp_withoutiib.s, (2, 0, 1))
        del s4
        del qp

    return rx_s4, rx_qp

rx_s4, rx_qp = run_mc(N, seed)

mydict = {'r_x_s4':rx_s4,
          'r_x_qp':rx_qp}


output = open('/pbs/home/m/mregnier/sps1/QUBIC+/results/betatemp_estimation_seed{}_{}reals_{}.pkl'.format(seed, N, nbmc), 'wb')
pickle.dump(mydict, output)
output.close()
