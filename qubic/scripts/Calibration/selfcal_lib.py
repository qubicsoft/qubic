import fibtools as ft
from qubicpack import qubicpack as qp
import numpy as np

def toto():
    print 'toto'

def tes2imgpix(tesnum, extra_args=None):
    if extra_args is None:
        a1 = qp()
        a1.assign_asic(1)
        a2 = qp()
        a2.assign_asic(2)
    else:
        a1 = extra_args[0]
        a2 = extra_args[1]
        
    ij = np.zeros((len(tesnum),2))
    for i in xrange(len(tesnum)):
        if i < 128:
            pixnum = a1.tes2pix(tesnum[i])
            ww = np.where(a1.pix_grid == pixnum)
        else:
            pixnum = a2.tes2pix(tesnum[i]-128)
            ww = np.where(a2.pix_grid == pixnum)
        if len(ww[0])>0:
            ij[i,:] = ww
        else:
            ij[i,:] = [17,17]
    return ij


def fringe_focalplane(x, pars, extra_args=None):    
    baseline = pars[0]
    alpha = pars[1]
    phase = pars[2]
    amplitude = pars[3]
    nu = 150e9
    lam = 3e8/nu
    f = 300e-3 # Focal Length in mm
    freq_fringe = baseline / lam
    TESsize = 3.e-3

    ijtes = tes2imgpix(np.arange(256)+1, extra_args=extra_args)
    
    fringe=amplitude*np.cos(2.*np.pi*freq_fringe*(ijtes[:,0]*np.cos(alpha*np.pi/180)+ijtes[:,1]*np.sin(alpha*np.pi/180))*TESsize/f+phase*np.pi/180)
    thermos = [4-1,36-1, 68-1, 100-1, 4-1+128, 36-1+128, 68-1+128, 100-1+128]
    fringe[thermos] = 0
    mask = x > 0
    fringe[~mask] = 0
    return fringe
