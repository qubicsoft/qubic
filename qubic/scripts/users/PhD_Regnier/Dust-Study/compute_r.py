import numpy as np
import pickle
import os.path as op
import myMCMC
import healpy as hp
import scipy
import warnings
from qubic import NamasterLib as nam
import forecast_def as fd
import sys
import scipy
warnings.filterwarnings("ignore")
CMB_CL_FILE = op.join('/work/regnier/forecast_deco/Cls_Planck2018_%s.fits')

myr = 0.000
Nf = int(sys.argv[1])

def _get_Cl_cmb(Alens, r):
    power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:,:4000]
    return power_spectrum
def cl2dl(ell, cl):

    dl=np.zeros(ell.shape[0])
    for i in range(ell.shape[0]):
        dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
    return dl
def myBBth(ell, r, Alens):
    clBB = cl2dl(ell, _get_Cl_cmb(Alens=Alens, r=r)[2, ell.astype(int)-1])
    return clBB


PATH_to_DATA = '/work/regnier/forecast_deco/results/new_data/'

with open(PATH_to_DATA+'cl_nsub{}_nside256_r{:.3f}_S4_d1_nsidefgb8_fullpipeline_corr2000_500reals.pkl'.format(Nf, myr), 
              'rb') as f:
    xd1 = pickle.load(f)
    
with open(PATH_to_DATA+'cl_nsub{}_nside256_r{:.3f}_S4_d1_nsidefgb8_fullpipeline_corr2000_500reals.pkl'.format(Nf, myr), 
              'rb') as f:
    data = pickle.load(f)

def give_allcl(allr, ndim, ll, ell, Alens, covariance):
    alldl = np.zeros((len(allr), ndim))
    cov = np.zeros((len(allr), ndim))
    for ii, i in enumerate(allr):
        #print(i)
        dli = cl2dl(ell, _get_Cl_cmb(Alens, i)[2, :1000])
        alldl[ii] = np.interp(ll, ell, dli)
        cov[ii] = np.diag(covariance(myBBth(data['leff'], i, Alens)))
        
    return alldl, cov
def give_r_sigr(r, like):
    maxL = r[like == like.max()]
    
    cumint = scipy.integrate.cumtrapz(like, x=r)
    cumint = cumint / np.max(cumint)
    onesigma = np.interp(0.68, cumint, r[1:])
    if len(maxL) != 1:
        maxL = maxL[0]
    
    return maxL, onesigma

rv=np.linspace(-0.001, 0.1, 600)
n=500
nsub=[1]
r=np.zeros((len(nsub), n))
sigr=np.zeros((len(nsub), n))

for jj, j in enumerate(nsub):
    #error = np.cov(xd1['mycl'][jj], rowvar=False)
    error=np.std(xd1['mycl'], axis=1)[jj, :-1]#np.cov(xd1['mycl'][jj], rowvar=False)
    _, _, sigrmean, _, rmean = myMCMC.JCHlike(myBBth, 256).explore_like(data['leff'][:-1], 
                                         np.mean(data['mycl'], axis=1)[jj, :-1], 
                                         error, 
                                         rv,
                                         otherp=0.95)
    print()
    print(rmean, sigrmean, sigrmean-rmean)
    print()

    for i in range(n):
        _, _, s, rr = myMCMC.JCHlike(myBBth, 256).explore_like(data['leff'][:-1], data['mycl'][jj, i, :-1], error, rv)
        sigr[jj, i], r[jj, i] = s, rr[0]
        print(r[jj, i])
mypath='/home/regnier/work/regnier/forecast_deco/results/new_data/'
mydict = {'r': r, 'sigr': sigr, 'rv':rv}
output = open(mypath+'allr_nsub{:.0f}_nside256_d1_r{:.3f}.pkl'.format(Nf, myr), 'wb')
pickle.dump(mydict, output)
output.close()
