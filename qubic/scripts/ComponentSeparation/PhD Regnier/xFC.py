import forecast_tools
import numpy as np
import pysm3
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import qubicplus
import qubic
import os
import fgbuster
import scipy
import pysm3.units as u
from pysm3 import utils
from pysm3 import bandpass_unit_conversion
from qubic import camb_interface as qc
import sys
import qubic
import pickle
from qubic import mcmc
from qubic import NamasterLib as nam
from forecast_tools import cl2dl, _get_Cl_cmb
import os.path as op
CMB_CL_FILE = op.join('/pbs/home/m/mregnier/sps1/QUBIC+/forecast/Cls_Planck2018_%s.fits')

center = qubic.equ2gal(-30, -30)
# If there is not this command, the kernel shut down every time..
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_coverage(fsky, nside, center_radec=[-30, -30]):
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
def fct_subopt(nus):
    subnus = [150., 220]
    subval = [1.4, 1.2]
    fct_subopt = np.poly1d(np.polyfit(subnus, subval, 1))
    return fct_subopt(nus)
def qubicify(config, qp_nsub, qp_effective_fraction):
    nbands = np.sum(qp_nsub)
    qp_config = config.copy()
    for k in qp_config.keys():
        qp_config[k]=[]
    qp_config['nbands'] = nbands
    qp_config['fsky'] = config['fsky']
    qp_config['ntubes'] = config['ntubes']
    qp_config['nyears'] = config['nyears']
    qp_config['initial_band'] = []

    for i in range(len(config['frequency'])):
        #print(config['edges'][i][0], config['edges'][i][-1])
        newedges = np.linspace(config['edges'][i][0], config['edges'][i][-1], qp_nsub[i]+1)
        #print(newedges)
        newfreqs = (newedges[0:-1]+newedges[1:])/2
        newbandwidth = newedges[1:] - newedges[0:-1]
        newdnu_nu = newbandwidth / newfreqs
        newfwhm = config['fwhm'][i] * config['frequency'][i]/newfreqs
        scalefactor_noise = np.sqrt(qp_nsub[i]) * fct_subopt(config['frequency'][i])# / qp_effective_fraction[i]
        newdepth_p = config['depth_p'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_i = config['depth_i'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_e = config['depth_e'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newdepth_b = config['depth_b'][i] * np.ones(qp_nsub[i]) * scalefactor_noise
        newell_min = np.ones(qp_nsub[i]) * config['ell_min'][i]
        newnside = np.ones(qp_nsub[i]) * config['nside'][i]
        neweffective_fraction = np.ones(qp_nsub[i]) * qp_effective_fraction[i]
        initial_band = np.ones(qp_nsub[i]) * config['frequency'][i]

        for k in range(qp_nsub[i]):
            if qp_effective_fraction[i] != 0:
                qp_config['frequency'].append(newfreqs[k])
                qp_config['depth_p'].append(newdepth_p[k])
                qp_config['depth_i'].append(newdepth_i[k])
                qp_config['depth_e'].append(newdepth_e[k])
                qp_config['depth_b'].append(newdepth_b[k])
                qp_config['fwhm'].append(newfwhm[k])
                qp_config['bandwidth'].append(newbandwidth[k])
                qp_config['dnu_nu'].append(newdnu_nu[k])
                qp_config['ell_min'].append(newell_min[k])
                qp_config['nside'].append(newnside[k])

                qp_config['effective_fraction'].append(neweffective_fraction[k])
                qp_config['initial_band'].append(initial_band[k])
        edges=get_edges(np.array(qp_config['frequency']), np.array(qp_config['bandwidth']))
        qp_config['edges']=edges.copy()
        #for k in range(qp_nsubs[i]+1):
        #    if qp_effective_fraction[i] != 0:
        #        qp_config['edges'].append(newedges[k])
    fields = ['frequency', 'depth_p', 'depth_i', 'depth_e', 'depth_b', 'fwhm', 'bandwidth',
              'dnu_nu', 'ell_min', 'nside', 'edges', 'effective_fraction', 'initial_band']
    for j in range(len(fields)):
        qp_config[fields[j]] = np.array(qp_config[fields[j]])

    return qp_config
def get_edges(nus, bandwidth):
    edges=np.zeros((len(nus), 2))
    dnu_nu=bandwidth/nus
    edges_max=nus * (1. + dnu_nu/2)
    edges_min=nus * (1. - dnu_nu/2)
    for i in range(len(nus)):
        edges[i, 0]=edges_min[i]
        edges[i, 1]=edges_max[i]
    return edges
def combine_config(list_of_config):
    mynewnus=[]
    mynewdepth_i=[]
    mynewdepth_p=[]
    for conf in list_of_config:
        mynus=conf['frequency']
        mydepth=conf['depth_p']
        for j in range(len(mynus)):
            mynewnus.append(mynus[j])
            mynewdepth_i.append(1e3)
            mynewdepth_p.append(mydepth[j])
    dict={}
    dict['frequency']=np.array(mynewnus)
    dict['depth_i']=np.array(mynewdepth_i)
    dict['depth_p']=np.array(mynewdepth_p)
    return dict
def get_list_config(name, nsub):
    myconf=[]
    tab=np.arange(0, len(name), 1)
    for i in range(len(name)):
        if i % 2 == 0:
            with open('/pbs/home/m/mregnier/sps1/QUBIC+/forecast_decorrelation/{}_config.pkl'.format(name[i:i+2]), 'rb') as f:
                config = pickle.load(f)

                if name[i:i+2] == 'PL':
                    myconf.append(config)
                elif nsub!=1:
                    qp_effective_fraction=np.ones(config['nbands'])
                    qp_config=qubicify(config, np.ones(config['nbands']).astype(int)*nsub, qp_effective_fraction)
                    myconf.append(qp_config)
                else:
                    myconf.append(config)
    return myconf

covmap = get_coverage(0.03, 64)
pixok = covmap > 0

######################################################################
############################## Parameters ############################
######################################################################

N=int(sys.argv[1])
nside_out=int(sys.argv[2])
r=float(sys.argv[3])
correlation_length=int(sys.argv[4])

number_of_subbands=int(sys.argv[5])
dust_model=str(sys.argv[6])
name=str(sys.argv[7])
fsky=float(sys.argv[8])
if fsky == 0.01:
    radec = [-30, 320]
elif fsky == 0.03:
    radec = [-30, -30]

Alens=float(sys.argv[9])

myconf=get_list_config(name=name, nsub=number_of_subbands)
config=combine_config(myconf)



object=forecast_tools.ForecastMC(config, r=r, Alens=Alens, fsky=fsky, radec=radec)  # S4 -> [-30, -30]   BK -> [-30, 320]

if dust_model == 'd6':
    presets_dust={'correlation_length':correlation_length}
else:
    presets_dust={}








leff, clBB, like, maxL, sigma_r, CL95, cmb_est, seed, cl_fg, index_est = object.RUN_MC(N, dust_model=dust_model, dict_dust=presets_dust, nside_fgb=nside_out)



print()
print('Results : \n')

dltrue=cl2dl(leff, _get_Cl_cmb(Alens=Alens, r=r)[2, leff.astype(int)-1])
print(np.mean(clBB, axis=0)[:5])
print()
print('dl true -> ', dltrue[:5])

print()
print()
print('r = {:.10f} +/- {:.10f}'.format(maxL, sigma_r))
print()
print('95% C.L. : {:.6f}'.format(CL95))



mydict = {'leff':leff,
          'clBB':clBB,
          'seed':seed,
          'index_est':index_est,
          'CMB':cmb_est,
          'cl_fg':cl_fg,
          'likelihood':like,
          'CL95':CL95,
          'r':maxL,
          'sigma_r':sigma_r
        }
output = open('/pbs/home/m/mregnier/sps1/QUBIC+/forecast_decorrelation/results/'+
                'forecast_Alens{:.1f}_{}_model{}_decorrelation{}_nfsub{}_{}reals.pkl'.format(Alens, name, dust_model, correlation_length, number_of_subbands, N), 'wb')
pickle.dump(mydict, output)
output.close()
