# QUBIC packages
import qubic
import sys
import os
sys.path.append('/Users/mregnier/Desktop/Libs/qubic/qubic/scripts/MapMaking')

import component_acquisition as Acq
import pickle
import gc

# Display packages
import healpy as hp
import matplotlib.pyplot as plt

# FG-Buster packages
import component_model as c

# General packages
import numpy as np
import warnings
from qubic import QubicSkySim as qss

from scipy.optimize import minimize
import ComponentsMapMakingTools as CMM
from functools import partial
import time
import configparser

from pyoperators import MPI

# PyOperators packages
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

C_1degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(1))
C_2degree = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(2))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

warnings.filterwarnings("ignore")
path = '/Users/mregnier/Desktop/PhD Regnier/MapMaking/bash'

seed = 1#int(sys.argv[1])
iteration = 1#int(sys.argv[2])

### Reading and loading configuration file
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Itérer sur chaque section et option
    external = []
    allnus = [30, 44, 70, 100, 143, 217, 353]
    k = 0
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)
                
            if section == 'EXTERNAL DATA':
                if value.lower() == 'true':
                    external.append(allnus[k])
                
                k+=1

            # Convertir la valeur en liste si elle est de la forme "1, 2, 3"
            if ',' in value:
                value = [x.strip() for x in value.split(',')]

            # Convertir la valeur en int, float ou bool si c'est possible
            elif value.isdigit():
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.')):
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            # Définir chaque option de configuration en tant que variable globale
            globals()[option] = value
            
    return external
def get_ultrawideband_config():
    
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave

nu_ave, delta_nu_over_nu = get_ultrawideband_config()
#########################################################################################################
############################################## Arguments ################################################
#########################################################################################################

external = load_config('config.ini')

if nside_fit != 0:
    raise TypeError('You must have to put nside_fit = 0 (constant spectral index)')


print('************ Configuration of the simulation ************\n')
print('Instrument      :')
print(f'    Type       : {type}')
print(f'    Nsub       : {nsub}')
print(f'    Pointings  : {pointing}')
print(f'    Noise      : {noisy}\n')
print('Pixelization    :')
print(f'    Nside      : {nside}\n')
print('Foregrounds     :')
print(f'    Seed       : {seed}')
print(f'    Iteration  : {seed}')

save_each_ite = f'{type}_seed{seed}_iteration{iteration}'
job_id = os.environ.get("SLURM_JOB_ID")
unique_id = f"{type}_wnoise{str(w_det)+str(w_pho_150)+str(w_pho_220)}_seed{seed}_iteration{iteration}"#"job_" + str(job_id) + f"seed{seed}"
#os.makedirs(unique_id)
path_to_save = str(unique_id)

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################

comp = []
comp_name = []
if cmb :
    comp.append(c.CMB())
    comp_name.append('CMB')
if dust[0].lower() == 'true':
    comp.append(c.Dust(nu0=nu0_d, temp=temp))
    comp_name.append('DUST')
if synchrotron[0].lower() == 'true':
    comp.append(c.Synchrotron(nu0=nu0_s, beta_pl=-3))                     # We remove a template of synchrotron emission -> fixing the spectral index
    comp_name.append('SYNCHROTRON')
if coline[0].lower() == 'true':
    comp.append(c.COLine(nu=float(coline[2])/1e9, active=False))
    comp_name.append('CO')

d, center = CMM.get_dictionary(nsub, nside, pointing, nu_ave, duration)
d['nprocs_instrument']=size
d['nprocs_sampling']=1
d['comm']=None#comm
d['filter_nu'] = nu_ave * 1e9
d['nf_recon'] = 1
d['filter_relative_bandwidth'] = delta_nu_over_nu
d['type_instrument'] = 'wide'

#########################################################################################################
############################################## Acquisitions #############################################
#########################################################################################################

# QUBIC Acquisition

myqubic = Acq.QubicFullBand(d, Nsub=nsub, comp=comp, kind=type)

#stop
#if type == 'Wide':
#    myqubic = Acq.QubicUltraWideBandComponentsMapMaking(d, Nsub=nsub, comp=comp)
#elif type == 'Two':
#    myqubic = Acq.QubicDualBandComponentsMapsMaking(d, Nsub=nsub, comp=comp)


isco = coline[0].lower() == 'true'

if isco == False:
    nu_co = None
else:
    nu_co = float(coline[2])

#qubic150 = myqubic.qubic150
#qubic220 = myqubic.qubic220
#qubic150 = Acq.QubicIntegratedComponentsMapMaking(d150, Nsub=int(nsub/2), comp=comp)
#qubic220 = Acq.QubicIntegratedComponentsMapMaking(d220, Nsub=int(nsub/2), comp=comp)
coverage = myqubic.get_coverage()
pixok = coverage/coverage.max() > thr



# Add external data
allexp = Acq.QubicOtherIntegratedComponentsMapMaking(myqubic, external, comp=comp, nintegr=nintegr)

# Input beta
beta=np.array([1.54])

array_of_operators = myqubic._get_array_operators(beta, convolution=False, list_fwhm=None)
array_of_operators150 = array_of_operators[:nsub]
array_of_operators220 = array_of_operators[nsub:2*nsub]

H = allexp.get_operator(beta, convolution)


#########################################################################################################
############################################## Components ###############################################
#########################################################################################################

dcomp = {}
if cmb:
    dcomp['cmb'] = seed
if dust[0].lower() == 'true':
    dcomp['dust'] = str(dust[1])
if synchrotron[0].lower() == 'true':
    dcomp['synchrotron'] = str(synchrotron[1])
if coline[0].lower() == 'true':
    dcomp['coline'] = str(coline[1])

components = myqubic.get_PySM_maps(dcomp)


# invN
invN = allexp.get_invntt_operator()
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))


#########################################################################################################
############################################## Systematics ##############################################
#########################################################################################################

# Input gain

#if type == 'Two':
#    gdet150 = np.random.normal(float(varg150[0]), float(varg150[1]), (992))
#    gdet150 /= gdet150[0]
#    gdet220 = np.random.normal(float(varg220[0]), float(varg220[1]), (992))
#    gdet220 /= gdet220[0]
#elif type == 'Wide':
#    gdet = np.random.normal(float(varg150[0]), float(varg150[1]), (992))
#    gdet /= gdet[0]

#if type == 'Wide':
#    g = np.array([gdet]).copy()
#elif type == 'Two':
#    g = np.array([gdet150, gdet220])
#else:
#    raise TypeError('Not right band')

#########################################################################################################
############################################## Reconstruction ###########################################
#########################################################################################################

if convolution:
    myfwhm = np.sqrt(myqubic.allfwhm**2 - np.min(myqubic.allfwhm)**2)
else:
    myfwhm = None

print(f'FWHM for Nsub : {myfwhm}')

# Get reconstruction operator
Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm)

# Get simulated data
tod = H(components)

if noisy:
    n = allexp.get_noise().ravel()
    tod += n.copy()

if convolution:
    tod = allexp.reconvolve_to_worst_resolution(tod)


#tod = allexp.get_observations(beta, g, components, convolution=convolution, noisy=noisy, nu_co=nu_co)

if type == 'Two':
    tod_150 = tod[:(myqubic.Ndets*myqubic.Nsamples)]
    tod_220 = tod[(myqubic.Ndets*myqubic.Nsamples):(myqubic.Ndets*myqubic.Nsamples*2)]
    tod_external = tod[((myqubic.Ndets*myqubic.Nsamples)*2):]
elif type == 'Wide':
    
    tod_w = tod[:(myqubic.Ndets*myqubic.Nsamples)]
    tod_external = tod[((myqubic.Ndets*myqubic.Nsamples)):]



if convolution:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=myqubic.allfwhm[-1], lmax=2*nside-1)
else:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=0.0, lmax=2*nside-1)

target = np.sqrt(myqubic.allfwhm[0]**2 - myqubic.allfwhm[-1]**2)
C_target = HealpixConvolutionGaussianOperator(fwhm=target)
### We can make the hypothesis that Planck's astrophysical foregrounds are a good starting point. We assume no prior on the CMB.
comp_for_pcg = components.copy()
rr = np.random.randn(12*nside**2, 3)
for i in range(len(comp)):

    if comp_name[i] == 'CMB':
        comp_for_pcg[i] = Ctrue(components[i]) * (np.random.randn(12*nside**2, 3)*8)
    elif comp_name[i] == 'DUST':
        comp_for_pcg[i] = Ctrue(components[i])
    elif comp_name[i] == 'SYNCHROTRON':
        comp_for_pcg[i] = Ctrue(components[i])
    elif comp_name[i] == 'CO':
        comp_for_pcg[i] = Ctrue(components[i])
    else:
        raise TypeError(f'{comp_name[i]} not recognize')


#########################################################################################################
############################################## Main Loop ################################################
#########################################################################################################


kmax=3000
k=0
beta_i = beta.copy()
#g_i = g.copy()
components_i = comp_for_pcg.copy()


def chi2_wide(x, solution):
    tod_s_i = tod_w.copy() * 0
    R = ReshapeOperator(((1,12*nside**2,3)), ((12*nside**2,3)))
    #G_w = DiagonalOperator(gw, broadcast='rightward')

    k=0
    for ii, i in enumerate(array_of_operators):
    
        A = CMM.get_mixing_operator(x, nus=np.array([myqubic.allnus[k]]), comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
            
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    
    tod_150_norm = tod_w#/tod_150.max()#/np.std(tod_150)
    tod_s_i_norm = tod_s_i#/tod_s_i.max()#/np.std(tod_s_i)

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_150(x, solution):

    tod_s_i = tod_150.copy() * 0
    R = ReshapeOperator(((1,12*nside**2,3)), ((12*nside**2,3)))
    #G150 = DiagonalOperator(g150, broadcast='rightward')
    k=0
    #print('len 150 = ', len(array_of_operators150))
    for ii, i in enumerate(array_of_operators150):
        #print(ii)
        A = CMM.get_mixing_operator(x, nus=np.array([myqubic.allnus[k]]), comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
        
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    
    tod_150_norm = tod_150#/tod_150.max()#/np.std(tod_150)
    tod_s_i_norm = tod_s_i#/tod_s_i.max()#/np.std(tod_s_i)

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_220(x, solution):

    #G220 = DiagonalOperator(g220, broadcast='rightward')
    tod_s_ii = tod_220.copy() * 0
    R = ReshapeOperator(((1,12*nside**2,3)), ((12*nside**2,3)))

    k=0
    #print('len 220 = ', len(array_of_operators220))
    for ii, i in enumerate(array_of_operators220):
        #print(ii)
        mynus = np.array([myqubic.allnus[k+int(nsub)]])
        A = CMM.get_mixing_operator(x, nus=mynus, comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
        tod_s_ii += Hi(solution[ii+int(nsub)]).ravel()
        k+=1
        

    tod_220_norm = tod_220#/tod_220.max()
    tod_s_ii_norm = tod_s_ii#/tod_s_ii.max()
    return np.sum((tod_220_norm - tod_s_ii_norm)**2)
def chi2_external(x, solution):

    tod_s_i = tod_external.copy() * 0

    Hexternal = Acq.OtherData(external, nside, comp).get_operator(nintegr=nintegr, beta=x, convolution=False, myfwhm=None, nu_co=nu_co)

    tod_s_i = Hexternal(solution[-1])

    
    tod_external_norm = tod_external#CMM.normalize_tod(tod_external, external, 12*nside**2)
    tod_s_i_norm = tod_s_i#CMM.normalize_tod(tod_s_i, external, 12*nside**2)

    return np.sum((tod_external_norm - tod_s_i_norm)**2)
def chi2_tot(x, solution):
    xi2_external = chi2_external(x, solution)
    if type == 'Two':
        xi2_150 = chi2_150(x, solution)
        xi2_220 = chi2_220(x, solution)
        return xi2_150 + xi2_220 + xi2_external
    elif type == 'Wide':
        xi2_w = chi2_wide(x, solution)
        return xi2_w + xi2_external
    
    #print(x, xi2_150, xi2_220, xi2_external)
    


'''
if save_each_ite is not None:
                
    dict_i = {'maps':components, 'initial':comp_for_pcg, 'beta':beta, 'gain':g, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':1,
              'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust, 'execution_time':0}

    output = open(path_to_save+'/Iter0_maps_beta_gain_rms_maps.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()
'''
del H
gc.collect()

while k < kmax :

    #####################################
    ######## Pixels minimization ########
    #####################################

    H_i = allexp.update_A(Hrecon, beta_i)
    #if type == 'Wide':
    #    H_i = allexp.update_systematic(H_i, newG=g_i[0], co=isco)
    #elif type == 'Two':
    #    H_i = allexp.update_systematic(H_i, newG=g_i, co=isco)

    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod
    
    ### PCG
    solution = pcg(A, b, M=M, tol=float(tol), x0=components_i, maxiter=int(maxite), disp=True)

    s = comm.allreduce(solution['x'], op=MPI.SUM) / size
    print(s.shape)
    plt.figure(figsize=(15, 5))
    hp.gnomview(C_target(components[0, :, 1]), rot=center, reso=15, cmap='jet', min=-6, max=6, sub=(1, 3, 1))
    hp.gnomview(C_target(s[0, :, 1]), rot=center, reso=15, cmap='jet', min=-6, max=6, sub=(1, 3, 2))
    hp.gnomview(C_target(s[0, :, 1]) - C_target(components[0, :, 1]), rot=center, reso=15, cmap='jet', min=-6, max=6, sub=(1, 3, 3))
    plt.savefig(f'Iter{k+1}.png')
    plt.close()
    
    ### Synchrotron is assumed to be a template removed to the TOD
    #if synchrotron[0].lower() == 'true':
    #    index = comp_name.index('SYNCHROTRON')
    #    C = HealpixConvolutionGaussianOperator(fwhm=np.min(myqubic.allfwhm))
    #    solution['x'][index] = C(components[index]).copy()

    ### Compute spectra
    components_i = solution['x'].copy()

    #spectra_cmb = s.get_observed_spectra(solution['x'][0].T)
    #spectra_dust = s.get_observed_spectra(solution['x'][1].T)

    components_for_beta = np.zeros((2*nsub, len(comp), 12*nside**2, 3))

    ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
    for i in range(2*nsub):
        for jcomp in range(len(comp)):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm = myfwhm[i], lmax=2*nside-1)
            else:
                C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=2*nside-1)
            components_for_beta[i, jcomp] = C(components_i[jcomp])

    ###################################
    ######## Gain minimization ########
    ###################################
    
    #if type == 'Wide':
    #    H_ii = allexp.update_systematic(Hrecon, newG=np.random.randn(myqubic.number_FP, myqubic.Ndets)*0+1.000000000000001, co=isco)
    #    g_i = CMM.get_gain_detector(H_ii, components_for_beta[-1], tod_w, myqubic.Nsamples, myqubic.Ndets, myqubic.number_FP)
    #elif type == 'Two':
    #    H_ii = allexp.update_systematic(Hrecon, newG=np.random.randn(myqubic.number_FP, myqubic.Ndets)*0+1.000000000000001, co=isco)
    #    g_i = CMM.get_gain_detector(H_ii, components_for_beta[-1], tod, myqubic.Nsamples, myqubic.Ndets, myqubic.number_FP)
    
    
    
    #if myqubic.number_FP == 2:
    #    g_i[0] /= g_i[0, 0]
    #    g_i[1] /= g_i[1, 0]
    #else:
    #    g_i /= g_i[0]
    #    g_i = np.array([g_i])
    #print(g_i.shape)
    #print(g_i[0, :5])
    #print(g[0, :5])
    
    #print(np.mean(g_i[:5]-g[:5], axis=1))

    ###################################
    ######## Beta minimization ########
    ###################################

    ### We define new chi^2 function for beta knowing the components at iteration i
    if type == 'Wide':
        chi2 = partial(chi2_tot, solution=components_for_beta)#, g150=g_i[0], g220=None)
    elif type == 'Two':
        chi2 = partial(chi2_tot, solution=components_for_beta)#, g150=g_i[0], g220=g_i[1])
    ### Doing minimization
    beta_i = minimize(chi2, x0=np.array([1.5]), method=str(method), tol=1e-4).x
    
    print(beta_i)
    
    ### Saving components, beta, gain, convergence, etc.. for each iteration
    #if save_each_ite is not None:
    #    dict_i = {'maps':components_i, 'beta':beta_i, 'gain':g_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error'],
    #              'spectra_cmb':spectra_cmb, 'spectra_dust':spectra_dust}
    #
    #    output = open(path_to_save+'/Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
    #    pickle.dump(dict_i, output)
    #    output.close()


    k+=1



