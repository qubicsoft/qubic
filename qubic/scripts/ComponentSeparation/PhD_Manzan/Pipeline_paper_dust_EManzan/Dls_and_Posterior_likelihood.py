#This is the .py version of the Notebook that I use on my pc called "Compare_instruments_v2.ipynb". It performs Dls comparison between instruments and computes the posterior likelihood

import numpy as np
import pickle
import glob

import qubic
from qubic import NamasterLib as nam
#from qubic import fibtools as ft
from qubic import camb_interface as qc
import healpy as hp
import scipy
from scipy.optimize import curve_fit

from pylab import *

import fgbuster

import likelihoodlib2

#FUNCTION DEFINITION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
def read_and_save_pickle_cls(path_to_file):
    #open file and save its content ad an object
    input_file = open(path_to_file, 'rb')
    obj_file = pickle.load(input_file)
    print('Loading and saving data from file: {}'.format(path_to_file))
    #assign the file content to a proper variable
    
    ell = obj_file[0]
    Dls = obj_file[1]
    #ell = obj_file['leff'][:-1]
    #Dls = obj_file['mycl'][0,:,:-1]
    print('ell shape = ', ell.shape)
    print('Dls shape = ', Dls.shape)
    
    return ell, Dls

#-------------------------------------------------------------------------------

N = 500 #100           # Number of iterations saved in the pickle file
instrum_config_list = 'S4,BI3,BI5,BI7' # how many instrument you want to analyze: 'S4', 'BI' or 'S4','BI'
N_sample_band = 100 # number of freqs used in the bandpass integration. At least 1
Nside_fit = 8 #0 #8 # nside of the pixel where the parameters are independently estimated
r = 0. #003 # r value
Alens = 0.1 #amplitude of lensing residual. Alens=0: no lensing. Alens=1: full lensing
corr_l = 0#10. #10. #100. #10.
call_number = 5 #1 #

apo_size_deg = 10. #5. #10.0

#sky and instrument definition -----------------------
seed = 42 #cmb seed
skyconfig = {'cmb':seed,'pysm_fg':['d1','s1']} #{'cmb':seed,'pysm_fg':['d6','s1']} #{'cmb':seed,'pysm_fg':['d6','s1']}
skyconfig_toimport = {'cmb':seed,'pysm_fg':['d1','s1']}
print('Sky config in use: ', skyconfig)
instrum_config = [i for i in instrum_config_list.split(',')] #e.g.: ['S4','BI']
print('Instruments in use: ', instrum_config)

#if there is no decorrelation, put corr_l to None
if corr_l == 0.:
    corr_l = None

#--------------------------------------------
#def variables for likelihood part

fsky=0.03 #CMB-S4 sky fraction
nside=256
center_ra_dec = [-30,-30] #CMB-S4 sky patch

lmin=21  #40
delta_ell=35  #30
lmax = 335 #2*nside-1

num_ell = round((lmax-lmin)/delta_ell) -3 #1 #takes into account the fact that we discard one or more bins (junk bins)
print('Number of bins used: ', num_ell) 

Niter = N
rv_vec = np.linspace(-0.001, 0.01, 600) #np.linspace(-0.001, 0.01, 600) #np.linspace(-0.001, 0.01, 600) #np.linspace(-0.01,0.01,200) #np.linspace(-0.001, 0.01, 600) #np.linspace(-0.01,0.01,200)
matrix = 'diagonal' #'dense' #'diagonal'
Add_knox_covar_externally = True

global_dir = './'   # '/sps/qubic/Users/emanzan/libraries/qubic/qubic'
camblib_filename = 'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle' #'camblib_with_r_from_0.0001.pickle' #  #'/doc/CAMB/camblib.pkl' 

#filename definition -----------------------
file_name_to_load_cls = []

#define a str with the simulated components to be used in title/name of the saved images/files
file_name = '{}_tesconf2_Comp'.format(call_number)
#file_name = '{}_template2_Comp'.format(call_number)
#file_name = '{}_testapo{}_Comp'.format(call_number, apo_size_deg)
#file_name = '{}_template_apo{}_Comp'.format(call_number, apo_size_deg)

if file_name == '{}_template5_Comp'.format(call_number) or file_name == '{}_tesconf2_Comp'.format(call_number) or file_name == '{}_testMathiasconf_Comp'.format(call_number) or file_name == '{}_template6_Comp'.format(call_number):
    CMB_from_Planck = True
    CMB_from_CAMB = False
else:
    CMB_from_Planck = False
    CMB_from_CAMB = True
    
if file_name == '{}_template4_Comp'.format(call_number):
    lmin=40
    delta_ell=30
    lmax = 355

for i in skyconfig.keys():
    file_name+='_{}'.format(skyconfig[i])
    
file_name += '_{}_{}_{}'.format(r, Alens, corr_l)
file_name += '_'

for i in range(len(instrum_config)):
    
    filename_cls = 'VaryingCMBseed_'
    #filename_cls = 'VaryingCMBandFG_'
    filename_cls += file_name+instrum_config[i]+"_nsidefit{}_Nbpintegr{}_iter{}".format(Nside_fit, N_sample_band, N)
    path_to_file_cls = './results/cls_'+filename_cls+'.pkl'
    #filename_cls = 'cl_nsub1_nside256_r0.000_S4_d1_nsidefgb8_fullpipeline_corr2000_500reals_v2'
    #path_to_file_cls = './results/'+filename_cls+'.pkl'
    file_name_to_load_cls.append(path_to_file_cls)
    
out_filename = file_name+'{}_nsidefit{}_Nbpintegr{}_iter{}'.format(instrum_config, Nside_fit, N_sample_band, N)
#out_filename = 'ResultsEleniaCode'+filename_cls
print('Saving to file: ', out_filename)
    
colors = ['r', 'orange', 'g', 'b']

ell = []
DlsBB = []
for i in range(len(instrum_config)):
    l, Dls = read_and_save_pickle_cls(file_name_to_load_cls[i])
    ell.append(l[:-1]) #[:-1] #[1:-1]
    DlsBB.append(Dls[:,:-1,2])#[:,:-1,2] #[:,:,2]
    
ell_array = np.array(ell)
DlsBB_array = np.array(DlsBB)

print('Shapes check: ', ell_array.shape, DlsBB_array.shape)
print('Ell used: ', ell_array[0])

outputfile = './results/Dls_and_ell_'+out_filename+'.pkl'
pickle.dump({'ell_vec': ell_array, 'Dls_BB': DlsBB_array}, open(outputfile, 'wb'))

#the Dls useful for comparison
if CMB_from_CAMB:
    covmap = likelihoodlib2.get_coverage(fsky, nside, center_radec=center_ra_dec) 
    DlsBB_theoretic_r = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], r, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_r005 = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 0.005, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_r01 = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 0.01, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_r03 = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 0.03, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_r1 = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 0.1, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_r5 = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 0.5, Alens, coverage=covmap, nside=nside)
    DlsBB_theoretic_runit = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], 1, Alens, coverage=covmap, nside=nside)
    
if CMB_from_Planck:
    ell_pl = np.arange(ell_array[0,0],ell_array[0,-1]+2,1, dtype=int)
    print(ell_pl, ell_pl.shape)
    ClsBB_theoretic_r_planck = likelihoodlib2.planck_cls(r, Alens)[2,:]
    print(ClsBB_theoretic_r_planck.shape)
    DlsBB_theoretic_r_planck = likelihoodlib2.ClsXX_2_DlsXX_binned(ell_pl, ClsBB_theoretic_r_planck) #[ell_pl[0]:ell_pl[-1]+1+2])
    
    
#plot Dls and theo. Dls for comparison-----------------------------------------------------------------------------------------------------------  
figure(figsize=(15,15))

for i in range(len(instrum_config)): #
    subplot(2,2,i+1)
    
    for j in range(N):
        plot(ell_array[i,:], DlsBB_array[i,j,:], color='k', linewidth=0.1, alpha=0.5) 
        if N > 1:
            mean = np.mean(DlsBB_array[i,:,:], axis=0)
            sigma = np.std(DlsBB_array[i,:,:], axis=0)#/2 #np.sqrt(2)
            errorbar(ell_array[i,:], mean, yerr=sigma, color='r', linewidth=2, marker='o', markersize=4, alpha=0.5, capsize=10)
    if CMB_from_Planck:
        plot(ell_pl, DlsBB_theoretic_r_planck, color='orange', linewidth=1, label = 'Theo. BB spectrum for input r = {} from Planck'. format(r))
    if CMB_from_CAMB:
        plot(ell_array[i,:], DlsBB_theoretic_r, color='orange', linewidth=2, label = 'Theo. BB spectrum for input r = {}'. format(r))
    xlabel('$\ell$', fontsize=16)
    ylabel('$D_{{\ell}}$', fontsize=16)
    title(instrum_config[i], fontsize=16)
    #xscale('log')
    yscale('log')
    #xlim([0,0.03])
    legend(loc='best', fontsize=16)
    tick_params(axis='both', which='major', labelsize=16)
tight_layout()
savefig('./Cls_Likelihood_images/Sim_vs_theo_Dls_'+out_filename+'.png')



#posterior likelihood---------------------------------------------------------------------------------------------------------------------
cambfilepath = global_dir + camblib_filename
#coverage = likelihoodlib2.get_coverage(fsky, nside, center_ra_dec)
#----------------Mathias changes-----------------------------------------
coverage = None
lmax = 355
#----------------End Mathias changes-----------------------------------------
theo_dls_class = likelihoodlib2.binned_theo_Dls(Alens, nside, coverage, cambfilepath, lmin=lmin, delta_ell=delta_ell, lmax_used=lmax)
if CMB_from_Planck:
    print('CMB generated from Planck best fit')
    theo_dls = theo_dls_class.get_DlsBB_Planck
elif CMB_from_CAMB:
    print('CMB generated from CAMB')
    theo_dls = theo_dls_class.get_DlsBB #get_cls(0.1, get_binned_camblib(256, [-30,-30], coverage)).myBBth
else:
    print('Specify where the CMB was generated from!')

knox_covar = theo_dls_class.sample_var
if Add_knox_covar_externally:
    covariance_model_funct = knox_covar #None #knox_covar
else:
    covariance_model_funct = None
print('Use covariance_model_function = ',  covariance_model_funct)

#---------------REMOVE THIS!!!!!!--------------------
#exit()
#-----------------------------------
if matrix == 'dense':
    Ncov = np.zeros((len(instrum_config), ell_array.shape[-1],ell_array.shape[-1]))
elif matrix == 'diagonal':   
    Ncov = np.zeros((len(instrum_config), ell_array.shape[-1]))
    
#figure(figsize=(15,15))
for i in range(len(instrum_config)):
    if matrix == 'dense':
        Ncov[i] = np.cov(DlsBB_array[i].T) #noise covariance matrix
        #invN = np.linalg.inv(Ncov) #invert N
        print(DlsBB_array[i].shape, Ncov[i].shape)
    elif matrix == 'diagonal':
        Ncov[i] = np.std(DlsBB_array[i], axis = 0)
        #sigma = np.std(DlsBB_array[i], axis = 0) #/np.sqrt(2) #np.sqrt(2)*
        #Ncov[i] = np.zeros((len(sigma),len(sigma)))
        #np.fill_diagonal(Ncov[i], sigma ** 2)
        #invN = np.linalg.inv(Ncov) #invert N
        print(DlsBB_array[i].shape, Ncov[i].shape)#, sigma.shape)
    
    subplot(2,2,i+1)
    imshow(Ncov[i], cmap='RdBu')
    colorbar()
    xlabel('$\ell$')
    ylabel('$\ell$')
    ell_to_use = np.round(ell_array[i], 1) 
    xticks(np.arange(len(ell_array[i])), labels=[str(ell_to_use[i]) for i in range(len(ell_to_use))])
    yticks(np.arange(len(ell_array[i])), labels=[str(ell_to_use[i]) for i in range(len(ell_to_use))])
    title('{} : Noise covariance matrix'.format(instrum_config[i]), y=1.1)
savefig('./Cls_Likelihood_images/Noise_cov_matrix_'+out_filename+'.png')


#likelihood
like2 = np.zeros((len(instrum_config), Niter, len(rv_vec)))
#log like eval
for k in range(len(instrum_config)): #len(instrum_config)
    print('Doing {}'.format(instrum_config[k]))
    for i in range(Niter): #Niter
        print('iter ', i+1)
        likelihood_class = qubic.mcmc.LogLikelihood(xvals=ell_array[k], yvals=DlsBB_array[k,i],
                                                    errors=Ncov[k], model = theo_dls,
                                                    flatprior=[[-1,1]], covariance_model_funct=covariance_model_funct)
        for j in range(len(rv_vec)):
            like2[k,i,j] = np.exp(likelihood_class([rv_vec[j]]))
            #(- 0.5) * (((estim_dls[i] - theo_dls(ell,rv_vec[j])).T @ invN) @ (estim_dls[i] - theo_dls(ell,rv_vec[j])))

            
#plots------------------------------------------------------------------------------------------------------------------------------------
color = cm.jet(np.linspace(0,1,Niter))
norm = mpl.colors.Normalize(vmin=1, vmax=Niter)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

figure(figsize=(15,15))
for j in range(len(instrum_config)): #len(instrum_config)
    subplot(2,2,j+1)
    for i in range(N): #Niter
        plot(rv_vec, like2[j,i], color=color[i], linewidth=0.5)  
    #vlines(0., 0, 1, color='k', linestyle='--')
    xlabel('r', fontsize=26)
    ylabel('likelihood', fontsize=26)
    title('{} likelihood'.format(instrum_config[j]), fontsize=22)
    xlim([-0.001,0.005])
    grid()
#ylim([0,0.6])
tight_layout()
c = colorbar(cmap)
c.set_label('N iter', fontsize=26)
savefig('./Cls_Likelihood_images/All_likelihoods_'+out_filename+'.png')

outputfile = './results/Reconstructed_r_'+out_filename+'.pkl'
pickle.dump({'r_vec': rv_vec, 'likelihood': like2}, open(outputfile, 'wb'))

#histogram
figure(figsize=(16,16))

ML_r = []
for i in range(len(instrum_config)): #len(instrum_config)
    #bin the data
    ML_r.append(rv_vec[np.argmax(like2[i],axis=1)])
    #print(ML_r[i])
    data_bins, bins = np.histogram(ML_r[i], bins=20, density=False) #
        #print(data_bins)
    #def array with the center of the bins, so that it has the same shape as binned data
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    mean = np.mean(ML_r[i])
    std = np.std(ML_r[i])
    
    #plot histogram    
    bar(binscenters, data_bins, width=bins[1] - bins[0], alpha=1-0.1*i,
        label = '{}: r = {:.1E}$\pm$ {:.1E}'.format(instrum_config[i], mean, std), color=colors[i])
    
vlines(r, 0, np.max(data_bins), linestyles ="dashed", colors ="g", label='Input r value = {}'.format(r))
legend(fontsize=25, loc='center right') #loc='center left', bbox_to_anchor=(1, 0.5))
xlabel('Best fit r', fontsize=25)
ylabel('counts', fontsize=25)
#xlim([-0.0025, 0.0025])
#xlim([-0.005, 0.005])
xlim([-0.004, 0.004])
tick_params(axis='both', which='major', labelsize=25)
title_str='{} iterations,  integration points = {}, fg = {}, r = {}, $A_\mathrm{{lens}}$ = {}, $\ell_\mathrm{{corr}}$ = {}'.format(Niter, N_sample_band, skyconfig['pysm_fg'], r, Alens, corr_l)
suptitle(title_str, fontsize=14, y=1)
savefig('./Cls_Likelihood_images/Best_fit_r_distribution_'+out_filename+'.png')

outputfile = './results/Reconstructed_r_'+out_filename+'.pkl'
pickle.dump({'r_vec': rv_vec, 'likelihood': like2, 'Best_fit_r_d1s1': np.array(ML_r)}, open(outputfile, 'wb'))