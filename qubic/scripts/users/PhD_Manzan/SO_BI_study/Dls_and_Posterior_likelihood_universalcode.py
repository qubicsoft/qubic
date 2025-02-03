# This is the .py version of the Notebook that I use on my pc called "Compare_instruments_v2.ipynb". It performs Dls comparison between instruments and computes the posterior likelihood

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
    #-----------Mod to load mathias pkl----------------------
    #ell = obj_file['leff'][:-1]
    #Dls = obj_file['mycl'][0,:,:-1]
    #---------------------------------
    print('ell shape = ', ell.shape)
    print('Dls shape = ', Dls.shape)
    
    return ell, Dls

# -------------------------------------------------------------------------------

N = 500 #500 #100           # Number of iterations saved in the pickle file
all_nsub_conf = False # if True, sim all nsub configuration; if False, do only S4, BI3, BI5, BI7
if all_nsub_conf:
    instrum_config_list = 'SO,BI2,BI3,BI4,BI5,BI6,BI7,BI8' #'S4,BI3,BI5,BI7' # how many instrument you want to analyze: 'S4', 'BI' or 'S4','BI'
else:
    instrum_config_list = 'SO' #,BI3,BI5,BI7'
N_sample_band = 100 #100 #100 # number of freqs used in the bandpass integration. At least 1
Nside_fit = 4 #0 #8 # nside of the pixel where the parameters are independently estimated
r = 0.0 #003 # r value
Alens = 0.5 #amplitude of lensing residual. Alens=0: no lensing. Alens=1: full lensing
corr_l = 0. #float(sys.argv[1]) # 10. #10. #10. #100. #10.
call_number = 5 #5 #1 #

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

# --------------------------------------------
# def variables for likelihood part

fsky=0.1 #CMB-S4 sky fraction
nside=256
center_ra_dec = [0,-45] #CMB-S4 sky patch

lmin=30  #40
delta_ell=10  #30
lmax = 310 #2*nside-1

Niter = N
if skyconfig['pysm_fg'][0]=='d6' and corr_l < 50.:
    rv_vec = np.linspace(-0.005, 0.15, 1000)
    print('Defining r range: {} , {}, step {}'.format(np.min(rv_vec), np.max(rv_vec), len(rv_vec)))
elif r>0.:
    rv_vec = np.linspace(0, r+0.01, 600)
    print('Defining r range: {} , {}, step {}'.format(np.min(rv_vec), np.max(rv_vec), len(rv_vec)))
else:
    rv_vec = np.linspace(0.0, 0.01, 600)
    #rv_vec = np.linspace(-0.005, 0.01, 600) 
    print('Defining r range: {} , {}, step {}'.format(np.min(rv_vec), np.max(rv_vec), len(rv_vec)))

matrix = 'diagonal' #'dense' #'diagonal'
Add_knox_covar_externally = False #True
flatprior=[[-1,1]]
add_sqrt2_4cross_likelihood = True
if add_sqrt2_4cross_likelihood:
    mult_factor = np.sqrt(2)
else:
    mult_factor = 1
    
compute_like_on_all_iter = False

#filename definition -----------------------
file_name_to_load_cls = []
file_name_to_load_cls_for_noise = []

#define a str with the simulated components to be used in title/name of the saved images/files
if skyconfig['pysm_fg'][0]=='d6':
    file_name = '{}_PipelineFinalConfNoiseNew2_Comp'.format(call_number)
else:
    file_name = '{}_commonbaseline1_Comp'.format(call_number)
    #file_name = '{}_testmaskproxy_whnoisepol_Comp'.format(call_number)
    #file_name = '{}_testmask_whnoise_Comp'.format(call_number)
    #file_name = '{}_templatewithrealcoveragewhnoise_Comp'.format(call_number)
    #file_name = '{}_FinalConfNoiseNew2_Comp'.format(call_number)

#specify if theo model is from Planck spectra or CAMB
CMB_from_Planck = True
CMB_from_CAMB = False


global_dir = './'   # '/sps/qubic/Users/emanzan/libraries/qubic/qubic'
camblib_filename = 'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle' #'camblib_with_r_from_0.0001.pickle' #  #'/doc/CAMB/camblib.pkl' 
cambfilepath = global_dir + camblib_filename

    
for i in skyconfig.keys():
    file_name+='_{}'.format(skyconfig[i])

file_name += '_{}_{}_{}'.format(r, Alens, corr_l)
file_name += '_'

for i in range(len(instrum_config)):
    if skyconfig['pysm_fg'][0]=='d6':
        print('Define d1s1 file to load noise info')
        filename_cls = 'VaryingCMBseed_'
        
        filename_cls_for_noise = 'VaryingCMBseed_'
        filename_cls_for_noise += '{}_FinalConfNoiseNew2_Comp'.format(5)
        for j in skyconfig_toimport.keys():
            filename_cls_for_noise +='_{}'.format(skyconfig_toimport[j])
        filename_cls_for_noise += '_{}_{}_{}'.format(r, Alens, None)
        filename_cls_for_noise += '_'    
        filename_cls_for_noise += instrum_config[i]+"_nsidefit{}_Nbpintegr{}_iter{}".format(Nside_fit, 100, N)
        if all_nsub_conf:
            path_to_file_cls_for_noise = './results/cls_'+filename_cls_for_noise+'_allnsubconf.pkl'
        else:
            path_to_file_cls_for_noise = './results/cls_'+filename_cls_for_noise+'.pkl'
        file_name_to_load_cls_for_noise.append(path_to_file_cls_for_noise)
    else:
        print('Only load d1 file')
        filename_cls = 'VaryingCMBseed_'
    filename_cls += file_name+instrum_config[i]+"_nsidefit{}_Nbpintegr{}_iter{}".format(Nside_fit, N_sample_band, N)
    if all_nsub_conf:
        path_to_file_cls = './results/cls_'+filename_cls+'_allnsubconf.pkl'
    else:
        path_to_file_cls = './results/cls_'+filename_cls+'.pkl'
    
    #-----------Mod to load mathias pkl----------------------
    #filename_cls = 'cl_nsub1_nside256_r0.000_S4_d1_nsidefgb8_fullpipeline_corr2000_500reals_v2'
    #path_to_file_cls = './results/'+filename_cls+'.pkl'
    #---------------------------------
    
    file_name_to_load_cls.append(path_to_file_cls)

if all_nsub_conf:
    out_filename = file_name+'{}_nsidefit{}_Nbpintegr{}_iter{}_allnsubconf'.format(instrum_config, Nside_fit, N_sample_band, N)
else:
    out_filename = file_name+'{}_nsidefit{}_Nbpintegr{}_iter{}'.format(instrum_config, Nside_fit, N_sample_band, N)

# -----------Mod to load mathias pkl----------------------
# out_filename = 'ResultsEleniaCode'+filename_cls
# ---------------------------------

print('Saving to file: ', out_filename)

if len(instrum_config)==4:
    colors = ['r', 'orange', 'g', 'b']
else:
    colors = ['r', 'm', 'orange', 'y', 'g', 'c', 'b', 'purple']   

ell = []
DlsBB = []
for i in range(len(instrum_config)):
    l, Dls = read_and_save_pickle_cls(file_name_to_load_cls[i])
    ell.append(l[:-1]) #discard last junk bin
    DlsBB.append(Dls[:,:-1,2])
    #-----------Mod to load mathias pkl----------------------
    #ell.append(l) #[:-1] #[1:-1]
    #DlsBB.append(Dls)#[:,:-1,2] #[:,:,2]
    #---------------------------------   

ell_array = np.array(ell)
DlsBB_array = np.array(DlsBB)

print('Shapes check: ', ell_array.shape, DlsBB_array.shape)
print('Ell used: ', ell_array[0])

outputfile = './results/Dls_and_ell_'+out_filename+'.pkl'
pickle.dump({'ell_vec': ell_array, 'Dls_BB': DlsBB_array}, open(outputfile, 'wb'))

#load the Dls from d1 case to use it for the noise computation
if skyconfig['pysm_fg'][0]=='d6':
    DlsBB_for_noise = []
    print()
    print('Load noise from d1 case')
    for i in range(len(instrum_config)):
        l, Dls = read_and_save_pickle_cls(file_name_to_load_cls_for_noise[i])
        DlsBB_for_noise.append(Dls[:,:-1,2])#[:,:-1,2] #[:,:,2]  
    DlsBB_array_for_noise = np.array(DlsBB_for_noise)

#the Dls useful for comparison
if CMB_from_CAMB:
    covmap = likelihoodlib2.get_coverage(fsky, nside, center_radec=center_ra_dec) 
    DlsBB_theoretic_r = likelihoodlib2.get_DlsBB_from_CAMB(ell_array[0], r, Alens, coverage=covmap, nside=nside)

if CMB_from_Planck:
    ell_pl = np.arange(ell_array[0,0],ell_array[0,-1]+2,1, dtype=int)
    print(ell_pl, ell_pl.shape)
    ClsBB_theoretic_r_planck = likelihoodlib2.planck_cls(r, Alens)[2,:]
    print(ClsBB_theoretic_r_planck.shape)
    DlsBB_theoretic_r_planck = likelihoodlib2.ClsXX_2_DlsXX_binned(ell_pl, ClsBB_theoretic_r_planck) #[ell_pl[0]:ell_pl[-1]+1+2])


#plot Dls and theo. Dls for comparison-----------------------------------------------------------------------------------------------------------  

print('Number of NaNs: {}/{}'.format(np.count_nonzero(np.isnan(DlsBB_array[0,:,0])), DlsBB_array[0,:,0].shape[0]) )

negativeiter = (DlsBB_array[0,:,1] < 0) | (DlsBB_array[0,:,0] < 0) 

#figure(figsize=(15,15))
figure(figsize=(10,8))

for i in range(len(instrum_config)): #
    #subplot(3,3,i+1)
    if N==1:
        plot(ell_array[i,:], DlsBB_array[i,0,:], color='r', alpha=0.5, label='Recon Dls') 
    else:
        for j in range(N):
            plot(ell_array[i,:], DlsBB_array[i,j,:], color='orange', linewidth=0.1, alpha=0.5) 
            if N > 1:
                mean = np.mean(ma.masked_invalid(DlsBB_array[i,:,:]), axis=0) #ma.masked_invalid(DlsBB_array[i,:,:])
                sigma = np.std(ma.masked_invalid(DlsBB_array[i,:,:]), axis=0)*mult_factor 
                errorbar(ell_array[i,:], mean, yerr=sigma, color='r', linewidth=2, marker='o', markersize=4, alpha=0.5, capsize=10)
    if CMB_from_Planck:
        plot(ell_pl, DlsBB_theoretic_r_planck, color='k', linewidth=2, label = 'r = {}, Alens = {} from Planck'. format(r, Alens))
    if CMB_from_CAMB:
        plot(ell_array[i,:], DlsBB_theoretic_r, color='k', linewidth=2)#, label = 'Theo. BB spectrum for input r = {}'. format(r))
    xlabel('$\ell$', fontsize=20)
    ylabel('$D_{{\ell}}$', fontsize=20)
    title(instrum_config[i], fontsize=20)
    xscale('log')
    yscale('log')
    xlim([np.min(ell_pl),np.max(ell_pl)])
    legend(loc='best', fontsize=20)
    tick_params(axis='both', which='major', labelsize=20)
tight_layout()
savefig('./Cls_Likelihood_images/Sim_vs_theo_Dls_'+out_filename+'.png')

#exit()

#posterior likelihood---------------------------------------------------------------------------------------------------------------------
#coverage = likelihoodlib2.get_coverage(fsky, nside, center_ra_dec)
#----------------Mathias changes-----------------------------------------
coverage = None
#lmax = 355
#----------------End Mathias changes-----------------------------------------
theo_dls_class = likelihoodlib2.binned_theo_Dls(Alens, nside, coverage, cambfilepath, lmin=lmin, delta_ell=delta_ell, lmax_used=lmax)
if CMB_from_Planck:
    print('CMB generated from Planck best fit')
    theo_dls = theo_dls_class.get_DlsBB_Planck
elif CMB_from_CAMB:
    print('CMB generated from CAMB')
    theo_dls = theo_dls_class.get_DlsBB
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

if matrix == 'dense':
    figure(figsize=(15,15))

for i in range(len(instrum_config)):
    if matrix == 'dense':
        if skyconfig['pysm_fg'][0]=='d6':
            Ncov[i] = np.cov(DlsBB_array_for_noise[i].T) #noise covariance matrix
        else:
            Ncov[i] = np.cov(DlsBB_array[i].T) #noise covariance matrix
        print(DlsBB_array[i].shape, Ncov[i].shape)
        
        subplot(2,2,i+1)
        imshow(Ncov[i], cmap='RdBu')
        colorbar()
        xlabel('$\ell$')
        ylabel('$\ell$')
        ell_to_use = np.round(ell_array[i], 1) 
        xticks(np.arange(len(ell_array[i])), labels=[str(ell_to_use[i]) for i in range(len(ell_to_use))])
        yticks(np.arange(len(ell_array[i])), labels=[str(ell_to_use[i]) for i in range(len(ell_to_use))])
        title('{} : Noise covariance matrix'.format(instrum_config[i]), y=1.1)
        
    elif matrix == 'diagonal':
        if skyconfig['pysm_fg'][0]=='d6':
            Ncov[i] = np.std(DlsBB_array_for_noise[i], axis = 0)*mult_factor
        else:
            Ncov[i] = np.std(ma.masked_invalid(DlsBB_array[i,:,:]), axis = 0)*mult_factor 
        print(DlsBB_array[i].shape, Ncov[i].shape)
        print('std {}: {}'.format(instrum_config[i], Ncov[i]))

if matrix == 'dense':
    savefig('./Cls_Likelihood_images/Noise_cov_matrix_'+out_filename+'.png')  

if compute_like_on_all_iter:
    #likelihood
    like2 = np.zeros((len(instrum_config), Niter, len(rv_vec))) #Niter
    ML_r = np.zeros((len(instrum_config), Niter)) #Niter
    onesigma = np.zeros((len(instrum_config), Niter)) #Niter
    #log like eval
    for k in range(len(instrum_config)): #len(instrum_config)
        print('Doing {}'.format(instrum_config[k]))
        for i in range(Niter): #Niter
            print('iter ', i+1)
            likelihood_class = qubic.mcmc.LogLikelihood(xvals=ell_array[k], yvals=DlsBB_array[k,i,:],
                                                        errors=Ncov[k], model = theo_dls, nbins=len(ell_array[k]),
                                                        flatprior=flatprior, covariance_model_funct=covariance_model_funct) #DlsBB_array[k,i]
            for j in range(len(rv_vec)):
                like2[k,i,j] = np.exp(likelihood_class([rv_vec[j]]))
                #(- 0.5) * (((estim_dls[i] - theo_dls(ell,rv_vec[j])).T @ invN) @ (estim_dls[i] - theo_dls(ell,rv_vec[j])))
            ML_r[k,i], onesigma[k,i] = likelihoodlib2.give_r_sigr(rv_vec, like2[k,i,:])
            
else:
    #likelihood
    like2 = np.zeros((len(instrum_config), 1, len(rv_vec))) #Niter
    ML_r = np.zeros((len(instrum_config), 1)) #Niter
    onesigma = np.zeros((len(instrum_config), 1)) #Niter
    #log like eval
    for k in range(len(instrum_config)): #len(instrum_config)
        print('Doing {}'.format(instrum_config[k]))
        for i in range(1): #Niter
            print('iter ', i+1)
            likelihood_class = qubic.mcmc.LogLikelihood(xvals=ell_array[k], yvals=np.mean(ma.masked_invalid(DlsBB_array[k,:,:]), axis=0),
                                                        errors=Ncov[k], model = theo_dls, nbins=len(ell_array[k]),
                                                        flatprior=flatprior, covariance_model_funct=covariance_model_funct) #DlsBB_array[k,i]
            for j in range(len(rv_vec)):
                like2[k,i,j] = np.exp(likelihood_class([rv_vec[j]]))
                #(- 0.5) * (((estim_dls[i] - theo_dls(ell,rv_vec[j])).T @ invN) @ (estim_dls[i] - theo_dls(ell,rv_vec[j])))
            ML_r[k,i], onesigma[k,i] = likelihoodlib2.give_r_sigr(rv_vec, like2[k,i,:])
    

#plots------------------------------------------------------------------------------------------------------------------------------------
color = cm.jet(np.linspace(0,1,Niter))
norm = mpl.colors.Normalize(vmin=1, vmax=Niter)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

figure(figsize=(15,15))
for j in range(len(instrum_config)): #len(instrum_config)
    subplot(3,3,j+1)
    for i in range(1): #Niter
        maxL_r = rv_vec[np.argmax(like2[j,i])]
        onesigma_r = rv_vec[(np.around(like2[j,i]/np.max(like2[j,i]),2)==0.61)][-1] #onesigma[j,i]
        onesigma_r_68 = rv_vec[(np.around(like2[j,i]/np.max(like2[j,i]),2)==0.68)][-1] #onesigma[j,i]
        print(onesigma_r)
        plot(rv_vec, like2[j,i]/np.max(like2[j,i]), color=color[i], linewidth=0.5, label=r'r = {:.4f}$\pm$ {:.4f}'.format(maxL_r, onesigma_r-maxL_r))  
    vlines(0., 0, 1.1, color='k', linestyle='solid')
    vlines(maxL_r, 0, 1, color='k', linestyle='--')
    vlines(onesigma_r, 0, 0.59, color='k', linestyle='--', label=r'$\mu + \sigma$(r) = {:.4f}'.format(onesigma_r))
    vlines(onesigma_r_68, 0, 0.7, color='b', linestyle='--', label='68-th {:.5f}'.format(onesigma_r_68))
    vlines(onesigma[j,i], 0, 0.7, color='g', linestyle='--', label='68-th {:.5f}'.format(onesigma[j,i]))
    xlabel('r', fontsize=26)
    ylabel('likelihood', fontsize=26)
    title('{} likelihood'.format(instrum_config[j]), fontsize=22)
    xlim([np.min(rv_vec),np.max(rv_vec)])
    grid()
    ylim([0,1.1])
    legend(loc='best')
tight_layout()
#c = colorbar(cmap)
#c.set_label('N iter', fontsize=26)
#savefig('./Cls_Likelihood_images/All_likelihoods_'+out_filename+'.png')
savefig('./Cls_Likelihood_images/Likelihood_meanDl_'+out_filename+'.png')

outputfile = './results/Reconstructed_r_'+out_filename+'.pkl'
pickle.dump({'r_vec': rv_vec, 'likelihood': like2}, open(outputfile, 'wb'))

'''
#histogram
figure(figsize=(16,16))

#ML_r = []
#for i in range(len(instrum_config)): #len(instrum_config)
    #save r peak
#    ML_r.append(rv_vec[np.argmax(like2[i],axis=1)])

for i in range(len(instrum_config)): #len(instrum_config)
    #bin the data
    data_bins, bins = np.histogram(ML_r[i], bins=20, range=(np.min(np.array(ML_r)), np.max(np.array(ML_r))), density=False) #
        #print(data_bins)
    #def array with the center of the bins, so that it has the same shape as binned data
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    mean = np.mean(ML_r[i])
    std = np.std(ML_r[i])
    
    #plot histogram    
    bar(binscenters, data_bins, width=bins[1] - bins[0], alpha=1-0.5,
        label = '{}: r = {:.1E}$\pm$ {:.1E}'.format(instrum_config[i], mean, std), color=colors[i])

vlines(r, 0, np.max(data_bins), linestyles ="dashed", colors ="g", label='Input r value = {}'.format(r))
legend(fontsize=25, loc='center right') #loc='center left', bbox_to_anchor=(1, 0.5))
xlabel('Best fit r', fontsize=25)
ylabel('counts', fontsize=25)
#xlim([-0.0025, 0.0025])
#xlim([-0.005, 0.005])
#xlim([-0.004, 0.004])
xlim([2*np.min(rv_vec),2*np.max(rv_vec)])
tick_params(axis='both', which='major', labelsize=25)
title_str='{} iterations,  integration points = {}, fg = {}, r = {}, $A_\mathrm{{lens}}$ = {}, $\ell_\mathrm{{corr}}$ = {}'.format(Niter, N_sample_band, skyconfig['pysm_fg'], r, Alens, corr_l)
suptitle(title_str, fontsize=14, y=1)
savefig('./Cls_Likelihood_images/Best_fit_r_distribution_'+out_filename+'.png')

outputfile = './results/Reconstructed_r_'+out_filename+'.pkl'
#pickle.dump({'r_vec': rv_vec, 'likelihood': like2, 'Best_fit_r_d1s1': np.array(ML_r)}, open(outputfile, 'wb'))
pickle.dump({'r_vec': rv_vec, 'likelihood': like2, 'Best_fit_r_d1s1': ML_r, 'Sigma_best_fit_r_d1s1': onesigma}, open(outputfile, 'wb'))
'''