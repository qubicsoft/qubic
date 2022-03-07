import pickle
import numpy as np
import likelihoodlib
import qubicplus
from pylab import *

'''
This code import the result file of MC_cls.py and makes additional analysis
The pickle file containing the results is configured like so: [leff, cl, param_comb, tabseed, db, props, bw, sys.argv]
'''

print('Start...')

#CHANGE THIS ACCORDINGLY

iib = int(sys.argv[6]) #number of freq used for freq integration
Temp_fix_var = int(sys.argv[4]) # if =1, then temp was fixed
Synch_fix_var = int(sys.argv[5]) #if =1, then synch was fixed
nubreak = int(sys.argv[3])     # True value of nubreak
N = int(sys.argv[1])           # Number of iterations saved in the pickle file
tot_ite = int(sys.argv[2])         # Number of times the code has been executed: if 1, single execution; if >1, array execution

if tot_ite==1: #single execution: the total number of iteration to average is N
    N_tot = N
else: #array execution: the tot. number of iters to average is N*tot_ite
    N_tot = N * tot_ite


if Temp_fix_var == 1:
    T=20
    name_T='_fixtemp'
else:
    T=None
    name_T=''
    nb_param+=1

if Synch_fix_var == 1:
    fix_sync=True
    name_s='_fixsync'
else:
    fix_sync=False
    name_s=''
    nb_param+=1

#def variables for likelihood part
fsky=0.03
nside=256
center_RA_DEC=[-30,-30]

method='sigma'
lmin=40
delta_ell=30
lmax = 2*nside-1
cov_cut=0.1
factornoise=1.


#FUNCTION DEFINITION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
def read_and_save_pickle(path_to_file):
    #open file and save its content ad an object
    input_file = open(path_to_file, 'rb')
    obj_file = pickle.load(input_file)
    
    print('Loading and saving data from file: {}'.format(path_to_file))
    
    #assign the file content to a proper variable
    ell = obj_file[0]
    cls = obj_file[1]
    params = obj_file[2]
    seed_list = obj_file[3]
    delta_beta_vec = obj_file[4]
    props_vec = obj_file[5]
    break_width = obj_file[6]
    system_argv = obj_file[7]
    #print info
    print('\nData from code --> {}'.format(system_argv[0]))
    print('Simulated {} iterations'.format(system_argv[1]))
    print('ite = {}'.format(system_argv[2]))
    print('Simulated double-beta dust with --> nu_break = {0} GHz, break_steep = {1}'.format(system_argv[3], break_width))
    print('Simulated {} delta_beta configurations: '.format(len(delta_beta_vec)), delta_beta_vec)
    print('Simulated {} instrument configurations: '.format(len(props_vec)), props_vec)    
    
    return ell, cls, params, seed_list, props_vec, delta_beta_vec

#CODE PART~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if tot_ite==1:
    namefile = 'cls_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}.pkl'.format(iib, name_T, name_s, nubreak, N, tot_ite)
    path_to_file = './results/'+namefile

    ell, cls, params, seed_list, props_vec, delta_beta_vec = read_and_save_pickle(path_to_file)

    #save cls_BB from s4
    clsBB_s4 = cls[:,0,:,:,2] #order is N_iter, props, db, ell, TT/EE/BB/TE
    #save cls_BB from bi
    clsBB_bi = cls[:,1,:,:,2]
else:
    namefile = 'cls_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}.pkl'.format(iib, name_T, name_s, nubreak, N, 0)
    path_to_file = './results/'+namefile

    ell, cls, params, seed_list, props_vec, delta_beta_vec = read_and_save_pickle(path_to_file)

    #save cls_BB from s4
    clsBB_s4 = cls[:,0,:,:,2] #order is N_iter, props, db, ell, TT/EE/BB/TE
    #save cls_BB from bi
    clsBB_bi = cls[:,1,:,:,2]
    for ite in range(1,tot_ite):
            namefile = 'cls_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}.pkl'.format(iib, name_T, name_s, nubreak, N, ite)
            path_to_file = './results/'+namefile
            _, cls_tmp, params_tmp, seed_list_tmp, _, _ = read_and_save_pickle(path_to_file)
            clsBB_s4 = np.append(clsBB_s4, cls_tmp[:,0,:,:,2], axis=0)
            clsBB_bi = np.append(clsBB_bi, cls_tmp[:,1,:,:,2], axis=0)
            print('cls shape: ', clsBB_s4.shape, clsBB_bi.shape )
            seed_list = np.append(seed_list, seed_list_tmp)
            print('seed list shape: ', seed_list.shape)


#print to check
print('Printing S4 and BI cls shape to check... ', clsBB_s4.shape, clsBB_bi.shape)

#perform mean and std over all iterations
mean_clsBB_s4 = np.mean(clsBB_s4, axis=0)
std_clsBB_s4 = np.std(clsBB_s4, axis=0)

mean_clsBB_bi = np.mean(clsBB_bi, axis=0)
std_clsBB_bi = np.std(clsBB_bi, axis=0)

#print check
print('Printing S4 and BI mean and sigmas shape to check...')
print(mean_clsBB_s4.shape, std_clsBB_s4.shape, mean_clsBB_bi.shape, std_clsBB_bi.shape)

#compute posterior likelihood, for each instrum config and for each delta_beta

#compute coverage
coverage = qubicplus.get_coverage(fsky=fsky, nside=nside, center_radec=center_RA_DEC)

#number of delta beta cases
num_db=len(delta_beta_vec)

#create arrays to save the maximum likelihood values of r and sigma at different delta beta
ML_r_s4 = np.empty(num_db)
ML_r_bi = np.empty(num_db)
sigma_r_s4 = np.empty(num_db)
sigma_r_bi = np.empty(num_db)

figure(figsize=(16, 10))

#def number of subplots
if(num_db==2):
    n_rows=1
    n_cols=2
elif(num_db==6):
    n_rows=2
    n_cols=3
else:
    print('Warning: define the subplots properly! The default configuration will be used for now')
    n_rows=1
    n_cols=num_db

for db in range(num_db):
    subplot(n_rows,n_cols,db+1)
    #s4
    leff_s4, scl_s4, rv_s4, like_s4, cumint_s4, rlim68_s4, rlim95_s4 = likelihoodlib.get_results(ell=ell, mcl=mean_clsBB_s4[db,:],
                                                                                                 scl=std_clsBB_s4[db,:], coverage=coverage, 
                method=method, lmin=lmin, delta_ell=delta_ell, covcut=cov_cut, rv=None, factornoise=1.)
        #eval ML value of r
    ML_index = np.where(like_s4==np.max(like_s4))[0][0] 
    print('S4: ML value of r = ', rv_s4[ML_index])
    ML_r_s4[db] = rv_s4[ML_index]
    sigma_r_s4[db] = rlim68_s4
    
    #bi
    leff_bi, scl_bi, rv_bi, like_bi, cumint_bi, rlim68_bi, rlim95_bi = likelihoodlib.get_results(ell=ell, mcl=mean_clsBB_bi[db,:],
                                                                                                 scl=std_clsBB_bi[db,:], coverage=coverage, 
                method=method, lmin=lmin, delta_ell=delta_ell, covcut=cov_cut, rv=None, factornoise=1.) 
        #eval ML value of r
    ML_index = np.where(like_bi==np.max(like_bi))[0][0] 
    print('BI: ML value of r = ', rv_bi[ML_index])
    ML_r_bi[db] = rv_bi[ML_index]
    sigma_r_bi[db] = rlim68_bi

    
    #plot likelihood for each delta_beta
    p = plot(rv_s4,  like_s4/np.max(like_s4), 'r', label='S4: ML r={0:6.4f}, $\sigma(r)={1:6.4f}$'.format(ML_r_s4[db], rlim68_s4))
    plot(rlim68_s4+np.zeros(2), [0,1.2], ':', color=p[0].get_color())
    p = plot(rv_bi,  like_bi/np.max(like_bi), 'b', label='BI: ML r={0:6.4f}, $\sigma(r)={1:6.4f}$'.format(ML_r_bi[db], rlim68_bi))
    plot(rlim68_bi+np.zeros(2), [0,1.2], ':', color=p[0].get_color())
    title(r'$\nu_b$ = {0} GHz, $\Delta \beta$ = {1}'.format(nubreak, delta_beta_vec[db]))
    xlabel('r')
    ylabel('posterior L')
    legend(fontsize=8, loc='upper right')
    xlim(0,0.03)
    ylim(0,1.2)
    savefig('./figures/PostLikelihood_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}betas.png'.format(iib, name_T, name_s, nubreak, N_tot, num_db))
    
#plot ML value of r as function of beta
figure(figsize=(16, 10))
title(r'$\nu_b$ = {0} GHz, tot. iterations = {1}'.format(nubreak, N_tot))
errorbar(delta_beta_vec, ML_r_s4, yerr=sigma_r_s4/np.sqrt(N_tot), marker='p', color='r',label='S4')    
errorbar(delta_beta_vec, ML_r_bi, yerr=sigma_r_bi/np.sqrt(N_tot), marker='p', color='b',label='BI')
hlines(0.0005, 0, delta_beta_vec[-1], colors='g', linestyles='dashed', label='CMB-S4 science target')
xlabel(r'$\Delta\beta$')
ylabel('ML r')
legend(fontsize=8, loc='upper right')
savefig('./figures/ML_of_r_vs_db_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}betas.png'.format(iib, name_T, name_s, nubreak, N_tot, num_db))

#plot delta btw ML value of r in the case db!=0 and db=0
error_delta_r_s4 = np.sqrt(sigma_r_s4[:]**2 + sigma_r_s4[0]**2)/np.sqrt(N_tot)
error_delta_r_bi = np.sqrt(sigma_r_bi[:]**2 + sigma_r_bi[0]**2)/np.sqrt(N_tot)
print('error on delta r has shape: ', error_delta_r_s4.shape, error_delta_r_bi.shape)

figure(figsize=(16, 10))
title(r'$\nu_b$ = {0} GHz, tot. iterations = {1}'.format(nubreak, N_tot))
errorbar(delta_beta_vec, ML_r_s4-ML_r_s4[0], yerr=error_delta_r_s4, marker='p', color='r',label='S4')
errorbar(delta_beta_vec, ML_r_bi-ML_r_bi[0], yerr=error_delta_r_bi, marker='p', color='b',label='BI')
hlines(0.0005, 0, delta_beta_vec[-1], colors='g', linestyles='dashed', label='CMB-S4 science target')
xlabel(r'$\Delta\beta$')
ylabel(r'$\Delta$(r) w.r.t. $\Delta\beta$=0')
legend(fontsize=8, loc='upper right')
savefig('./figures/Delta_r_vs_db_iib{:.0f}_QU{}{}_truenub{:.0f}_{}reals_{}betas.png'.format(iib, name_T, name_s, nubreak, N_tot, num_db))

