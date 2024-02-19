import numpy as np
import pickle


s = 20
exp='S4'
corr=2000           ### 1000 for d0, 2000 for d1
r=0.000
typ=0             ### 0 for Fullpipeline, 1 for propagation
n_ite=25
nside_fgb=8
Nbands=9
dust='d1'
Nf = 7
nsub=[Nf]

aposize=4


path='/work/regnier/forecast_deco/results/new_data/'
name='cl_nsub{}_r{:.3f}_{}_{}_nsidefgb{}_fullpipeline_corr{}_{}reals_{}.pkl'.format(Nf, r, exp, dust, nside_fgb, corr, s, 1)

with open(path+name, 'rb') as f:
    x = pickle.load(f)
    shape_maps=np.sum(x['pixok'])
    leff=x['leff']
    pixok=x['pixok']
    
    
cl=np.zeros((len(nsub), s*n_ite, 9))
beta=np.zeros((len(nsub), s*n_ite, 2, 12*nside_fgb**2))
    
k=0
for i in range(n_ite):
    print(k*s, (k+1)*s)
    name='cl_nsub{}_r{:.3f}_{}_{}_nsidefgb{}_fullpipeline_corr{}_{}reals_{}.pkl'.format(Nf, r, exp, dust, nside_fgb, corr, s, i+1)
    with open(path+name, 'rb') as f:
        x = pickle.load(f)
    #print(x['beta'].shape, x['index_true'].shape)
    for j in range(s):
        cl[0, k]=x['mycl'][j, 0].copy()
        beta[0, k]=x['beta'][j, 0].copy()
        k+=1
        #residuals[j, k]=x['residuals'][j, :, pixok].T.copy()
        #residualsf[j, k]=x['residuals_fore'][j, :, pixok].T.copy()
    #index[k]=x['index_true'].copy()

    
    #index[k]=np.array([1.54, -3])
    #x['index_true'][0].copy()
    
    #allfore[k]=x['allfore'][0].copy()
    #index[l]=x['index_true'][1].copy()
    
mydict = {'mycl': cl, 'leff': leff, 
         'beta':beta}#, 'index_true':index}
output = open(path+'cl_nsub{}_nside256_r{:.3f}_{}_{}_nsidefgb{}_fullpipeline_corr{:.0f}_{}reals.pkl'.format(Nf, r, exp, dust, nside_fgb, corr, n_ite*s), 'wb')
pickle.dump(mydict, output)
output.close()