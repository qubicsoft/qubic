from __future__ import division, print_function
import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys



try:
    os.makedirs('result')
except:
    pass



mpl.style.use('classic')
name='run_dict'

# INSTRUMENT
d = qubic.qubicdict.qubicDict()
d.read_from_file(sys.argv[1])

q = qubic.QubicMultibandInstrument(d)
p= qubic.get_pointing(d)

#q[0].horn.plot()
#q[0].detector.plot()

s = qubic.QubicScene(d)

m0=hp.fitsfunc.read_map('CMB_test.fits', field=(0,1,2))

x0=np.zeros((d['nf_sub'],len(m0[0]),len(m0)))
for j in range(len(m0)):
    for i in range(d['nf_sub']):
        x0[i,:,j]=m0[j]

center_gal = qubic.equ2gal(d['RA_center'], d['DEC_center'])

Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = qubic.compute_freq(d['filter_nu']/1e9, d['nf_sub'], d['filter_relative_bandwidth']) # Multiband instrument model

a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)

TOD, maps_convolved_useless = a.get_observation(x0, noiseless=True)


nf_sub_rec = 2
Nbfreq, nus_edge, nus, deltas, Delta, Nbbands = qubic.compute_freq(d['filter_nu']/1e9, nf_sub_rec, d['filter_relative_bandwidth']) 
arec = qubic.QubicMultibandAcquisition(q, p, s,d, nus_edge)
maps_recon = arec.tod2map(TOD, tol=1e-3, maxiter=100000)



if nf_sub_rec==1: maps_recon=np.reshape(maps_recon, np.shape(maps_convolved))
TOD_useless, maps_convolved = arec.get_observation(x0)
maps_convolved = np.array(maps_convolved)
cov = arec.get_coverage()
cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1
diffmap = maps_convolved - maps_recon
maps_convolved[:,unseen,:] = hp.UNSEEN
maps_recon[:,unseen,:] = hp.UNSEEN
diffmap[:,unseen,:] = hp.UNSEEN
stokes = ['I', 'Q', 'U'] 

for istokes in [0,1,2]:
    plt.figure(istokes,figsize=(12,12)) 
    if istokes==0:
        xr=200 
    else:
        xr=10
    for i in range(nf_sub_rec):
        # proxy to get nf_sub_rec maps convolved
        in_old=hp.gnomview(maps_convolved[i,:,istokes], rot=center_gal, reso=10, sub=(nf_sub_rec,3,3*i+1), min=-xr, max=xr,title='Input '+stokes[istokes]+' SubFreq {}'.format(i), return_projected_map=True)
        np.savetxt('result/in_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),in_old)
        out_old=hp.gnomview(maps_recon[i,:,istokes], rot=center_gal, reso=10,sub=(nf_sub_rec,3,3*i+2), min=-xr, max=xr,title='Output '+stokes[istokes]+' SubFreq {}'.format(i), return_projected_map=True)
        np.savetxt('result/out_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),out_old)
        res_old=hp.gnomview(diffmap[i,:,istokes], rot=center_gal, reso=10,sub=(nf_sub_rec,3,3*i+3), min=-xr, max=xr,title='Residual '+stokes[istokes]+' SubFreq {}'.format(i), return_projected_map=True)
        np.savetxt('result/res_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),res_old)

    plt.savefig('result/%s_map_%s.png'%(name,stokes[istokes]),bbox_inches='tight')
    plt.clf()
    plt.close()
