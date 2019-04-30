import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys
from time import time
 
try:
    os.makedirs('result')
except:
    pass

mpl.style.use('classic')
name='run_dict'

# Choose here your favourite dictionary 
d = qubic.qubicdict.qubicDict()
# d.read_from_file(sys.argv[1])
d.read_from_file('global_test.dict')

# Reading the beam_shape from the dictionary. You can change it  as follows:
# d['beam_shape' = 'multi_freq' # or 'fitted_beam' or  'multi_freq' 
print 'beam shape :', d['beam_shape']
name += '_' + d['beam_shape']

# Constructing a multiband instrument, sampling, and scene
q = qubic.QubicMultibandInstrument(d)
p= qubic.get_pointing(d)
s = qubic.QubicScene(d)

# Constructing a  reduced instrument with a subset of detectors chosen with their indices
dets = [231, 232]
subq = q.detector_subset(dets)

# Reading a sky map
m0=hp.fitsfunc.read_map('CMB_test.fits', field=(0,1,2))

x0=np.zeros((d['nf_sub'],len(m0[0]),len(m0)))
for j in range(len(m0)):
    for i in range(d['nf_sub']):
        x0[i,:,j]=m0[j]

# field center in galactic coordinates
center_gal = qubic.equ2gal(d['RA_center'], d['DEC_center'])

# Choosing the subfrequencies for map acquisition
Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in =\
  qubic.compute_freq(d['filter_nu']/1e9, d['nf_sub'],
                     d['filter_relative_bandwidth']) # Multiband instrument model

# Map acquisition for the full and reduced instruments
a = qubic.QubicMultibandAcquisition(q, p, s, d, nus_edge_in)
suba = qubic.QubicMultibandAcquisition(subq, p, s, d, nus_edge_in)


t = time()
# Time lines construction
TOD, maps_convolved= a.get_observation(x0, noiseless=True)
subTOD= suba.get_observation(np.array(maps_convolved), convolution=False,
                                 noiseless=True)
print 'TOD time =', time()-t

# Choosing the subfrequencies for map reconstruction
nf_sub_rec = 2
Nbfreq_edge, nus_edge, nus, deltas, Delta, Nbbands = \
  qubic.compute_freq(d['filter_nu']/1e9, nf_sub_rec,
                         d['filter_relative_bandwidth'])

# A new instance of the acquisition class is required
arec = qubic.QubicMultibandAcquisition(q, p, s,d, nus_edge)
subarec = qubic.QubicMultibandAcquisition(subq, p, s,d, nus_edge)

# map reconstuction
# tol < 1e-3 or smaller required for reasonable reconstructio
# tol = 1e-1 is used to test changes to the qubic code for bugs
t = time()
maps_recon = arec.tod2map(TOD, tol=1e-1, maxiter=100000)
submaps_recon = subarec.tod2map(subTOD, tol=1e-1, maxiter=100000)
print ' beam shape :', d['beam_shape'], ', iteration time =', time()-t

# For comparison the convolved with the beam is required
TOD_useless, maps_convolved = arec.get_observation(x0)
maps_convolved = np.array(maps_convolved)

# keeping only the sky region which has been significantly observed
cov = arec.get_coverage()
cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1
diffmap = maps_convolved - maps_recon
maps_convolved[:,unseen,:] = hp.UNSEEN
maps_recon[:,unseen,:] = hp.UNSEEN
diffmap[:,unseen,:] = hp.UNSEEN


# Plotting and saving
stokes = ['I', 'Q', 'U'] 
for istokes in [0,1,2]:
    plt.figure(istokes,figsize=(12,12)) 
    if istokes==0:
        xr=200 
    else:
        xr=10
    for i in xrange(nf_sub_rec):
        # proxy to get nf_sub_rec maps convolved
        in_old=hp.gnomview(maps_convolved[i,:,istokes],
                               rot=center_gal, reso=10, sub=(nf_sub_rec,3,3*i+1), min=-xr,
                               max=xr,title='Input '+stokes[istokes]+' SubFreq {}'.format(i),
                               return_projected_map=True)
        np.savetxt('result/in_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),in_old)
        out_old=hp.gnomview(maps_recon[i,:,istokes], rot=center_gal, reso=10,
                                sub=(nf_sub_rec,3,3*i+2), min=-xr, max=xr,title='Output '
                                +stokes[istokes]+' SubFreq {}'.format(i),
                                return_projected_map=True)
        np.savetxt('result/out_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),out_old)
        res_old=hp.gnomview(diffmap[i,:,istokes], rot=center_gal, reso=10,
                                sub=(nf_sub_rec,3,3*i+3), min=-xr, max=xr,title='Residual '
                                +stokes[istokes]+' SubFreq {}'.format(i),
                                return_projected_map=True)
        np.savetxt('result/res_%s_%s_subfreq_%d.dat'%(name,stokes[istokes],i),res_old)

    plt.savefig('result/%s_map_%s.png'%(name,stokes[istokes]),bbox_inches='tight')
    plt.clf()
    plt.close()
