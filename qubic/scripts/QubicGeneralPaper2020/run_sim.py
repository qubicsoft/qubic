#!/usr/bin/python
from pylab import *
import os
import sys

# Specific science modules
import healpy as hp
import numpy as np

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
import pysm
import qubic
from qubic import SpectroImLib as si
from qubic import QubicSkySim as qss
from qubic import NamasterLib as nam

## Instructions:
## python ./run_sim.py <Dictname> <seed> <generic_string> <npointings> <tol>
## Exemple:
## python ./run_sim.py BmodesNoDustNoSystPaper0_2020.dict 42 CMB-Only-12sub_3000_1e-4 3000 1e-4
##
## NB: the  ouput files are for now by default in /qubic/Sims/EndToEndMaps/ (see line 121)
## On a nother machine it  has to be changed before running...


print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

dictname = str(sys.argv[1])
if str(sys.argv[2])=='None':
	seed = None
	print('Seed is None',seed)
else:
	seed = int(sys.argv[2])
	print('Seed is NOT None',seed)

namesim = str(sys.argv[3])
nptg = int(sys.argv[4])
tol = float(sys.argv[5])


print(dictname)
print(seed)
print(namesim)
print(nptg)
print(tol)



# Repository for dictionary and input maps
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
dictfilename = global_dir + '/dicts/'+dictname

# Read dictionary chosen
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
d['npointings'] = nptg
d['tol'] = tol
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])

print(d)

# Make a sky using QubicSkySim with r=0 (default simple request - the camb spectrum is calculated inside)
#seed = 42
sky_config = {'cmb': seed}
Qubic_sky = qss.Qubic_sky(sky_config, d)
x0 = np.reshape(Qubic_sky.get_simple_sky_map(),(d['nf_sub'],d['nside']**2*12,3))

# The input spectra are
input_cl = Qubic_sky.input_cmb_spectra

print('Input SubFrequencies Map with shape (nf_sub, #pixels, #stokes) : ', np.shape(x0))


# Pointing strategy
p = qubic.get_pointing(d)
print('=== Pointing DONE! ===')

np.random.seed(int((time.time()-1585480000)*1000))
print('Test RND: {}'.format(np.random.rand(1)))
# ==== TOD making ====
TOD, maps_convolved = si.create_TOD(d, p, x0)
maps_convolved = np.array(maps_convolved)
print('--------- Noiseless TOD with shape: {} - Done ---------'.format(np.shape(TOD)))
print('-------- Maps Convolved with shape: {} - Done ---------'.format(np.shape(maps_convolved)))

q = qubic.QubicMultibandInstrument(d)
s = qubic.QubicScene(d)


# ==== Map Making ====
nf_sub_rec = d['nf_recon']

Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands = qubic.compute_freq(150, nf_sub_rec)
print('Band center:', nus)
print('Band edges:', nus_edge)
print('Band width:', deltas)

len(nus_edge)
for i in range(len(nus_edge) - 1):
    print('base =', nus_edge[i+1] / nus_edge[i])


maps_recon, cov, nus, nus_edge, maps_convolved = si.reconstruct_maps(TOD, d, p,
                                                                    nf_sub_rec, x0=x0)
maps_convolved = np.reshape(maps_convolved,(d['nf_recon'], 12*d['nside']**2, 3))
maps_recon = np.reshape(maps_recon,(d['nf_recon'], 12*d['nside']**2, 3))
print(maps_recon.shape)

# Look at the coverage of the sky
coverage = np.sum(cov.copy(), axis=0)
maxcov = np.max(coverage)
unseen = coverage < maxcov * 0.1
print(maps_convolved.shape)
maps_convolved[:, unseen, :] = hp.UNSEEN
maps_recon[:, unseen, :] = hp.UNSEEN

#### Save maps
rnd_name = qss.random_string(10)
directory = '/qubic/Sims/EndToEndMaps/'
FitsArray(maps_recon).save(directory+namesim+'_maps_recon_seed_'+str(seed)+'_'+rnd_name+'.fits')
FitsArray(maps_convolved).save(directory+namesim+'_maps_convolved_seed_'+str(seed)+'_'+rnd_name+'.fits')
FitsArray(coverage).save(directory+namesim+'_maps_coverage_'+rnd_name+'.fits')
   
import pickle
with open(directory+namesim+'_dictionary_'+rnd_name+'.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(directory+namesim+'_input_cell_'+rnd_name+'.pickle', 'wb') as handle:
    pickle.dump(Qubic_sky.input_cmb_spectra, handle, protocol=pickle.HIGHEST_PROTOCOL)













