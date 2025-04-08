import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle as pkl

from pyoperators import (
    MPI,
    BlockRowOperator,
    DenseOperator,
    ReshapeOperator,
)
from pyoperators.iterative.core import AbnormalStopIteration
from pysimulators.interfaces.healpy import (
    HealpixConvolutionGaussianOperator,
    Spherical2HealpixOperator,
)

from qubic.lib.Instrument.Qacquisition import QubicDualBand
from qubic.lib.Instrument.Qinstrument import compute_freq
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import CMBModel
from qubic.lib.MapMaking.Qatmosphere_2d import AtmosphereMaps
from qubic.lib.MapMaking.Qcg_test_for_atm import PCGAlgorithm
from qubic.lib.Qsamplings import QubicSampling, equ2gal, get_pointing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Import simulation parameters
with open("params.yml", "r") as file:
    params = yaml.safe_load(file)

np.random.seed(params["seed"])

# Call the class which build the atmosphere maps
atm = AtmosphereMaps(params)
qubic_dict = atm.qubic_dict

npix = hp.nside2npix(params["nside"])

### Random pointing
qubic_dict["random_pointing"] = True
qubic_dict['date_obs'] = '2023-10-01 22:57:00.000'
qubic_dict['period'] = 3

qubic_dict["fix_azimuth"]["apply"] = False

q_sampling_gal = get_pointing(qubic_dict)
qubic_patch = np.array([0, -57])
center_gal = equ2gal(qubic_patch[0], qubic_patch[1])
center_local = np.array(
    [np.mean(q_sampling_gal.azimuth), np.mean(q_sampling_gal.elevation)]
)

q_sampling_local = QubicSampling(
    q_sampling_gal.index.size,  # int(np.ceil(qubic_dict['duration']*3600/qubic_dict['period'])),
    date_obs=qubic_dict["date_obs"],
    period=qubic_dict["period"],
    latitude=qubic_dict["latitude"],
    longitude=qubic_dict["longitude"],
)

q_sampling_local.azimuth = q_sampling_gal.azimuth
q_sampling_local.elevation = q_sampling_gal.elevation
q_sampling_local.pitch = q_sampling_gal.pitch
q_sampling_local.angle_hwp = q_sampling_gal.angle_hwp

q_sampling_local.fix_az = True

# Build CMB map
cl_cmb = CMBModel(None).give_cl_cmb(r=0, Alens=1)
cmb_map = hp.synfast(cl_cmb, params["nside"], new=True, verbose=False).T

cmb_maps = np.ones((params["nsub_in"], hp.nside2npix(params["nside"]), 3))
cmb_maps *= cmb_map[None]

atm_maps = np.zeros((cmb_maps.shape))
atm_maps[..., 0] = atm.get_temp_maps(atm.delta_rho_map)

fwhm_synthbeam150 = 0.006853589624526168

_, _, filter_nus150, deltas150, _, _ = compute_freq(
    150,
    int(params["nsub_in"] / 2),
    relative_bandwidth=qubic_dict["filter_relative_bandwidth"],
    frequency_spacing="log",
)
_, _, filter_nus220, deltas220, _, _ = compute_freq(
    220,
    int(params["nsub_in"] / 2),
    relative_bandwidth=qubic_dict["filter_relative_bandwidth"],
    frequency_spacing="log",
)

nus_tod = np.concatenate((filter_nus150, filter_nus220)) * 1e9
fwhm_tod = fwhm_synthbeam150 * 150e9 / nus_tod

for isub in range(nus_tod.size):
    C = HealpixConvolutionGaussianOperator(fwhm=fwhm_tod[isub])
    atm_maps[isub] = C(atm_maps[isub])
    cmb_maps[isub] = C(cmb_maps[isub])
    
input_maps = np.zeros((2 * params["nsub_in"], hp.nside2npix(params["nside"]), 3))

input_maps[: params["nsub_in"]] = cmb_maps
input_maps[params["nsub_in"] :] = atm_maps

### Build Expected Atm Maps
true_maps = np.zeros((2, 12 * params["nside"] ** 2, 3))

# Build the reconstructed maps and frequency by taking the mean inside each reconstructed frequency band
C = HealpixConvolutionGaussianOperator(fwhm=np.mean(fwhm_tod))
true_maps[0] = C(cmb_map)
true_maps[1] = C(np.mean(atm_maps, axis=0))

min_input = np.min(true_maps, axis=1)
max_input = np.max(true_maps, axis=1)

max = np.max([min_input, max_input], axis=0)
min_input = -max
max_input = max

MixingMatrix = np.ones((params["nsub_in"], 2))
# Atm mixing matrix
MixingMatrix[:, 1] = atm.temperature * atm.integrated_abs_spectrum * atm.mean_water_vapor_density

q_acquisition_local = QubicDualBand(
    qubic_dict, params["nsub_in"], params["nsub_in"], sampling=q_sampling_local
)

q_acquisition_gal = QubicDualBand(
    qubic_dict, params["nsub_in"], params["nsub_in"], sampling=q_sampling_gal
)

coverage_gal = q_acquisition_gal.coverage
covnorm_gal = coverage_gal / coverage_gal.max()
seenpix_gal = covnorm_gal > params["coverage_cut"]
seenpix_gal_wo_cut = covnorm_gal > 0

coverage_local = q_acquisition_local.coverage
covnorm_local = coverage_local / coverage_local.max()
seenpix_local = covnorm_local > params["coverage_cut"]
seenpix_local_wo_cut = covnorm_local > 0

seenpix = np.array([seenpix_gal, seenpix_local])

# Galactic Coordinates

H_gal = q_acquisition_gal.get_operator()

invN_gal = q_acquisition_gal.get_invntt_operator(False, False)

# Local Coordinates

H_local = q_acquisition_local.get_operator()

invN_local = q_acquisition_local.get_invntt_operator(False, False)

### Full MM
R = ReshapeOperator((2 * 992, params["npointings"]), (params["npointings"] * 992 * 2))

r = ReshapeOperator((npix, 3), (1, npix, 3))
A_gal = (
    DenseOperator(
        MixingMatrix[:, 0, None],
        broadcast="rightward",
        shapein=(1, npix, 3),
        shapeout=(params["nsub_in"], npix, 3),
    )
    * r
)
A_local = (
    DenseOperator(
        MixingMatrix[:, 1, None],
        broadcast="rightward",
        shapein=(1, npix, 3),
        shapeout=(params["nsub_in"], npix, 3),
    )
    * r
)

H = BlockRowOperator([H_gal(A_gal), H_local(A_local)], axisin=0) * ReshapeOperator(
    (2, npix, 3), (2 * npix, 3)
)
print(H.shapein, H.shapeout)

invN = invN_gal
print(invN.shapein, invN.shapeout)

tod = H(true_maps)
print(tod.shape)

# Ax=b equation to be solve by PCG
A = H.T * invN * H
b = H.T * invN * tod

# I start from an atm map for the CMB and from a cmb map for the atmosphere
x0 = true_maps * np.random.uniform(0.5, 1.5, true_maps.shape)

# Run PCG

seenpix_pcg = np.array([seenpix_gal, seenpix_local])

algo = PCGAlgorithm(
    A,
    b,
    comm,
    x0=x0,
    tol=1e-4,
    maxiter=10,
    disp=True,
    M=None,
    center=[0, -57],
    reso=15,
    seenpix=seenpix_pcg,
    input=true_maps,
)
try:
    result = algo.run()
    success = True
    message = "Success"
except AbnormalStopIteration as e:
    result = algo.finalize()
    success = False
    message = str(e)
    
dict_solution = {
    'result': result,
    'true_maps': true_maps,
    'seenpix': seenpix,
    'center_gal': center_gal,
    'center_local': center_local,
    'min_input': min_input,
    'max_input': max_input,
}
    
with open("solution.pkl", "wb") as pkl_file:
    pkl.dump(dict_solution, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
