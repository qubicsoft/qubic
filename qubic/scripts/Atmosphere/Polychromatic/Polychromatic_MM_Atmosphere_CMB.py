import pickle as pkl

import healpy as hp
import numpy as np
import yaml
from pyoperators import (
    MPI,
    BlockDiagonalOperator,
    BlockRowOperator,
    DenseOperator,
    DiagonalOperator,
    IdentityOperator,
    ReshapeOperator,
)
from pyoperators.iterative.core import AbnormalStopIteration
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

from qubic.lib.Instrument.Qacquisition import QubicInstrumentType
from qubic.lib.Instrument.Qinstrument import compute_freq
from qubic.lib.MapMaking.FrequencyMapMaking.Qspectra_component import CMBModel
from qubic.lib.MapMaking.Qatmosphere import AtmosphereMaps
from qubic.lib.MapMaking.Qcg_test_for_atm import PCGAlgorithm
from qubic.lib.Qsamplings import QubicSampling, equ2gal, get_pointing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Import simulation parameters
with open("params.yml", "r") as file:
    params = yaml.safe_load(file)

np.random.seed(params["seed"])

# Call the class which builds the atmosphere maps
atm = AtmosphereMaps(params)
qubic_dict = atm.qubic_dict

npix = hp.nside2npix(params["nside"])

qubic_dict["instrument_type"] = "UWB"
qubic_dict["interp_projection"] = False

### Scanning strategy
# Galactic coordinates
qubic_dict["random_pointing"] = True
qubic_dict["date_obs"] = "2023-10-01 18:57:00.000"
qubic_dict["period"] = 3
qubic_dict["sweeping_pointing"] = False
qubic_dict["repeat_pointing"] = False
qubic_dict["fix_azimuth"]["apply"] = False

q_sampling_gal = get_pointing(qubic_dict)
qubic_patch = np.array([0, -57])
center_gal = equ2gal(qubic_patch[0], qubic_patch[1])
center_local = np.array([np.mean(q_sampling_gal.azimuth), np.mean(q_sampling_gal.elevation)])

# Local coordinates: same sampling as the galactic one, just interpreted as fixed to the
# instrument's az/el frame instead of the sky (fix_az=True) -- this is how the atmosphere
# component (fixed relative to the ground) is distinguished from the CMB (fixed on the sky).
q_sampling_local = QubicSampling(
    q_sampling_gal.index.size,
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

### Input maps
# CMB
cl_cmb = CMBModel(None).give_cl_cmb(r=0, Alens=1)
cmb_map = hp.synfast(cl_cmb, params["nside"], new=True, verbose=False).T

cmb_maps = np.ones((params["nsub_in"], hp.nside2npix(params["nside"]), 3))
cmb_maps *= cmb_map[None]

# Atmosphere
atm_maps = np.zeros(cmb_maps.shape)
atm_maps[..., 0] = atm.get_temp_maps(atm.delta_rho_map) / 1e3

# Apply convolutions
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

### True maps
true_maps = np.zeros((2, 12 * params["nside"] ** 2, 3))

# Build the reconstructed maps and frequency by taking the mean inside each reconstructed frequency band
C = HealpixConvolutionGaussianOperator(fwhm=np.mean(fwhm_tod))
true_maps[0] = C(cmb_map)
true_maps[1] = C(np.mean(atm_maps, axis=0))

min_input = np.min(true_maps, axis=1)
max_input = np.max(true_maps, axis=1)

max_range = np.max([min_input, max_input], axis=0)
min_input = -max_range
max_input = max_range

### Mixing matrix
MixingMatrix = np.ones((params["nsub_in"], 2))
# Atm mixing matrix
MixingMatrix[:, 1] = atm.temperature * atm.integrated_abs_spectrum * atm.mean_water_vapor_density

### Build QUBIC instances
q_acquisition_local = QubicInstrumentType(
    qubic_dict, params["nsub_in"], params["nsub_in"], sampling=q_sampling_local
)

q_acquisition_gal = QubicInstrumentType(
    qubic_dict, params["nsub_in"], params["nsub_in"], sampling=q_sampling_gal
)

coverage_gal = q_acquisition_gal.coverage
covnorm_gal = coverage_gal / coverage_gal.max()
seenpix_gal = covnorm_gal > params["coverage_cut"]

coverage_local = q_acquisition_local.coverage
covnorm_local = coverage_local / coverage_local.max()
seenpix_local = covnorm_local > params["coverage_cut"]

seenpix = np.array([seenpix_gal, seenpix_local])

### Build QUBIC operators
# Galactic coordinates
H_gal = q_acquisition_gal.get_operator()
invN_gal = IdentityOperator()

# Local coordinates
H_local = q_acquisition_local.get_operator()
invN_local = IdentityOperator()

### Full MM
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

invN = invN_gal

tod = H(true_maps)  # noiseless: no noise term is added here

### Map-making
# preconditioner
ncomp = 2
no_det = 992

stacked_dptdp_inv = np.empty((ncomp, npix))

q_acq = [q_acquisition_gal, q_acquisition_local]

for icomp in range(ncomp):
    H_qubic = q_acq[icomp].operator

    stacked_dptdp_inv_nsub = np.empty((params["nsub_in"], npix))

    for j_nsub in range(params["nsub_in"]):
        H_single = H_qubic[j_nsub]

        D = H_single.operands[1]
        P = H_single.operands[4]
        sh = P.matrix.data.index.shape

        point_per_det = sh[0] // no_det
        mapPtP_perdet_seq = np.empty((no_det, npix))

        for det in range(no_det):
            start, end = det * point_per_det, (det + 1) * point_per_det
            indices = P.matrix.data.index[start:end, :]
            weights = P.matrix.data.r11[start:end, :]
            flat_indices = indices.ravel()
            flat_weights = weights.ravel()

            mapPitPi = np.bincount(flat_indices, weights=flat_weights**2, minlength=npix)
            mapPtP_perdet_seq[det, :] = mapPitPi

        D_sq = D.data**2
        mapPtP_seq_scaled = D_sq[:, np.newaxis] * mapPtP_perdet_seq
        dptdp = mapPtP_seq_scaled.sum(axis=0)

        # Safe inversion
        dptdp_inv = np.zeros_like(dptdp)
        nonzero = dptdp != 0
        dptdp_inv[nonzero] = 1.0 / dptdp[nonzero]
        stacked_dptdp_inv_nsub[j_nsub] = dptdp_inv

    stacked_dptdp_inv[icomp] = stacked_dptdp_inv_nsub.sum(axis=0)

preconditioner = BlockDiagonalOperator(
    [DiagonalOperator(ci, broadcast="rightward") for ci in stacked_dptdp_inv],
    new_axisin=0,
)

# Ax=b equation to be solved by PCG
A = H.T * invN * H
b = H.T * invN * tod
x0 = np.zeros_like(true_maps)

# Attention: reconstruire uniquement les pixels a l'interieur du patch (cf cas avec Planck dans le FMM)

# Run PCG
algo = PCGAlgorithm(
    A,
    b,
    comm,
    x0=x0,
    tol=1e-12,
    maxiter=1000,
    disp=True,
    M=None,
    center=[0, -57],
    reso=15,
    seenpix=seenpix,
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
    "result": result,
    "true_maps": true_maps,
    "seenpix": seenpix,
    "center_gal": center_gal,
    "center_local": center_local,
    "min_input": min_input,
    "max_input": max_input,
    "success": success,
    "message": message,
}

with open("solution_cmb.pkl", "wb") as pkl_file:
    pkl.dump(dict_solution, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
