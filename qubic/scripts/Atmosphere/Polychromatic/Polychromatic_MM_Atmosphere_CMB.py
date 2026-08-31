import os
import pickle as pkl

import healpy as hp
import matplotlib.pyplot as plt
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
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator, Spherical2HealpixOperator

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

plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

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

az, el = q_sampling_gal.azimuth, q_sampling_gal.elevation

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

# Azimuth plot
axs[0].plot(az)
axs[0].set_title("Azimuth")
axs[0].set_xlabel("Time samples")
axs[0].set_ylabel("Angles (degrees)")

# Elevation plot
axs[1].plot(el)
axs[1].set_title("Elevation")
axs[1].set_xlabel("Time samples")
axs[1].set_ylabel("Angles (degrees)")

# Scanning strategy plot
axs[2].plot(az, el)
axs[2].set_title("Scanning strategy")
axs[2].set_xlabel("Azimuth (degrees)")
axs[2].set_ylabel("Elevation (degrees)")

# Equatorial coordinates plot
axs[3].plot(
    (q_sampling_gal.equatorial[:, 0] + 180) % 360 - 180, q_sampling_gal.equatorial[:, 1]
)
axs[3].set_title("Equatorial coordinates")
axs[3].set_xlabel("Right ascension (degrees)")
axs[3].set_ylabel("Declination (degrees)")

# Galactic coordinates plot
axs[4].plot(q_sampling_gal.galactic[:, 0], q_sampling_gal.galactic[:, 1])
axs[4].set_title("Galactic coordinates")
axs[4].set_xlabel("Longitude (degrees)")
axs[4].set_ylabel("Latitude (degrees)")

fig.suptitle("Qubic Sampling")
plt.tight_layout()
fig.savefig(f"{plot_dir}/scanning_strategy.png")
plt.close(fig)

test_gal = np.zeros(hp.nside2npix(params["nside"]))
index = np.array(
    Spherical2HealpixOperator(params["nside"], "azimuth, elevation")(
        np.radians(q_sampling_gal.galactic)
    ),
    dtype="int",
)
test_gal[index] = 1
hp.mollview(test_gal, title="test_gal", cmap="viridis")
plt.savefig(f"{plot_dir}/test_gal_mollview.png")
plt.close()
hp.gnomview(test_gal, title="test_gal", cmap="viridis", reso=15, rot=center_gal)
plt.savefig(f"{plot_dir}/test_gal_gnomview.png")
plt.close()

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

test_local = np.zeros(hp.nside2npix(params["nside"]))
index = np.array(
    Spherical2HealpixOperator(params["nside"], "azimuth, elevation")(
        np.radians([q_sampling_local.azimuth, q_sampling_local.elevation]).T
    ),
    dtype="int",
)
test_local[index] = 1
hp.mollview(test_local, title="test_local", cmap="viridis")
plt.savefig(f"{plot_dir}/test_local_mollview.png")
plt.close()
hp.gnomview(
    test_local,
    title="test_local",
    cmap="viridis",
    reso=15,
    rot=(np.mean(q_sampling_local.azimuth), np.mean(q_sampling_local.elevation)),
)
plt.savefig(f"{plot_dir}/test_local_gnomview.png")
plt.close()

### Input maps
# CMB
cl_cmb = CMBModel(None).give_cl_cmb(r=0, Alens=1)
cmb_map = hp.synfast(cl_cmb, params["nside"], new=True, verbose=False).T

cmb_maps = np.ones((params["nsub_in"], hp.nside2npix(params["nside"]), 3))
cmb_maps *= cmb_map[None]

hp.mollview(cmb_map[:, 0], cmap="jet", title="CMB map", unit=r"$µK_{CMB}$")
plt.savefig(f"{plot_dir}/cmb_map.png")
plt.close()

# Atmosphere
atm_maps = np.zeros(cmb_maps.shape)
atm_maps[..., 0] = atm.get_temp_maps(atm.delta_rho_map) / 1e3

index_nu = 0
hp.mollview(
    atm_maps[index_nu, :, 0],
    cmap="jet",
    unit="µK_CMB",
    title="Atmosphere map {:.2f} GHz".format(atm.frequencies[index_nu]),
)
plt.savefig(f"{plot_dir}/atm_map_first_freq_mollview.png")
plt.close()
hp.gnomview(
    atm_maps[index_nu, :, 0],
    rot=center_local,
    reso=20,
    title="Atmosphere map {:.2f} GHz".format(atm.frequencies[index_nu]),
    unit=r"$µK_{CMB}$",
    cmap="jet",
)
plt.savefig(f"{plot_dir}/atm_map_first_freq_gnomview.png")
plt.close()

index_nu = -1
hp.mollview(
    atm_maps[index_nu, :, 0],
    cmap="jet",
    unit="µK_CMB",
    title="Atmosphere map {:.2f} GHz".format(atm.frequencies[index_nu]),
)
plt.savefig(f"{plot_dir}/atm_map_last_freq_mollview.png")
plt.close()
hp.gnomview(
    atm_maps[index_nu, :, 0],
    rot=center_local,
    reso=20,
    title="Atmosphere map {:.2f} GHz".format(atm.frequencies[index_nu]),
    unit=r"$µK_{CMB}$",
    cmap="jet",
)
plt.savefig(f"{plot_dir}/atm_map_last_freq_gnomview.png")
plt.close()

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

hp.mollview(coverage_gal, title="Galactic Coverage")
plt.savefig(f"{plot_dir}/coverage_gal_mollview.png")
plt.close()
hp.gnomview(coverage_gal, rot=center_gal, reso=20, title="Galactic Coverage")
plt.savefig(f"{plot_dir}/coverage_gal_gnomview.png")
plt.close()
hp.mollview(coverage_local, title="Local Coverage")
plt.savefig(f"{plot_dir}/coverage_local_mollview.png")
plt.close()
hp.gnomview(coverage_local, rot=center_local, reso=20, title="Local Coverage")
plt.savefig(f"{plot_dir}/coverage_local_gnomview.png")
plt.close()

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

plt.plot(result["convergence"])
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Convergence")
plt.savefig(f"{plot_dir}/convergence.png")
plt.close()

input = true_maps.copy()
output = result["x"].copy()
residual = output - input
stk = ["I", "Q", "U"]

# One triptych (Input / Output / Residual) per reconstructed Stokes parameter -- I (istk=0)
# and Q (istk=1); U isn't reconstructed here. Residual color scale is +/- 3 sigma of the
# residual within the seen pixels of each map.
for istk in [0, 1]:
    plt.figure(figsize=(15, 12))
    k = 1
    reso = 20

    for imap in range(input.shape[0]):
        if imap == 0:
            map_name = "CMB"
            center = center_gal
            input[imap, ~seenpix_gal, :] = hp.UNSEEN
            output[imap, ~seenpix_gal, :] = hp.UNSEEN
            residual[imap, ~seenpix_gal, :] = hp.UNSEEN
        else:
            map_name = "Atm"
            center = center_local
            input[imap, ~seenpix_local, :] = hp.UNSEEN
            output[imap, ~seenpix_local, :] = hp.UNSEEN
            residual[imap, ~seenpix_local, :] = hp.UNSEEN

        sigma = np.std(residual[imap, seenpix[imap], istk])
        nsigma = 3

        hp.gnomview(
            input[imap, :, istk],
            reso=reso,
            rot=center,
            min=min_input[imap, istk],
            max=max_input[imap, istk],
            cmap="jet",
            sub=(input.shape[0], 3, k),
            title=f"{stk[istk]} - Input - {map_name}",
            notext=True,
        )
        hp.gnomview(
            output[imap, :, istk],
            reso=reso,
            rot=center,
            min=min_input[imap, istk],
            max=max_input[imap, istk],
            cmap="jet",
            sub=(input.shape[0], 3, k + 1),
            title=f"{stk[istk]} - Output - {map_name}",
            notext=True,
        )
        hp.gnomview(
            residual[imap, :, istk],
            reso=reso,
            rot=center,
            min=-nsigma * sigma,
            max=nsigma * sigma,
            cmap="jet",
            sub=(input.shape[0], 3, k + 2),
            title=f"{stk[istk]} - Residual - {map_name}",
            notext=True,
        )
        k += 3

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/maps_triptych_{stk[istk]}.png")
    plt.close()

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
