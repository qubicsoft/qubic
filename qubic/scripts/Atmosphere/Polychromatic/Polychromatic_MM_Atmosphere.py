import os
import pickle as pkl

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyoperators import MPI, BlockDiagonalOperator, DiagonalOperator, ReshapeOperator
from pyoperators.iterative.core import AbnormalStopIteration
from pysimulators.interfaces.healpy import Spherical2HealpixOperator

from qubic.lib.Instrument.Qacquisition import QubicInstrumentType
from qubic.lib.Instrument.Qnoise import QubicTotNoise
from qubic.lib.MapMaking.Qatmosphere import AtmosphereMaps
from qubic.lib.MapMaking.Qcg_test_for_atm import PCGAlgorithm
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Import simulation parameters
with open("params.yml", "r") as file:
    params = yaml.safe_load(file)

plot_dir = "Plots"
os.makedirs(plot_dir, exist_ok=True)

# Call the class which builds the atmosphere maps
atm = AtmosphereMaps(params)
qubic_dict = atm.qubic_dict

qubic_dict["instrument_type"] = "UWB"
qubic_dict["interp_projection"] = False

qubic_dict["effective_duration"] = None
qubic_dict["effective_duration150"] = None
qubic_dict["effective_duration220"] = None

### Scanning strategy
qubic_dict["random_pointing"] = True
qubic_dict["date_obs"] = "2023-10-01 22:57:00.000"
qubic_dict["period"] = 3
qubic_dict["fix_azimuth"]["apply"] = True

if params["sweeping_pointings"]:
    qubic_dict["random_pointing"] = False
    qubic_dict["sweeping_pointing"] = True

    qubic_dict["angspeed"] = 0.1
    qubic_dict["delta_az"] = 20
    qubic_dict["nsweeps_per_elevation"] = 3
    qubic_dict["duration"] = 3
    qubic_dict["period"] = 3600 * qubic_dict["duration"] / params["npointings"]

q_sampling = get_pointing(qubic_dict)
q_scene = QubicScene(qubic_dict)
center = np.array([np.mean(q_sampling.azimuth), np.mean(q_sampling.elevation)])

### Plot scanning strategy
az, el = q_sampling.azimuth, q_sampling.elevation

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
    (q_sampling.equatorial[:, 0] + 180) % 360 - 180, q_sampling.equatorial[:, 1]
)
axs[3].set_title("Equatorial coordinates")
axs[3].set_xlabel("Right ascension (degrees)")
axs[3].set_ylabel("Declination (degrees)")

# Galactic coordinates plot
axs[4].plot(q_sampling.galactic[:, 0], q_sampling.galactic[:, 1])
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
        np.radians([q_sampling.azimuth, q_sampling.elevation]).T
    ),
    dtype="int",
)
test_gal[index] = 1
hp.mollview(test_gal, title="test_local", cmap="viridis")
plt.savefig(f"{plot_dir}/test_local_mollview.png")
plt.close()
hp.gnomview(
    test_gal,
    title="test_local",
    cmap="viridis",
    reso=15,
    rot=(np.mean(q_sampling.azimuth), np.mean(q_sampling.elevation)),
)
plt.savefig(f"{plot_dir}/test_local_gnomview.png")
plt.close()

### Atmosphere maps
# Import the atm absorption spectrum
abs_spectrum = atm.absorption_spectrum()

plt.plot(atm.integration_frequencies, abs_spectrum)
plt.ylim(0, 0.0002)
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"Absorption ($m^{2}/g$)")
plt.title("Atmospheric Absorption Spectrum")
plt.savefig(f"{plot_dir}/absorption_spectrum.png")
plt.close()

atm_maps = np.zeros((len(atm.frequencies), hp.nside2npix(params["nside"]), 3))
atm_maps[..., 0] = atm.get_temp_maps(atm.delta_rho_map)

index_nu = 0
hp.mollview(
    atm_maps[index_nu, :, 0],
    cmap="jet",
    unit="µK_CMB",
    title="Atmosphere map {:.2f} GHz".format(atm.frequencies[index_nu]),
)
plt.savefig(f"{plot_dir}/atm_map_first_freq.png")
plt.close()

# Import the atm integrated absorption spectrum
integrated_abs_spectrum, frequencies, bandwidth = atm.integrated_absorption_spectrum()

mean_atm_maps = atm.get_mean_atm_temperature()
plt.figure()
plt.plot(frequencies, mean_atm_maps, ".")
plt.title("Atmosphere maps spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"Mean temperature ($\mu K_{CMB}$)")
plt.savefig(f"{plot_dir}/atm_mean_temperature_spectrum.png")
plt.close()
plt.figure()
plt.plot(frequencies, integrated_abs_spectrum, ".")
plt.xlabel("Frequency (GHz)")
plt.ylabel(r"Integrated absorption spectrum ($m^{2}/g$)")
plt.title("Integrated absorption spectrum")
plt.savefig(f"{plot_dir}/atm_integrated_absorption_spectrum.png")
plt.close()

# Spectral weight of each sub-band relative to a 150 GHz reference (the atmosphere's
# brightness varies by a factor of several across nsub_in sub-bands, so a single
# achromatic reconstructed map needs this mixing matrix to account for it).
ref_freq = 150
ref_idx = np.argmin(np.abs(atm.frequencies - ref_freq))
mixing_matrix = atm.get_atm_mixing_matrix(ref_freq=ref_freq)[:, None]

true_maps = atm_maps[ref_idx : ref_idx + 1, :, :]

# used by the preconditioner below
fsub = params["nsub_in"] // params["nrec"]

### Map-making
# Build the QUBIC operators
H_tod = QubicInstrumentType(
    atm.qubic_dict, nsub=params["nsub_in"], nrec=params["nsub_in"]
).get_operator()

qubic_noise = QubicTotNoise(qubic_dict, q_sampling, q_scene)

tod = H_tod(atm_maps).ravel() + qubic_noise.total_noise(
    params["wdet"], params["wpho150"], params["wpho220"], seed_noise=params["seed"]
).ravel()

del H_tod

Qacq = QubicInstrumentType(atm.qubic_dict, nsub=params["nsub_in"], nrec=params["nrec"])

invN = Qacq.get_invntt_operator(params["wdet"], params["wpho150"], params["wpho220"])

# mixing_matrix scales each sub-band's contribution by the atmosphere's known spectrum,
# so H_rec reconstructs a single spatial template (ncomp=1) rather than an achromatic map.
H_rec = Qacq.get_operator(A=mixing_matrix)

coverage = Qacq.coverage
covnorm = coverage / coverage.max()
seenpix = covnorm > params["coverage_cut"]

# Build PCG
R = ReshapeOperator(tod.shape, H_rec.shapeout)
R_invN = ReshapeOperator(H_rec.shapeout, invN.shapein)
A = H_rec.T * R_invN.T * invN * R_invN * H_rec
b = H_rec.T * R_invN.T * invN * R_invN * R(tod)
x0 = true_maps * 0.0

hp.mollview(x0[0, :, 0], title="Initial guess", cmap="jet", unit="µK_CMB")
plt.savefig(f"{plot_dir}/initial_guess_mollview.png")
plt.close()
hp.mollview(true_maps[0, :, 0], title="True map", cmap="jet", unit="µK_CMB")
plt.savefig(f"{plot_dir}/true_map_mollview.png")
plt.close()

# preconditioner
# Note about preconditioner: stacked_dptdp_inv should have the shape (Nrec, Npix). But, we
# can compute that from H, which contains Nsub acquisition operators. In the next block, only
# the first Nrec operators are used rather than Nsub, because it's not clear how to reduce
# them. Computing it with another H which had exactly Nrec sub-operators didn't work either.
# We need to find a solution to this problem.
nrec = params["nrec"]
npix = 12 * params["nside"] ** 2
no_det = 992

stacked_dptdp_inv = np.empty((nrec, npix))

H_qubic = Qacq.operator

stacked_dptdp_inv_nsub = np.empty((fsub, npix))

for irec in range(nrec):
    for j_fsub in range(fsub):
        isub = irec * fsub + j_fsub
        H_single = H_qubic[isub]

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
        # scale by this sub-band's mixing-matrix weight squared, since H_rec now weights
        # each sub-band's contribution by mixing_matrix[isub] (see H_rec above)
        dptdp = mapPtP_seq_scaled.sum(axis=0) * mixing_matrix[isub, 0] ** 2

        # Safe inversion
        dptdp_inv = np.zeros_like(dptdp)
        nonzero = dptdp != 0
        dptdp_inv[nonzero] = 1.0 / dptdp[nonzero]
        stacked_dptdp_inv_nsub[j_fsub] = dptdp_inv

    stacked_dptdp_inv[irec] = stacked_dptdp_inv_nsub.sum(axis=0)

preconditioner = BlockDiagonalOperator(
    [DiagonalOperator(ci, broadcast="rightward") for ci in stacked_dptdp_inv],
    new_axisin=0,
)

# Run PCG
algo = PCGAlgorithm(
    A,
    b,
    comm,
    x0=x0,
    tol=1e-10,
    maxiter=200,
    disp=True,
    M=None,
    center=[0, -57],
    reso=15,
    seenpix=seenpix,
    input=true_maps,
)
try:
    output = algo.run()
    success = True
    message = "Success"
except AbnormalStopIteration as e:
    output = algo.finalize()
    success = False
    message = str(e)

plt.plot(output["convergence"])
plt.title("Polychromatic")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Convergence")
plt.savefig(f"{plot_dir}/convergence.png")
plt.close()

plt.figure(figsize=(12, 12), dpi=200)
k = 1

res_maps = output["x"] - true_maps
true_maps[:, ~seenpix, :] = hp.UNSEEN
output["x"][:, ~seenpix, :] = hp.UNSEEN
res_maps[:, ~seenpix, :] = hp.UNSEEN

stk = ["I", "Q", "U"]
istk = 0
n_sig = 3
reso = 10

for inu in range(output["x"].shape[0]):
    sigma = np.std(true_maps[inu, seenpix, istk])
    hp.gnomview(
        true_maps[inu, :, istk],
        min=np.min(true_maps[inu, seenpix, istk]),
        max=np.max(true_maps[inu, seenpix, istk]),
        cmap="jet",
        rot=center,
        title="{} - Input".format(stk[istk]),
        reso=reso,
        sub=(output["x"].shape[0], 3, k),
        notext=True,
    )
    hp.gnomview(
        output["x"][inu, :, istk],
        min=np.min(true_maps[inu, seenpix, istk]),
        max=np.max(true_maps[inu, seenpix, istk]),
        cmap="jet",
        rot=center,
        title="{} - Output".format(stk[istk]),
        reso=reso,
        sub=(output["x"].shape[0], 3, k + 1),
        notext=True,
    )
    hp.gnomview(
        res_maps[inu, :, istk],
        cmap="jet",
        rot=center,
        title="{} - Residual".format(stk[istk]),
        reso=reso,
        sub=(output["x"].shape[0], 3, k + 2),
        notext=True,
    )
    k += 3

plt.savefig(f"{plot_dir}/maps_triptych.png")
plt.close()

for inu in range(output["x"].shape[0]):
    sigma = np.std(true_maps[inu, seenpix, istk])
    hp.mollview(
        res_maps[inu, :, istk],
        cmap="jet",
        title="{} - Residual - {:.2f} GHz".format(stk[istk], ref_freq),
    )
    plt.savefig(f"{plot_dir}/residual_mollview_sub{inu}.png")
    plt.close()

dict_solution = {
    "output": output,
    "true_maps": true_maps,
    "seenpix": seenpix,
    "center": center,
    "mixing_matrix": mixing_matrix,
    "ref_freq": ref_freq,
    "success": success,
    "message": message,
}

with open("solution.pkl", "wb") as pkl_file:
    pkl.dump(dict_solution, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
