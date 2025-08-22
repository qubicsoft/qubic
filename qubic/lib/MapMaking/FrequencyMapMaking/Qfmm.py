### General packages
import os
import pickle
import time
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from fgbuster.component_model import CMB, Dust, Synchrotron
from pyoperators import (BlockDiagonalOperator,
                         DiagonalOperator,
                         ReshapeOperator,
                         IdentityOperator,
                         ConvolutionOperator,
                         MPI)
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from scipy.optimize import minimize

from qubic.lib.Instrument.Qacquisition import JointAcquisitionFrequencyMapMaking, PlanckAcquisition
from qubic.lib.Instrument.Qnoise import QubicTotNoise
from qubic.lib.MapMaking.FrequencyMapMaking.FMM_errors_checking import ErrorChecking
from qubic.lib.MapMaking.Qcg import pcg
from qubic.lib.MapMaking.Qmap_plotter import PlotsFMM
from qubic.lib.MapMaking.Qmaps import InputMaps, PlanckMaps
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qfoldertools import create_folder_if_not_exists, do_gif
from qubic.lib.Qmpi_tools import MpiTools
from qubic.lib.Qsamplings import equ2gal
from qubic.lib.Qspectra import Spectra

__all__ = ["PipelineFrequencyMapMaking", "PipelineEnd2End"]


class PipelineFrequencyMapMaking:
    """
    Instance to reconstruct frequency maps using QUBIC abilities.

    Parameters
        ----------
        comm :
            MPI communicator
        file : str
            used to create the forlder for data saving
        params : dict
            dictionary containing all the simulation parameters

    """

    def __init__(self, comm, file, parameters_dict):
        """
        Initialise PipelineFrequencyMapMaking

        """

        ### MPI common arguments
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.job_id = os.environ.get("SLURM_JOB_ID")

        self.mpi = MpiTools(self.comm)
        self.mapmaking_time_0 = time.time()

        ### Parameters file
        self.params = parameters_dict.copy()

        ### Sky configuration
        self.skyconfig = self.get_sky_config()

        ### fsub
        self.fsub_in = int(self.params["QUBIC"]["nsub_in"] / self.params["QUBIC"]["nrec"])
        self.fsub_out = int(self.params["QUBIC"]["nsub_out"] / self.params["QUBIC"]["nrec"])

        self.file = file
        self.plot_folder = "FMM/" + self.params["path_out"] + "png/"

        ### Create folders
        create_folder_if_not_exists(self.comm, "FMM/" + self.params["path_out"] + "maps/")
        create_folder_if_not_exists(self.comm, "FMM/" + self.params["path_out"] + "png/")

        ### Center of the QUBIC patch
        self.center = equ2gal(self.params["SKY"]["RA_center"], self.params["SKY"]["DEC_center"])

        ## Sky
        self.dict_in = self.get_dict(key="in")
        self.dict_out = self.get_dict(key="out")
        # change config and detector_nep

        ### Joint acquisition for TOD making
        self.joint_tod = JointAcquisitionFrequencyMapMaking(
            self.dict_in,
            # self.params["QUBIC"]["instrument"],
            self.params["QUBIC"]["nsub_in"], # TODO: dovrebbe essere n_rec
            self.params["QUBIC"]["nsub_in"], 
            H=None,
        )

        ### Joint acquisition
        if self.params["QUBIC"]["nsub_in"] == self.params["QUBIC"]["nsub_out"]:
            H = self.joint_tod.qubic.H
        else:
            H = None

        self.joint = JointAcquisitionFrequencyMapMaking(
            self.dict_out,
            # self.params["QUBIC"]["instrument"],
            self.params["QUBIC"]["nrec"],
            self.params["QUBIC"]["nsub_out"],
            H=H,
        )

        ### Ensure that all processors have the same external dataset
        self.externaldata = PlanckMaps(self.skyconfig, self.joint_tod.qubic.allnus, self.params["QUBIC"]["nrec"],
                                       nside=self.params["SKY"]["nside"])
        if self.rank == 0:
            self.externaldata.maps, self.externaldata.maps_noise = self.externaldata.run(
                use_fwhm=self.params["QUBIC"]["convolution_in"])
        else:
            self.externaldata.maps = None
            self.externaldata.maps_noise = None

        self.externaldata.maps = self.comm.bcast(self.externaldata.maps, root=0)
        self.externaldata.maps_noise = self.comm.bcast(self.externaldata.maps_noise, root=0)

        self.planck_acquisition = []
        for band_pl in [143, 217]:
            self.planck_acquisition.append(PlanckAcquisition(band_pl, self.joint.qubic.scene))

        self.nus_Q = self.get_averaged_nus()

        ### Coverage map
        self.coverage = self.joint.qubic.subacqs[0].get_coverage()
        self.seenpix = self.coverage / self.coverage.max() > self.params["SKY"]["coverage_cut"]
        self.seenpix_qubic = self.coverage / self.coverage.max() > 0
        self.mask = np.ones(12 * self.params["SKY"]["nside"] ** 2)
        self.mask[self.seenpix] = self.params["PLANCK"]["weight_planck"]

        ### Angular resolutions
        self.fwhm_in, self.fwhm_out, self.fwhm_rec = self.get_convolution()

        ### Build the Input Maps
        self.maps_input = InputMaps(
            self.skyconfig,
            self.joint_tod.qubic.allnus,
            self.params["QUBIC"]["nrec"],
            nside=self.params["SKY"]["nside"],
            corrected_bandpass=self.params["QUBIC"]["bandpass_correction"],
        )

        ps_cfg = self.params.get("POINT_SOURCE", {})
        self._ps_added = False
        self.ps_mnu = None
        
        if ps_cfg.get("enabled", False):
            self.ps_mnu = self.build_point_source(
                lon_deg=float(self.center[0]),
                lat_deg=float(self.center[1]),
                stokes_uK=tuple(ps_cfg.get("stokes_uK", (1000.0, 0.0, 0.0))),
                radius_arcmin=float(ps_cfg.get("radius_arcmin", 20.0)),
            )

            self.maps_input.m_nu = self.maps_input.m_nu + self.ps_mnu
            self._ps_added = True

            # Plot th input map 
            stokes = ["I", "Q", "U"]
            rows = 2   # number of bands to show as rows (0 and 1)
            cols = 3   # I, Q, U as columns

            fig = plt.figure(figsize=(4 * cols, 4 * rows), tight_layout=True)

            for i in range(rows):  # protect in case q > 2
                for s, st in enumerate(stokes):
                    # panel index increases row-wise
                    sub_index = i * cols + s + 1
                    hp.gnomview(self.maps_input.m_nu[i, :, s],
                                rot=self.center,
                                reso=10,
                                xsize=200,
                                ysize=200,
                                sub=(rows, cols, sub_index),
                                title=rf"${st} \, band \, {i}$",
                                notext=False)

            plt.savefig("input_maps_w_ps.png", bbox_inches="tight", dpi=1000)
            plt.close(fig)

         ### Convolve the Nsub input maps at QUBIC resolution
        for i in range(len(self.fwhm_in)):
            C = HealpixConvolutionGaussianOperator(self.fwhm_in[i], lmax=3 * self.params["SKY"]["nside"] - 1)
            self.maps_input.m_nu[i] = C(self.maps_input.m_nu[i])

        ## Initial maps
        self.m_nu_in = self.get_input_map(m_nu=self.maps_input.m_nu)

        ### Define reconstructed and TOD operator
        self.get_H()

        ### Inverse noise covariance matrix
        if self.params["PLANCK"]["external_data"]:
            self.invN = self.joint.get_invntt_operator(
                mask=self.mask,
                qubic_ndet=self.params["QUBIC"]["NOISE"]["ndet"],
                qubic_npho150=self.params["QUBIC"]["NOISE"]["npho150"],
                qubic_npho220=self.params["QUBIC"]["NOISE"]["npho220"],
                planck_ntot=self.params["PLANCK"]["level_noise_planck"],
            )
        else:
            self.invN = self.joint.qubic.get_invntt_operator(self.params["QUBIC"]["NOISE"]["ndet"],
                                                             self.params["QUBIC"]["NOISE"]["npho150"],
                                                             self.params["QUBIC"]["NOISE"]["npho220"])
            R = ReshapeOperator(self.invN.shapeout, self.invN.shape[0])
            self.invN = R(self.invN(R.T))

        self.ITF = IdentityOperator()
        # Instrument Transfer Function (Map-Making-Level deconvolution)
        if self.params["QUBIC"]['deconvolution']:
            dt = self.joint_tod.qubic.dict['period']
            tau = self.joint_tod.qubic.dict['detector_tau']

            print(f"{dt = } | {tau =}")

            # Choose the kernel length L so that the exponential tail we cut off
            # is numerically negligible.
            #
            #   G[k] = exp(-k * dt / tau)  #  response of a 1-pole bolometer
            #
            # The contribution beyond k = k_cut drops like exp(-k_cut).
            # By taking
            #       L = ceil( k * tau / dt )      with k ≈ 5-6,
            #
            # we keep the kernel out to t = k τ, where its amplitude is
            #     exp(-k)  ≤ 1%  ( ≈ 0.7 % for k=5, 0.25 % for k=6 )
            # and the power we discard is exp(-2k) ≲ 10⁻⁴.
            # This gives a good compromise: the truncated kernel is short
            # (so the Toeplitz bandwidth and mat-vec cost stay low) while the error
            # introduced by the missing tail is far below detector noise

            L = int(np.ceil(5 * tau / dt))
            kernel = np.exp(-np.arange(L) * dt / tau)
            kernel /= np.sum(kernel)  # normalizzazione
            # Number of QUBIC time samples
            self.bolometer_kernel = kernel 
            Nt_q = self.H_in_qubic.shapeout[0] if isinstance(self.H_in_qubic.shapeout,
                                                             tuple) else self.H_in_qubic.shapeout
            # self.ITF = SymmetricBandToeplitzOperator(
            #     firstrow=kernel,
            #     shapein=(Nt_q,),
            #     shapeout=(Nt_q,),
            # )

            self.ITF = ConvolutionOperator(kernel, axes=(0,), shapein=(Nt_q,), shapeout=(Nt_q,))

            nrec = self.params["QUBIC"]["nrec"]
            nside = self.params["SKY"]["nside"]
            npix = 12 * nside ** 2

            Nt_p = 3 * npix * max(nrec, 2)

            if self.params["PLANCK"]["external_data"]:
                # Build a block‑diagonal operator: ITF for QUBIC (Nt_q × Nt_q)
                # followed by an identity for the Planck part (Nt_p × Nt_p).
                # We concatenate along the time‑sample axis (axis 0) and give PyOperators
                # the exact size of each block so it does not have to guess.
                self.ITF = BlockDiagonalOperator(
                    (self.ITF, IdentityOperator(shapein=(Nt_p,), shapeout=(Nt_p,))),
                    axisin=0,
                    axisout=0,
                    partitionin=(Nt_q, Nt_p),
                    # partitionout=(Nt_q, Nt_p),
                )

        # Keep a handle to the QUBIC‑only part of the operator for TOD generation.
        if self.params["PLANCK"]["external_data"]:
            if self.params["QUBIC"]['deconvolution']:
                # Deconvolution ON: first operand is the SymmetricBandToeplitzOperator
                # that acts only on the QUBIC samples.
                self.ITF_QUBIC = self.ITF.operands[0]
            else:
                # Deconvolution OFF: use a simple identity on the QUBIC TOD length.
                Nt_q = self.H_in_qubic.shapeout[0] if isinstance(self.H_in_qubic.shapeout,
                                                                 tuple) else self.H_in_qubic.shapeout
                self.ITF_QUBIC = IdentityOperator(shapein=(Nt_q,), shapeout=(Nt_q,))
        else:
            # No external Planck data: ITF already acts only on the QUBIC samples.
            self.ITF_QUBIC = self.ITF

        ### Noises

        rng_noise_planck = np.random.default_rng(self.params["PLANCK"]["seed_noise"])
        self.noise_planck = []
        for i in range(2):
            self.noise_planck.append(
                self.planck_acquisition[i].get_noise(rng_noise_planck) * self.params["PLANCK"]["level_noise_planck"])

        qubic_noise = QubicTotNoise(self.dict_out, self.joint.qubic.sampling, self.joint.qubic.scene)

        self.noiseq = qubic_noise.total_noise(
            self.params["QUBIC"]["NOISE"]["ndet"],
            self.params["QUBIC"]["NOISE"]["npho150"],
            self.params["QUBIC"]["NOISE"]["npho220"],
            seed_noise=self.params["QUBIC"]["NOISE"]["seed_noise"],
        ).ravel()

        ### Initialize plot instance
        self.plots = PlotsFMM(self.seenpix)

    def get_components_fgb(self):
        """Components FGbuster

        Method to build a dictionary containing all the wanted components to generate sky maps.
        Based on FGBuster.

        Returns
        -------
        dict_comps: dict
            Dictionary containing the component instances.

        """

        dict_comps = []

        if self.params["CMB"]["cmb"]:
            dict_comps += [CMB()]

        if self.params["Foregrounds"]["Dust"]:
            dict_comps += [Dust(nu0=150, beta_d=1.54, temp=20)]

        if self.params["Foregrounds"]["Synchrotron"]:
            dict_comps += [Synchrotron(nu0=150, beta_pl=-3)]

        return dict_comps
        

    def build_point_source(self,
                           lon_deg: float,
                           lat_deg: float,
                           stokes_uK=(1000.0, 0.0, 0.0),
                           radius_arcmin: float = 5.0):
        """
        Create a small "point" source (top-hat disk) map cube with the same
        shape as `self.maps_input.m_nu` (N_sub, N_pix, 3), without modifying
        the existing maps. The total integrated Stokes (I, Q, U) equals
        `stokes_uK` in µK_CMB, uniformly distributed over the selected pixels.

        Parameters
        ----------
        lon_deg, lat_deg : float
            Galactic longitude and latitude in degrees.
        stokes_uK : tuple(float, float, float)
            Integrated (I, Q, U) amplitudes in µK_CMB.
        radius_arcmin : float
            Disk radius in arcminutes (keep small to mimic a point source).
        """

        nside = int(self.params["SKY"]["nside"])  # HEALPix nside
        v0 = hp.ang2vec(lon_deg, lat_deg, lonlat=True)
        r_disc = np.deg2rad(radius_arcmin / 60.0)

        pix = hp.query_disc(nside, v0, r_disc, inclusive=False)
        if np.size(pix) == 0:
            single_pix = hp.ang2pix(nside, lon_deg, lat_deg, lonlat=True)
            pix = np.array([single_pix], dtype=int)

        # Uniform weights normalized to sum to 1
        w = np.ones(pix.shape[0], dtype=float)
        w = w / w.sum()

        st = np.asarray(stokes_uK, dtype=float)
        if st.ndim == 0:
            st = np.array([float(st), 0.0, 0.0])

        ps = np.zeros_like(self.maps_input.m_nu)
        Nsub = ps.shape[0]
        for i in range(Nsub):
            ps[i, pix, 0] += st[0] * w
            ps[i, pix, 1] += st[1] * w
            ps[i, pix, 2] += st[2] * w
        return ps

    def get_H(self):
        """Acquisition operator

        Method to compute QUBIC acquisition operators.

        """

        ### QUBIC Pointing matrix for TOD generation
        self.H_in_qubic = self.joint_tod.qubic.get_operator()
        ### Pointing matrix for reconstruction
        if self.params["PLANCK"]["external_data"]:
            self.H_out_all_pix = self.joint.get_operator(fwhm=self.fwhm_out)
            self.H_out = self.joint.get_operator(fwhm=self.fwhm_out, seenpix=self.seenpix)
        else:
            self.H_out = self.joint.qubic.get_operator(fwhm=self.fwhm_out)

    def get_averaged_nus(self):
        """Average frequency

        Method to average QUBIC frequencies according to the number of reconstructed frequency maps.

        Returns
        -------
        nus_ave: array_like
            array containing the averaged QUBIC frequencies.

        """

        nus_ave = []
        for i in range(self.params["QUBIC"]["nrec"]):
            nus_ave += [np.mean(self.joint.qubic.allnus[i * self.fsub_out: (i + 1) * self.fsub_out])]

        return np.array(nus_ave)

    def get_sky_config(self):
        """Sky configuration.

        Method that read 'params.yml' file and create dictionary containing sky emission model.

        Returns
        -------
        dict_sky: dict
            Sky config dictionary.

        Notes
        -----
        Note that the key denote the emission and the value denote the sky model using PySM convention.
        For CMB, seed denote the realization.

        Example
        -------
        d = {'cmb':seed, 'dust':'d0', 'synchrotron':'s0'}

        """

        dict_sky = {}

        if self.params["CMB"]["cmb"]:
            if self.params["CMB"]["seed"] == 0:
                if self.rank == 0:
                    seed = np.random.randint(10000000)
                else:
                    seed = None
                seed = self.comm.bcast(seed, root=0)
            else:
                seed = self.params["CMB"]["seed"]

            dict_sky["cmb"] = seed

        for j in self.params["Foregrounds"]:
            if j == "Dust":
                if self.params["Foregrounds"][j]:
                    dict_sky["dust"] = "d0"
            elif j == "Synchrotron":
                if self.params["Foregrounds"][j]:
                    dict_sky["synchrotron"] = "s0"

        return dict_sky

    def get_dict(self, key="in"):
        """QUBIC dictionary.

        Method to modify the qubic dictionary.

        Parameters
        ----------
        key : str, optional
            Can be "in" or "out".
            It is used to build respectively the instances to generate the TODs or to reconstruct the sky maps,
            by default "in".

        Returns
        -------
        dict_qubic: dict
            Modified QUBIC dictionary.

        """

        args = {
            "npointings": self.params["QUBIC"]["npointings"],
            "nf_recon": self.params["QUBIC"]["nrec"],
            "nf_sub": self.params["QUBIC"][f"nsub_{key}"],  # here is the difference between in and out dictionaries
            "nside": self.params["SKY"]["nside"],
            "MultiBand": True,
            # "period": 1,
            "RA_center": self.params["SKY"]["RA_center"],
            "DEC_center": self.params["SKY"]["DEC_center"],
            "filter_nu": 220 * 1e9,
            "noiseless": False,
            "beam_shape": "gaussian",
            "comm": self.comm,
            "dtheta": self.params["QUBIC"]["dtheta"],
            "nprocs_sampling": 1,
            "nprocs_instrument": self.size,
            "photon_noise": True,
            "nhwp_angles": 3,
            #'effective_duration':3,
            "effective_duration150": 3,
            "effective_duration220": 3,
            "filter_relative_bandwidth": 0.25,
            "type_instrument": "wide",  # ?
            "TemperatureAtmosphere150": None,
            "TemperatureAtmosphere220": None,
            "EmissivityAtmosphere150": None,
            "EmissivityAtmosphere220": None,
            # mettre if ici pour fixer detector_nep
            "detector_nep": float(self.params["QUBIC"]["NOISE"]["detector_nep"]),
            "synthbeam_kmax": self.params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"],
            "synthbeam_fraction": self.params["QUBIC"]["SYNTHBEAM"]["synthbeam_fraction"],
            "interp_projection": False,
            "instrument_type": self.params["QUBIC"]["instrument"],
            "config": self.params["QUBIC"]["configuration"],
        }

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        dict_qubic = qubicDict()
        dict_qubic.read_from_file(dictfilename)

        for i in args.keys():
            dict_qubic[str(i)] = args[i]

        return dict_qubic

    def _get_scalar_acquisition_operator(self):
        """
        Function that will compute "scalar acquisition operatord" by applying the acquisition operators to a vector full of ones.
        These scalar operators will be used to compute the resolutions in the case where we do not add convolutions during reconstruction.
        """
        ### Import the acquisition operators
        acquisition_operators = self.joint.qubic.H

        ### Create the vector full of ones which will be used to compute the scalar operators
        vector_ones = np.ones(acquisition_operators[0].shapein)

        ### Apply each sub_operator on the vector
        scalar_acquisition_operators = np.empty(len(self.joint.qubic.allnus))
        for freq in range(len(self.joint.qubic.allnus)):
            scalar_acquisition_operators[freq] = np.mean(acquisition_operators[freq](vector_ones))
        return scalar_acquisition_operators

    def get_convolution(self):
        """QUBIC resolutions.

        Method to define expected QUBIC angular resolutions (radians) as function of frequencies.

        Returns
        -------
        fwhm_in: array_like
            Intrinsic resolutions, used to build the simulated TOD.
        fwhm_out: array_like
            Output resolutions. If we don't apply convolutions during reconstruction, array of zeros.
        fwhm_rec: array_like
            Reconstructed resolutions. Egal the output resolutions if we apply convolutions during reconstructions, evaluate through analytic formula otherwise.

        """

        ### Define FWHMs
        fwhm_in = np.zeros(self.params["QUBIC"]["nsub_in"])
        fwhm_out = np.zeros(self.params["QUBIC"]["nsub_out"])
        fwhm_rec = np.zeros(self.params["QUBIC"]["nrec"])

        ### FWHMs during map-making
        if self.params["QUBIC"]["convolution_in"]:
            fwhm_in = self.joint_tod.qubic.allfwhm.copy()
        if self.params["QUBIC"]["convolution_out"]:
            fwhm_out = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                fwhm_out = np.append(
                    fwhm_out,
                    np.sqrt(
                        self.joint.qubic.allfwhm[irec * self.fsub_out: (irec + 1) * self.fsub_out] ** 2 - np.min(
                            self.joint.qubic.allfwhm[irec * self.fsub_out: (irec + 1) * self.fsub_out]) ** 2
                    ),
                )

        ### Define reconstructed FWHM depending on the user's choice
        if self.params["QUBIC"]["convolution_in"] and self.params["QUBIC"]["convolution_out"]:
            fwhm_rec = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                fwhm_rec = np.append(fwhm_rec,
                                     np.min(self.joint.qubic.allfwhm[irec * self.fsub_out: (irec + 1) * self.fsub_out]))

        elif self.params["QUBIC"]["convolution_in"] and self.params["QUBIC"]["convolution_out"] is False:
            fwhm_rec = np.array([])
            scalar_acquisition_operators = self._get_scalar_acquisition_operator()

            if self.params["Foregrounds"]["Dust"]:
                f_dust = Dust(nu0=353, beta_d=1.54, temp=20)
                weight_factor = f_dust.eval(self.joint.qubic.allnus)

                def fun(nu):
                    return np.abs(fraction - f_dust.eval(nu))
            else:
                f_cmb = CMB()
                weight_factor = f_cmb.eval(self.joint.qubic.allnus)

                def fun(nu):
                    return np.abs(fraction - f_cmb.eval(nu))

            ### Compute expected resolutions and frequencies when not adding convolutions during reconstruction
            ### See FMM annexe B to understand the computations

            for irec in range(self.params["QUBIC"]["nrec"]):
                numerator_fwhm, denominator_fwhm = 0, 0
                numerator_nus, denominator_nus = 0, 0
                for jsub in range(irec * self.fsub_out, (irec + 1) * self.fsub_out):
                    # Compute the expected reconstructed resolution for sub-acquisition
                    numerator_fwhm += scalar_acquisition_operators[jsub] * weight_factor[jsub] * fwhm_in[jsub]
                    denominator_fwhm += scalar_acquisition_operators[jsub] * weight_factor[jsub]

                    # Compute the expected reconstructed frequencies for sub_acquisition
                    numerator_nus += scalar_acquisition_operators[jsub] * weight_factor[jsub]
                    denominator_nus += scalar_acquisition_operators[jsub]

                # Compute the expected resolution
                fwhm_rec = np.append(fwhm_rec, np.sum(numerator_fwhm) / np.sum(denominator_fwhm))

                # Compute the expected frequency
                fraction = np.sum(numerator_nus) / np.sum(denominator_nus)
                x0 = self.nus_Q[irec]
                corrected_nu = minimize(fun, x0)
                self.nus_Q[irec] = corrected_nu["x"]

        if self.rank == 0:
            print(f"FWHM for TOD generation : {fwhm_in}")
            print(f"FWHM for reconstruction : {fwhm_out}")
            print(f"Final FWHM : {fwhm_rec}")

        return fwhm_in, fwhm_out, fwhm_rec

    def get_input_map(self, m_nu):
        r"""Input maps.

        Function to get the input maps from PySM3.

        Returns
        -------
        maps_in: array_like
            Input maps :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.

        """

        m_nu_in = np.zeros((self.params["QUBIC"]["nrec"], 12 * self.params["SKY"]["nside"] ** 2, 3))

        for i in range(self.params["QUBIC"]["nrec"]):
            m_nu_in[i] = np.mean(m_nu[i * self.fsub_out: (i + 1) * self.fsub_out], axis=0)

        return m_nu_in

    def get_tod(self):
        r"""Simulated TOD.

        Method that compute observed TODs with :math:`\vec{TOD} = H \cdot \vec{s} + \vec{n}`, with H the QUBIC operator, :math:`\vec{s}` the sky signal and :math:`\vec{n}` the instrumental noise`.

        Returns
        -------
        TOD: array_like
            Simulated TOD :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.

        """

        TOD_QUBIC = self.ITF_QUBIC * self.H_in_qubic(self.maps_input.m_nu).ravel() + self.noiseq

        if not self.params["PLANCK"]["external_data"]:
            return TOD_QUBIC

        # To handle the case nrec == 1, even if it is broken because of the way compute_freq is used in QubicMultiAcquisitions
        TOD_PLANCK = np.zeros((max(self.params["QUBIC"]["nrec"], 2), 12 * self.params["SKY"]["nside"] ** 2, 3))

        for irec in range(self.params["QUBIC"]["nrec"]):
            fwhm_irec = np.min(self.fwhm_in[irec * self.fsub_in: (
                                                                         irec + 1) * self.fsub_in])  # self.fwhm_in = 0 if convolution_in == False
            C = HealpixConvolutionGaussianOperator(
                fwhm=fwhm_irec,
                lmax=3 * self.params["SKY"]["nside"] - 1,
            )
            if irec < self.params["QUBIC"]["nrec"] / 2:  # choose between the two levels of noise
                noise = self.noise_planck[0]
            else:
                noise = self.noise_planck[1]
            TOD_PLANCK[irec] = C(self.maps_input.maps[irec] + noise)

        if self.params["QUBIC"]["nrec"] == 1:  # To handle the case nrec == 1, TOD_PLANCK[0] alreay computed above
            TOD_PLANCK[1] = C(self.maps_input.maps[1] + self.noise_planck[1])

        TOD_PLANCK = TOD_PLANCK.ravel()

        TOD = np.r_[TOD_QUBIC, TOD_PLANCK]
        return TOD

    def get_preconditioner(self):
        """PCG Preconditioner.

        Returns
        -------
        M: DiagonalOperator
            Preconditioner for PCG algorithm.
        """
        if not self.params["PCG"]["preconditioner"]:
            return None

        itf_energy = 1 if not self.params["QUBIC"].get("deconvolution", False) \
                   else float(np.sum(self.bolometer_kernel ** 2))

        print(f'{itf_energy = }')

        nrec = self.params["QUBIC"]["nrec"]
        nside = self.params["SKY"]["nside"]
        nsub_out = self.params["QUBIC"]["nsub_out"]
        fsub = nsub_out // nrec
        npix = 12 * nside ** 2
        no_det = len(self.joint.qubic.multiinstrument[0].detector)
        H_qubic = self.joint.qubic.operator

        stacked_dptdp_inv = np.empty((nrec, npix))

        # Pre-fetch Planck diagonals if needed
        if self.params["PLANCK"]["external_data"] and self.params["PLANCK"]["level_noise_planck"] > 0:
            Diag_planck_143 = self.joint.pl143.get_invntt_operator(self.params["PLANCK"]["level_noise_planck"]).data[:,
            0]
            Diag_planck_217 = self.joint.pl217.get_invntt_operator(self.params["PLANCK"]["level_noise_planck"]).data[:,
            0]
            planck_diag_sum = Diag_planck_143 ** 2 + Diag_planck_217 ** 2

        for irec in range(nrec):
            stacked_dptdp_inv_fsub = np.empty((fsub, npix))

            for j_fsub in range(fsub):
                H_single = H_qubic[irec * fsub + j_fsub]

                D = H_single.operands[1]
                P = H_single.operands[-1]
                sh = P.matrix.data.index.shape

                point_per_det = sh[0] // no_det
                mapPtP_perdet_seq = np.empty((no_det, npix))

                for det in range(no_det):
                    start, end = det * point_per_det, (det + 1) * point_per_det
                    indices = P.matrix.data.index[start:end, :]
                    weights = P.matrix.data.r11[start:end, :]
                    flat_indices = indices.ravel()
                    flat_weights = weights.ravel()

                    mapPitPi = np.bincount(flat_indices, weights=flat_weights ** 2, minlength=npix)
                    mapPtP_perdet_seq[det, :] = mapPitPi

                D_sq = D.data ** 2
                mapPtP_seq_scaled = D_sq[:, np.newaxis] * mapPtP_perdet_seq
                dptdp = mapPtP_seq_scaled.sum(axis=0) * itf_energy

                if self.params["PLANCK"]["external_data"] and self.params["PLANCK"]["level_noise_planck"] > 0:
                    dptdp = dptdp + planck_diag_sum

                # Safe inversion
                dptdp_inv = np.zeros_like(dptdp)
                nonzero = dptdp != 0
                dptdp_inv[nonzero] = 1.0 / dptdp[nonzero]
                stacked_dptdp_inv_fsub[j_fsub] = dptdp_inv

            stacked_dptdp_inv[irec] = stacked_dptdp_inv_fsub.sum(axis=0)

        if self.params["PLANCK"]["external_data"]:
            preconditioner = BlockDiagonalOperator(
                [DiagonalOperator(ci[self.seenpix], broadcast="rightward") for ci in stacked_dptdp_inv], new_axisin=0)
        else:
            preconditioner = BlockDiagonalOperator(
                [DiagonalOperator(ci, broadcast="rightward") for ci in stacked_dptdp_inv], new_axisin=0)

        return preconditioner

    def call_pcg(self, d, x0, seenpix):
        r"""Preconditioned Conjugate Gradiant algorithm.

        Solve the map-making equation iteratively : :math:`(H^T . N^{-1} . H) . x = H^T . N^{-1} . d`.

        The PCG used for the minimization is intrinsequely parallelized (e.g see PyOperators).

        Parameters
        ----------
        d : array_like
            Array containing the TODs generated previously :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.
        x0 : array_like
            Starting point of the PCG algorithm :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.
        seenpix : array_like
            Boolean array to define the pixels seen by QUBIC :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.

        Returns
        -------
        solution: array_like
            Reconstructed maps :math:`(N_{rec}, 12 \times N^{2}_{side}, N_{stk})`.

        """

        ### Update components when pixels outside the patch are fixed (assumed to be 0)
        print("H_out", self.H_out.shapein, self.H_out.shapeout)
        print("invN", self.invN.shapein, self.invN.shapeout)
        A = self.H_out.T * self.ITF.T * self.invN * self.ITF * self.H_out

        if self.params["PLANCK"]["external_data"]:
            x_planck = self.m_nu_in * (1 - seenpix[None, :, None])
            b = self.H_out.T * self.ITF.T * self.invN * (d - self.H_out_all_pix(x_planck))
        else:
            b = self.H_out.T * self.ITF.T * self.invN * d

        ### Preconditionning
        M = self.get_preconditioner()

        if self.params["PCG"]["gif"]:
            gif_folder = self.plot_folder + f"{self.job_id}/iter/"
        else:
            gif_folder = None

        true_maps = self.m_nu_in.copy()
        for irec in range(self.params["QUBIC"]["nrec"]):
            C = HealpixConvolutionGaussianOperator(fwhm=self.fwhm_rec[irec], lmax=3 * self.params["SKY"]["nside"] - 1)
            true_maps[irec] = C(self.m_nu_in[irec])

        ### PCG
        solution_qubic_planck = pcg(
            A=A,
            b=b,
            comm=self.comm,
            x0=x0,
            M=M,
            tol=self.params["PCG"]["tol_pcg"],
            disp=True,
            maxiter=self.params["PCG"]["n_iter_pcg"],
            gif_folder=gif_folder,
            job_id=self.job_id,
            seenpix=seenpix,
            seenpix_plot=seenpix,
            center=self.center,
            reso=self.params["PCG"]["resolution_plot"],
            fwhm_plot=self.params["PCG"]["fwhm_plot"],
            input=true_maps,
            is_planck=self.params["PLANCK"]["external_data"],
        )

        self.convergence_pcg = solution_qubic_planck["x"]["convergence"]

        if self.params["PCG"]["gif"]:
            do_gif(gif_folder, "iter_", output="animation.gif")

        self.mpi._barrier()

        if self.params["QUBIC"]["nrec"] == 1:
            solution_qubic_planck["x"]["x"] = np.array([solution_qubic_planck["x"]["x"]])

        solution = np.ones(self.m_nu_in.shape) * hp.UNSEEN
        if self.params["PLANCK"]["external_data"]:
            solution[:, seenpix, :] = solution_qubic_planck["x"]["x"].copy()
        else:
            solution[:, seenpix, :] = solution_qubic_planck["x"]["x"][:, seenpix, :].copy()

        return solution

    def _save_data(self, name, d):
        """
        Method to save data using pickle convention.

        """

        with open(name, "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        """Run the FMM Pipeline.

        Method to run the whole pipeline from TOD generation from sky reconstruction by reading `params.yml` file.

        """

        self.mpi._print_message("\n=========== Map-Making ===========\n")

        ### Get simulated data
        self.TOD = self.get_tod()

        ### Wait for all processes
        self.mpi._barrier()

        if self.params["PLANCK"]["external_data"]:
            starting_point = np.zeros(self.m_nu_in[:, self.seenpix, :].shape)
            if self.params["PCG"]["initial_guess_intensity_to_zero"] is False:
                starting_point[..., 0] = self.m_nu_in[:, self.seenpix, 0].copy()
        else:
            starting_point = np.zeros(self.m_nu_in.shape)
            if self.params["PCG"]["initial_guess_intensity_to_zero"] is False:
                starting_point[..., 0] = self.m_nu_in[..., 0].copy()

        ### Solve map-making equation
        self.s_hat = self.call_pcg(self.TOD, x0=starting_point, seenpix=self.seenpix)

        ### Wait for all processes
        self.mpi._barrier()

        ### n = m_signalnoisy - m_signal
        self.s_hat_noise = self.s_hat - self.m_nu_in

        ### Ensure that non seen pixels is 0 for spectrum computation
        self.s_hat[:, ~self.seenpix, :] = 0
        self.s_hat_noise[:, ~self.seenpix, :] = 0

        ### Plots and saving
        if self.rank == 0:
            self.external_maps = self.externaldata.maps.copy()
            self.external_maps[:, ~self.seenpix, :] = 0

            self.external_maps_noise = self.externaldata.maps_noise.copy()
            self.external_maps_noise[:, ~self.seenpix, :] = 0

            self.nus_rec = self.nus_Q.copy()
            if len(self.externaldata.experiments["Planck"]["frequency"]) != 0:
                fwhm_ext = self.externaldata.fwhm_ext.copy()
                self.s_hat = np.concatenate((self.s_hat, self.external_maps), axis=0)
                self.s_hat_noise = np.concatenate((self.s_hat_noise, self.external_maps_noise), axis=0)
                self.nus_rec = np.array(list(self.nus_Q) + list(self.externaldata.experiments["Planck"]["frequency"]))
                self.fwhm_rec = np.array(list(self.fwhm_rec) + list(fwhm_ext))
            self.plots.plot_frequency_maps(
                self.m_nu_in[: len(self.nus_Q)],
                self.s_hat[: len(self.nus_Q)],
                self.center,
                reso=15,
                nsig=3,
                filename=self.plot_folder + "/all_maps.png",
                figsize=(10, 5),
            )

            mapmaking_time = time.time() - self.mapmaking_time_0
            if self.comm is None:
                print(f"Map-making done in {mapmaking_time:.3f} s")
            else:
                if self.rank == 0:
                    print(f"Map-making done in {mapmaking_time:.3f} s")

            dict_solution = {
                "maps_in": self.m_nu_in,
                "maps": self.s_hat,
                "maps_noise": self.s_hat_noise,
                "tod": self.TOD,
                "nus": self.nus_rec,
                "coverage": self.coverage,
                "convergence": self.convergence_pcg,
                "center": self.center,
                "parameters": self.params,
                "fwhm_in": self.fwhm_in,
                "fwhm_out": self.fwhm_out,
                "fwhm_rec": self.fwhm_rec,
                "seenpix": self.seenpix,
                "duration": mapmaking_time,
                "qubic_dict": {k: v for k, v in self.dict_out.items() if k != "comm"},
                # I have to remove the MPI communicator, which is not supported by pickle
            }

            self._save_data(self.file, dict_solution)

        ### Wait for all processors
        self.mpi._barrier()


class PipelineEnd2End:
    """FMM Pipeline.

    Wrapper for End-2-End pipeline. It added class one after the others by running method.run().

    """

    def __init__(self, comm, parameters_path):
        with open(parameters_path, "r") as tf:
            self.params = yaml.safe_load(tf)

        self.comm = comm
        self.job_id = os.environ.get("SLURM_JOB_ID")

        self.folder = "FMM/" + self.params["path_out"] + "maps/"
        self.file = self.folder + self.params["datafilename"] + f"_{self.job_id}.pkl"
        self.file_spectrum = "FMM/" + self.params["path_out"] + "spectrum/" + "spectrum_" + self.params[
            "datafilename"] + f"_{self.job_id}.pkl"
        self.mapmaking = None

    def main(self, specific_file=None):
        ### Execute Frequency Map-Making
        if self.params["Pipeline"]["mapmaking"]:
            ### Checking errors
            ErrorChecking(self.params).check_errors()
            ### Initialization
            self.mapmaking = PipelineFrequencyMapMaking(self.comm, self.file, self.params)

            ### Run
            self.mapmaking.run()

            ### Get synthesized beam
            for prefix, obj in zip(["after_dict_in", "after_dict_out"],
                                   [self.mapmaking.joint_tod, self.mapmaking.joint]):
                scene = obj.scene
                qubic_multi = obj.qubic.multiinstrument
                synth_beam = qubic_multi[0].get_synthbeam(scene, idet=154)  # this is the QubicSoft number for TES #33

                rot_custom = hp.Rotator(rot=[180, 90 - 50], inv=True)
                sbrot = rot_custom.rotate_map_alms(synth_beam)
                hp.gnomview(10 * np.log10(sbrot / sbrot.max()),
                            rot=[0, 50],
                            reso=30,
                            xsize=200,
                            ysize=200,
                            sub=(1, 1, 1),
                            title=f'{prefix} - Theoretical SB',
                            unit='dB',
                            #min=-40,
                            max=0,
                            margins=(0.05, 0.05, 0.05, 0.05)
                            )
                plt.savefig(self.folder + f"{prefix}_sb_theoretical.pdf", dpi=1000)
                plt.close()

        ### Execute spectrum
        if self.params["Pipeline"]["spectrum"]:
            if self.params["Spectrum"]["lmax"] > 2 * self.params["SKY"]["nside"] - 1:
                raise ValueError("lmax should be lower than 2*nside - 1")

            if self.comm.Get_rank() == 0:
                create_folder_if_not_exists(self.comm, "FMM/" + self.params["path_out"] + "spectrum/")

                if self.mapmaking is not None:
                    self.spectrum = Spectra(self.file)
                else:
                    self.spectrum = Spectra(specific_file)

                ### Signal
                DlBB_maps = self.spectrum.run(maps=self.spectrum.maps)

                ### Noise
                DlBB_noise = self.spectrum.run(maps=self.spectrum.dictionary["maps_noise"])

                dict_solution = {
                    "nus": self.spectrum.dictionary["nus"],
                    "ell": self.spectrum.ell,
                    "Dls": DlBB_maps,
                    "Nl": DlBB_noise,
                    "parameters": self.params,
                }

                self.mapmaking._save_data(self.file_spectrum, dict_solution)


if __name__ == "__main__":
    pipeline_path = "/home/sofiafera/thesis/qubic/qubic/scripts/MapMaking/src/FMM/params.yml"

    comm = MPI.COMM_WORLD
    pipeline_end2end = PipelineEnd2End(comm, pipeline_path)
    pipeline_end2end.main()
