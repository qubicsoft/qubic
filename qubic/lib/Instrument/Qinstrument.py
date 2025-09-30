# coding: utf-8
import copy

import healpy as hp
import numexpr as ne
import numpy as np

from scipy.constants import c, h, k, sigma
from scipy.integrate import quad

from pyoperators import (
    Cartesian2SphericalOperator,
    DenseBlockDiagonalOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
    IntegrationTrapezeOperator,
    ReshapeOperator,
    Rotation2dOperator,
    Rotation3dOperator,
    Spherical2CartesianOperator,
)
from pyoperators.utils import operation_assignment, pool_threading, product, split
from pyoperators.utils.ufuncs import abs2
from pysimulators import (
    ConvolutionTruncatedExponentialOperator,
    Instrument,
    Layout,
    ProjectionOperator,
)
from pysimulators.geometry import surface_simple_polygon
from pysimulators.interfaces.healpy import (
    Cartesian2HealpixOperator,
    HealpixConvolutionGaussianOperator,
)
from pysimulators.sparse import FSRMatrix, FSRRotation2dMatrix, FSRRotation3dMatrix
from qubic import _flib as flib

from ..Qbeams import BeamFitted, BeamGaussian, MultiFreqBeam
from ..Calibration.Qcalibration import QubicCalibration
from ..Qripples import BeamGaussianRippled, ConvolutionRippledGaussianOperator
from ..Qutilities import _compress_mask

from ..Qbilin_interp import Cartesian2HealpixOperator_bilin_interp


__all__ = ["QubicInstrument", "QubicMultibandInstrumentTrapezoidalIntegration","QubicMultibandInstrument"]

def compute_freq(band, Nfreq=None, relative_bandwidth=0.25):
    """
    Prepare frequency bands parameters
    band -- int,
        QUBIC frequency band, in GHz.
        Typical values: 150, 220
    relative_bandwidth -- float, optional
        Ratio of the difference between the edges of the
        frequency band over the average frequency of the band:
        2 * (nu_max - nu_min) / (nu_max + nu_min)
        Typical value: 0.25
    Nfreq -- int, optional
        Number of frequencies within the wide band.
        If not specified, then Nfreq = 15 if band == 150
        and Nfreq = 20 if band = 220
    """

    if Nfreq is None:
        Nfreq = {150: 15, 220: 20}[band]

    nu_min = band * (1 - relative_bandwidth / 2)
    nu_max = band * (1 + relative_bandwidth / 2)

    Nfreq_edges = Nfreq + 1
    base = (nu_max / nu_min) ** (1. / Nfreq)

    nus_edge = nu_min * np.logspace(0, Nfreq, Nfreq_edges, endpoint=True, base=base)
    nus = np.array([(nus_edge[i] + nus_edge[i - 1]) / 2 for i in range(1, Nfreq_edges)])
    deltas = np.array([(nus_edge[i] - nus_edge[i - 1]) for i in range(1, Nfreq_edges)])
    Delta = nu_max - nu_min
    Nbbands = len(nus)
    return Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands

class Filter(object):
    def __init__(self, nu, relative_bandwidth):
        self.nu = float(nu)
        self.relative_bandwidth = float(relative_bandwidth)
        self.bandwidth = self.nu * self.relative_bandwidth


class Optics(object):
    pass


class SyntheticBeam(object):
    pass

class Noise(object):
    pass

def funct(x, p, n):
    return x**p / (np.exp(x) - 1) ** n


class QubicInstrument(Instrument):
    """
    The QubicInstrument class. It represents the instrument setup.

    """

    def __init__(self, d, FRBW=None):
        """
        d : Input dictionary, from which the following Parameters are read
        FRBW: float, optional
            keeps the Full Relative Band Width when building subinstruments
            Needed to compute the photon noise        
        Parameters
        ----------
        calibration : QubicCalibration
            The calibration tree.
        detector_fknee : array-like, optional
            The detector 1/f knee frequency in Hertz.
        detector_fslope : array-like, optional
            The detector 1/f slope index.
        detector_ncorr : int, optional
            The detector 1/f correlation length.
        detector_ngrids : int, optional
            Number of detector grids.
        detector_nep : array-like, optional
            The detector NEP [W/sqrt(Hz)].
        detector_tau : array-like, optional
            The detector time constants in seconds.
        filter_nu : float, optional
            The filter central wavelength, in Hz.
        filter_relative_bandwidth : float, optional
            The filter relative bandwidth Δν/ν.
        polarizer : boolean, optional
            If true, the polarizer grid is present in the optics setup.
        primary_beam : function f(theta [rad], phi [rad]), optional
            The primary beam transmission function.
        secondary_beam : function f(theta [rad], phi [rad]), optional
            The secondary beam transmission function.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        synthbeam_kmax : integer, optional
            The diffraction order above which the peaks are ignored.
            For instance, a value of kmax=2 will model the synthetic beam by
            (2 * kmax + 1)**2 = 25 peaks and a value of kmax=0 will only sample
            the central peak.
        synthbeam_fraction: float, optional
            The fraction of significant peaks retained for the computation
            of the synthetic beam.
        beam_shape: dictionary entry, string
            the shape of the primary and secondary beams:
            "gaussian", "fitted_beam" or "multi_freq"

        """
        self.d = d.copy()
        self.debug = d["debug"]  # if True allows debuging prints
        filter_nu = d["filter_nu"]
        filter_relative_bandwidth = d["filter_relative_bandwidth"]
        if FRBW is not None:
            self.FRBW = FRBW
        else:
            self.FRBW = filter_relative_bandwidth
        if self.debug:
            print("filter_nu = ",filter_nu,"FRBW = ", self.FRBW, "dnu = ", filter_relative_bandwidth)

        ## Choose the relevant Optics calibration file
        epsilon = 1.0e-3 # one mHz of margin for the comparisons to compensate for python roundoff error
        self.nu1 = 150e9
        self.nu1_up = self.nu1 * (1 + self.FRBW / 2) + epsilon
        self.nu1_down = self.nu1 * (1 - self.FRBW / 2) - epsilon
        self.nu2 = 220e9
        self.nu2_up = self.nu2 * (1 + self.FRBW / 2) + epsilon
        self.nu2_down = self.nu2 * (1 - self.FRBW / 2) - epsilon
        if (filter_nu <= self.nu1_up) and (filter_nu >= self.nu1_down):
            d["optics"] = d["optics"].replace(d["optics"][-7:-4], "150")
        elif (filter_nu <= self.nu2_up) and (filter_nu >= self.nu2_down):
            d["optics"] = d["optics"].replace(d["optics"][-7:-4], "220")
            if d["config"] == "TD":
                raise ValueError("TD Not used at frequency " +
                                 str(int(d["filter_nu"] / 1e9)) + " GHz")
        else:
            raise ValueError("frequency = " + str(int(d["filter_nu"] / 1e9)) + " out of bounds: %.1f - %.1f")

        d["optics"] = d["optics"].replace(d["optics"][-10:-8], d["config"])
        d["detarray"] = d["detarray"].replace(d["detarray"][-7:-5], d["config"])
        d["hornarray"] = d["hornarray"].replace(d["hornarray"][-7:-5], d["config"])

        if d["nf_sub"] is None and d["MultiBand"] is True:
            raise ValueError("Error: number of subband not specified")

        detector_fknee = d["detector_fknee"]
        detector_fslope = d["detector_fslope"]
        detector_ncorr = d["detector_ncorr"]
        detector_nep = d["detector_nep"]
        detector_ngrids = d["detector_ngrids"]
        detector_tau = d["detector_tau"]

        polarizer = d["polarizer"]
        synthbeam_dtype = np.float32
        synthbeam_fraction = d["synthbeam_fraction"]
        synthbeam_kmax = d["synthbeam_kmax"]
        synthbeam_peak150_fwhm = np.radians(d["synthbeam_peak150_fwhm"])
        ripples = d["ripples"]
        nripples = d["nripples"]

        #check if synthetic beam peaks are from file or calculated in peak_angles
        use_file=d.get(("use_synthbeam_fits_file"))
        self.use_file=bool(use_file)
        sbeam=bool(d.get("synthbeam"))
        
        if sbeam == False:
            #Put in a dummy synthetic beam
            self.sbeam_fits = "CalQubic_Synthbeam_Calibrated_Multifreq_FI.fits"
            d["synthbeam"] = "CalQubic_Synthbeam_Calibrated_Multifreq_FI.fits"
            print("There is no fits file given in this dictionary. Using analytical model of beam parameters")
            self.use_file=False
        else:
            self.sbeam_fits = d["synthbeam"]


        # Choose the primary beam calibration file
        if d["beam_shape"] == "gaussian":
            d["primbeam"] = d["primbeam"].replace(d["primbeam"][-6], "2")
            primary_shape = "gaussian"
            secondary_shape = "gaussian"
        elif d["beam_shape"] == "fitted_beam":
            d["primbeam"] = d["primbeam"].replace(d["primbeam"][-6], "3")
            primary_shape = "fitted_beam"
            secondary_shape = "fitted_beam"
        else:
            d["primbeam"] = d["primbeam"].replace(d["primbeam"][-6], "4")
            primary_shape = "multi_freq"
            secondary_shape = "multi_freq"
        if self.debug:
            print("primary_shape", primary_shape)
            print("d['primbeam']", d["primbeam"])
        self.config = d["config"]
        calibration = QubicCalibration(d)
        self.calibration = calibration

        self.ripples = ripples
        self.nripples = nripples
        self._init_beams(primary_shape, secondary_shape, filter_nu)
        self._init_filter(filter_nu, filter_relative_bandwidth)
        self._init_horns(filter_nu)
        self._init_optics(polarizer, d)
        self._init_synthbeam(synthbeam_dtype, synthbeam_peak150_fwhm)
        self.synthbeam.fraction = synthbeam_fraction
        self.synthbeam.kmax = synthbeam_kmax
        if self.use_file:
            self.synthbeam_file(d)

        layout = self._get_detector_layout(
            detector_ngrids,
            detector_nep,
            detector_fknee,
            detector_fslope,
            detector_ncorr,
            detector_tau
        )
        Instrument.__init__(self, layout)

    def _get_detector_layout(self, ngrids, nep, fknee, fslope, ncorr, tau):
        shape, vertex, removed, ordering, quadrant, efficiency = \
            self.calibration.get("detarray")
        if ngrids == 2:
            shape = (2,) + shape
            vertex = np.array([vertex, vertex])
            removed = np.array([removed, removed])
            ordering = np.array([ordering, ordering + np.max(ordering) + 1], ordering.dtype)
            quadrant = np.array([quadrant, quadrant + 4], quadrant.dtype)
            efficiency = np.array([efficiency, efficiency])
        vertex = np.concatenate([vertex, np.full_like(vertex[..., :1],
                                                      -self.optics.focal_length)], -1)

        def theta(self):
            return np.arctan2(
                np.sqrt(np.sum(self.center[..., :2] ** 2, axis=-1)),
                self.center[..., 2])

        def phi(self):
            return np.arctan2(self.center[..., 1], self.center[..., 0])

        layout = Layout(
            shape, vertex=vertex, selection=~removed, ordering=ordering,
            quadrant=quadrant, nep=nep, fknee=fknee, fslope=fslope,
            tau=tau, theta=theta, phi=phi, efficiency=efficiency)

        # assume all detectors have the same area
        layout.area = surface_simple_polygon(layout.vertex[0, :, :2])
        layout.ncorr = ncorr
        layout.ngrids = ngrids
        modif = 3 # 3 is good, 0 is no modif
        if modif == 1:
            print("correction = {}".format([-6, -16, 0]))
            layout.center += np.array([0.003*(-6), 0.003*(-16), 0.003*0]) # fit unflipped numerotation? no
        # rotation_center = Rotation3dOperator("XYZ", 0, 0, 5, degrees=True, shapein=np.shape(layout.center))
        # layout.center = rotation_center(layout.center)
        elif modif == 2:
            print("correction = {}".format([-1, 8, 0]))
            layout.center += np.array([0.003*(-1), 0.003*(+8), 0.003*0]) # fit flipped numerotation? no
        elif modif == 3:
            correction = [1, -8]
            print("correction = {}".format(correction))
            layout.center += np.array([0.003*correction[0], 0.003*correction[1], 0.003*0]) # fit flipped numerotation? yes
        print(layout.center)
        # aze
        return layout

    def _init_beams(self, primary, secondary, filter_nu):
        # The beam shape is taken into account
        nu = filter_nu / 1e9   ### NB: this has been corrected on Nov 17th by JCH before nu was cast into an integer for a mysterious reason
        if primary == "gaussian":
            PrimBeam = BeamGaussian(
                np.radians(self.calibration.get("primbeam")), nu=nu)
        elif primary == "fitted_beam":
            par, omega = self.calibration.get("primbeam")
            PrimBeam = BeamFitted(par, omega, nu=nu)
        elif primary == "multi_freq":
            parth, parfr, parbeam, alpha, xspl = self.calibration.get("primbeam")
            PrimBeam = MultiFreqBeam(parth, parfr, parbeam, alpha, xspl,
                                     nu=nu)
        self.primary_beam = PrimBeam
        if secondary == "gaussian":
            SecBeam = BeamGaussian(
                np.radians(self.calibration.get("primbeam")), nu=nu,
                backward=True)
        elif secondary == "fitted_beam":
            par, omega = self.calibration.get("primbeam")
            SecBeam = BeamFitted(par, omega, nu=nu, backward=True)
        elif secondary == "multi_freq":
            parth, parfr, parbeam, alpha, xspl = self.calibration.get("primbeam")
            SecBeam = MultiFreqBeam(parth, parfr, parbeam, alpha, xspl, nu=nu,
                                    backward=True)
        self.secondary_beam = SecBeam

    def _init_filter(self, nu, relative_bandwidth):
        self.filter = Filter(nu, relative_bandwidth)

    def _init_horns(self, filter_nu):
        self.horn = self.calibration.get("hornarray")
        self.horn.radeff = self.horn.radius
        # In the 150 GHz band, horns are one moded 
        if (filter_nu <= self.nu1_up) and (filter_nu >= self.nu1_down):
            kappa = (
                np.pi
                * self.horn.radius**2
                * self.primary_beam.solid_angle
                * filter_nu**2
                / c**2
            )
            self.horn.radeff = self.horn.radius / np.sqrt(kappa)

    def _init_optics(self, polarizer, d):
        optics = Optics()
        calib = self.calibration.get("optics")
        optics.components = calib["components"]
        optics.focal_length = d["focal_length"]
        optics.polarizer = bool(polarizer)
        self.optics = optics

    def synthbeam_file(self, d):
        #read in beam peak locations from a fits file
        thetafits,phifits,valfits,freqs, mheader = self.calibration.get("synthbeam")
        self.thetafits = thetafits
        self.phifits = phifits
        self.valfits = valfits
        self.freqs = freqs
        self.numpeaks=np.zeros(10)

    def _init_synthbeam(self, dtype, synthbeam_peak150_fwhm):
        sb = SyntheticBeam()
        sb.dtype = np.dtype(dtype)
        if not self.ripples:
            sb.peak150 = BeamGaussian(synthbeam_peak150_fwhm)
        else:
            sb.peak150 = BeamGaussianRippled(synthbeam_peak150_fwhm,
                                             nripples=self.nripples)
        self.synthbeam = sb

    def __str__(self):
        state = [("ngrids", self.detector.ngrids),
                 ("selection", _compress_mask(~self.detector.all.removed)),
                 ("synthbeam_fraction", self.synthbeam.fraction),
                 ("synthbeam_peak150_fwhm_deg",
                  np.degrees(self.synthbeam.peak150.fwhm)),
                 ("synthbeam_kmax", self.synthbeam.kmax)]
        return "Instrument:\n" + \
               "\n".join(["    " + a + ": " + repr(v) for a, v in state]) + \
               "\n\nCalibration:\n" + "\n". \
                   join("    " + l for l in str(self.calibration).splitlines())

    __repr__ = __str__

    def get_noise(self, sampling, scene, det_noise=True, photon_noise=True, out=None,
                  operation=operation_assignment):
        """
        Return a noisy timeline.

        """
        if out is None:
            out = np.empty((len(self), len(sampling)))
        self.get_noise_detector(sampling, out=out)
        if det_noise is False:
            out *= 0
        if photon_noise:
            out += self.get_noise_photon(sampling, scene)
        return out

    def get_noise_detector(self, sampling, out=None):
        """
        Return the detector noise (#det, #sampling).

        """
        return Instrument.get_noise(
            self, sampling, nep=self.detector.nep, fknee=self.detector.fknee,
            fslope=self.detector.fslope, out=out)

    def get_noise_photon(self, sampling, scene, out=None):
        """
        Return the photon noise (#det, #sampling).

        """
        nep_photon = self._get_noise_photon_nep(scene)
        return Instrument.get_noise(self, sampling, nep=nep_photon, out=out)

    def _get_noise_photon_nep(self, scene):
        
        """
        This method computes the NEP photon noise. 
        It works as follow:
            1st) Load the noise attributes in an object called noise using load_NEP_parameters
                Some of them are:   . photon power, NEP (empty arrays to be filled)  (P_phot, NEP_phot2)
                                    . indexes for each component (bsb, combiner, ndf, etc) (ib2b, icomb,indf, etc)
                                    . the cumulative product of the transmissions (tr_prod)
                                    . solid angle as seen by a detector (omega_det)
                                    . physical horn area (S_horns)
                                    . effective horn area (S_horns_eff)
                                    among others...
                you can see them by doing:
                # qinstrument = qubic.QubicMultibandInstrument(dict)
                # scene = qubic.QubicScene(dict)
                noisepar = qinstrument[0].load_NEP_parameters(scene)

            2nd) It computes the NEP in each bolometer for each component of the noise.
            The components are: "CMB","atm","winb1","block1",
                                "block2","block3","block4","block5",
                                "block6","12cmed","hwp","polgr","ba2ba",
                                "combin","cslpe","ndf","7cmlpe","6.2cmlpe",
                                "5.6cmlpe"

        Finally it computes the total NEP as sqrt{sum_i NEP_phot2_i + NEP_env2}

        ================================
        Return the photon noise NEP (#det,).

        """

        noise = self.load_NEP_parameters(scene)

        # Compute noise of the components before the horn array
        self.NEP_before_horns(noise, noise.nu)

        # bifurcation for the whole 150 GHz
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            #noise.nu_up = 168e9

            #Compute NEP contribution from horn array
            self.NEP_horns(noise)

            ## Environment NEP
            self.NEP_environment(noise, noise.names)

            # Combiner - compute
            self.NEP_combiner(noise)

            # cold stop low pass edge - compute
            self.NEP_coldstop(noise)

            # Dichroic
            if self.config == "FI":
                #Compute
                self.NEP_dichroic(noise)

            if noise.emissivities[noise.indf] == 0.0:
                noise.P_phot[noise.indf] = 0.0
                noise.NEP_phot2[noise.indf] = 0.0
            else:
                #Compute NEP neutral density filter

                self.NEP_neutraldensityfilter(noise)
            
            # The two before last low pass Edges - compute
            self.NEP_lowpassedge(noise, noise.lpe1)

            # Compute
            self.NEP_lowpassedge(noise, noise.lpe2)

        else:  # 220 GHz
            self.NEP_horns(noise)

            # Environment NEP
            self.NEP_environment(noise, noise.names)

            # combiner
            self.NEP_combiner(noise)

            # coldstop
            self.NEP_coldstop(noise)

            # dichroic
            self.NEP_dichroic(noise)

            # Last three filters (ndf, lpe1, lpe2?)
            self.NEP_lpefilter_220(noise, noise.indf)

            self.NEP_lpefilter_220(noise, noise.lpe1)

            self.NEP_lpefilter_220(noise, noise.lpe2)

            #self.NEP_lastfilters_220(noise) #Old
        # 5.6 cm EDGE (150 GHz) or Band Defining Filter (220 GHZ)
        
        self.NEP_lastfilter(noise)
        
        # Total NEP
        noise.P_phot_tot = np.sum(noise.P_phot, axis=0)
        noise.NEP_tot = np.sqrt(np.sum(noise.NEP_phot2, axis=0) + noise.NEP_phot2_env)
        
        if self.debug:
            print("Total photon power =  {0:.2e} W".format(noise.P_phot_tot.max()) +
                  ", Total photon NEP = " + "{0:.2e}".format(noise.NEP_tot.max()) + " W/sqrt(Hz)")
        
        return noise.NEP_tot

    def load_NEP_parameters(self, scene):
        
        """
        This method loads the parameters for the photon noise (NEP) computation.
        The attributes are loaded into a Noise() class. The attributes are:

        temperatures, transmissions, emissivities, gp (polarization) of each component of the instrument,
        names: name of each component,
        tr_prod: cumulative multiplication of the transmissions of each components,
        dnu: bandwidth,
        S_det: detector area,
        omega_det: solid angle sustained by a given detector on the sky,
        S_horns: physical horn area,
        S_horns_eff: effective horn area,
        sec_beam: secondary beam QubicInstrument.secondary_beam,
        
        P_phot, NEP_phot: empty arrays to be loaded by NEP-like methods (below),
        indexes for each component,

        ========
        Return:
            noise: object with attributes loaded
        """
        T_atm = scene.atmosphere.temperature
        tr_atm = scene.atmosphere.transmission
        em_atm = scene.atmosphere.emissivity
        T_cmb = scene.temperature
        # to avoid atmospheric emission in room testing
        if T_cmb > 100:
            em_atm = 0.
        cc = self.optics.components

        # Create an object to load and then move the attributes
        # for photon noise computation 
        noise = Noise()

        # adding sky components to the photon power and noise sources
        noise.temperatures = np.r_[T_cmb, T_atm, cc["temperature"]]
        noise.transmissions = np.r_[1, tr_atm, cc["transmission"]]
        noise.emissivities = np.r_[1, em_atm, cc["emissivity"]]
        noise.gp = np.r_[1, 1, cc["nstates_pol"]] # polarization of each component

        n = len(noise.temperatures) # MARTIN: number of optical elements + cmb + atm
        ndet = len(self.detector) # MARTIN: number of detectors
        # tr_prod : product of transmissions of all components lying
        # after the present one        
        noise.tr_prod = np.r_[[np.prod(noise.transmissions[j + 1:]) for j in range(n - 1)], 1]
        # insures that the noise is comuted for the full bandwidth.
        if (self.filter.nu <= self.nu1_up) \
                and (self.filter.nu >= self.nu1_down):
            noise.nu = self.nu1
        elif (self.filter.nu <= self.nu2_up) \
                and (self.filter.nu >= self.nu2_down):
            noise.nu = self.nu2
        noise.dnu = noise.nu * self.FRBW
        noise.S_det = self.detector.area
        noise.omega_det = -self.detector.area / \
                    self.optics.focal_length ** 2 * \
                    np.cos(self.detector.theta) ** 3
        # Physical horn area   
        noise.S_horns = np.pi * self.horn.radius ** 2 * len(self.horn)
        # Effective horn area, taking the number of modes into account 
        noise.S_horns_eff = np.pi * self.horn.radeff ** 2 * len(self.horn)
        noise.sec_beam = self.secondary_beam(self.detector.theta,
                                       self.detector.phi)
        alpha = np.arctan(0.5)  # half oppening angle of the combiner
        noise.omega_comb = np.pi * (1 - np.cos(alpha) ** 2)  # to be revisited,
        # depends on the detector position
        noise.omega_dichro = noise.omega_comb  # must be improved
        noise.omega_coldstop = 0.09  # average, depends slightly on
        # the detector position

        noise.P_phot = np.zeros((n, ndet))
        noise.NEP_phot2_nobunch = np.zeros_like(noise.P_phot)
        noise.NEP_phot2 = np.zeros_like(noise.P_phot)
        noise.g = np.zeros_like(noise.P_phot)
        names = ["CMB", "atm"]
        for i in range(len(cc)):
            names.append(cc[i][0])

        # Upper frequency for 150GHz channel computation of 
        noise.nu_up = 168e9
        # Load indexes  
        # components before the horn plane
        noise.ib2b = names.index(b"ba2ba")
        # Combiner 
        # the combiner is the component just after the horns
        noise.icomb = noise.ib2b + 1  
        # Cold stop
        noise.ics = noise.icomb + 1
        # Dichroic
        if self.config == "FI":
            noise.idic = noise.ics + 1
            #Neutral Density Filter
            noise.indf = noise.idic + 1
        else:
            noise.indf = noise.ics + 1
        #Low pass edges
        noise.lpe1 = noise.indf + 1
        noise.lpe2 = noise.lpe1 + 1
        noise.ilast = noise.lpe2 + 1

        noise.names = names
        
        if self.debug:
            print(self.config, ", central frequency:", int(noise.nu / 1e9), "+-",
                  int(noise.dnu / 2e9), "GHz, subband:", int(self.filter.nu / 1e9),
                  "GHz, n_modes =", np.pi * self.horn.radeff ** 2 * \
                  self.primary_beam.solid_angle * \
                  self.filter.nu ** 2 / c ** 2)
            indf = names.index(b"ndf") - 2
            if cc[noise.indf][2] != 1.0:
                print("Neutral density filter present, trans = ",
                      cc[indf][2])
            else:
                print("No neutral density filter")
        
        return noise

    def _raise_sampling_error(self, return_only, sampling):
        """
        Raise an error in case you want to extract only one component of the photon noise (return_only=True)
        but you did not give the sampling.
        """
        if return_only:
            if sampling == None:
                raise ValueError("If you want only a component of the photon noise, I need a qubic sampling to map it (qubic.get_sampling(dictionary)) ")
        else:
            return

    def _raise_debug(self, noisepar, indx,
                    before_b2b = False,
                    environment = False):
        """
        Print information for each component of the noise to easily debug.

        Arguments:
            noisepar: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            indx: 
                noise.[index] where "index" correspond to the index of the component
            before_b2b: 
                if True, it return the data for all the components before back-to-back horns
            environment:
                if True, return the NEP environment data    
        ===========
        Return: 
            print statements.
        """

        if before_b2b:
            for j in range(noisepar.ib2b):
                print(noisepar.names[j], ", T=", noisepar.temperatures[j],
                      "K, P = {0:.2e} W".format(noisepar.P_phot[j].max()),
                      ", NEP = {0:.2e}".format(np.sqrt(noisepar.NEP_phot2[j]).max()) + "  W/sqrt(Hz)")
        else:
            if not environment:
                print(noisepar.names[indx], ", T=", noisepar.temperatures[indx],
                      "K, P = {0:.2e} W".format(noisepar.P_phot[indx].max()),
                      ", NEP = {0:.2e}".format(np.sqrt(noisepar.NEP_phot2[indx]).max()) + "  W/sqrt(Hz)")
            else:
                # Temperature is the same as the one for back-to-back horns
                print("Environment T =", noisepar.temperatures[indx],
                      "K, P = {0:.2e} W".format(noisepar.P_phot_env.max()),
                      ", NEP = {0:.2e}".format(np.sqrt(noisepar.NEP_phot2_env).max()) + "  W/sqrt(Hz)")
        return



    def NEP_before_horns(self, noise, nu, 
                        return_only = False, sampling = None):
        """
        This method computes the noise for all the components before 
        back-to-back array.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            nu: frequency
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot_nobunch"
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)

        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        ib2b = noise.ib2b
        noise.g[:ib2b] = noise.gp[:ib2b, None] * noise.S_horns_eff * noise.omega_det * (nu / c) ** 2 \
                   * noise.sec_beam * noise.dnu
        noise.P_phot[:ib2b] = (noise.emissivities * noise.tr_prod * h * nu /
                         (np.exp(h * nu / k / noise.temperatures) - 1))[:ib2b, None] * \
                        noise.g[:ib2b]

        noise.P_phot[:ib2b] = noise.P_phot[:ib2b] * self.detector.efficiency
        noise.NEP_phot2_nobunch[:ib2b] = h * nu * noise.P_phot[:ib2b] * 2
        # note the factor 2 in the definition of the NEP^2
        noise.NEP_phot2[:ib2b] = noise.NEP_phot2_nobunch[:ib2b] * (1 + noise.P_phot[:ib2b] /
                                                       (h * nu * noise.g[:ib2b]))

        if self.debug: self._raise_debug(noise, noise.ib2b, 
                                        before_b2b = True)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[:ib2b], axis = 1))
            loadsampling = []
            for inep in nep_intern:
                loadsampling.append(Instrument.get_noise(self, sampling, 
                                                        nep = inep))
            return {"power": noise.P_phot[:ib2b],
                    "NEP_phot2_nobunch": noise.NEP_phot2_nobunch[:ib2b],
                    "NEP_phot2": noise.NEP_phot2[:ib2b],
                    "NEP_array": np.array(loadsampling)}
        else:
            return

    def NEP_horns(self, noise, 
                  return_only = False, sampling = None):
        """
        This method calculates the noise of the array of horns.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        ib2b = noise.ib2b
        #150GHz band
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            #print("======== 150GHz band horns NEP")
            # back to back horns, as seen by the detectors through the combiner
            T = noise.temperatures[ib2b]
            b = h * noise.nu_up / k / T
            I1 = quad(funct, 0, b, (4, 1))[0]
            I2 = quad(funct, 0, b, (4, 2))[0]
            K1 = quad(funct, 0, b, (3, 1))[0]
            eta = (noise.emissivities * noise.tr_prod)[ib2b] * \
                                    self.detector.efficiency
            # Here the physical horn area S_horns must be used
            noise.NEP_phot2[ib2b] = 2 * noise.gp[ib2b] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                              (I1 + eta * I2) * noise.S_horns * noise.omega_det * noise.sec_beam
            noise.P_phot[ib2b] = noise.gp[ib2b] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * K1 * \
                           noise.S_horns * noise.omega_det * noise.sec_beam
        else: #220GHz band
            # back to back horns, as seen by the detectors through the combiner   
            # Here the physical horn area S_horns must be used
            noise.g[ib2b] = noise.gp[ib2b, None] * noise.S_horns * noise.omega_det * (self.filter.nu / c) ** 2 * \
                      noise.sec_beam * noise.dnu
            #[MARTIN note: tr_prod has the proper indexes? (see eta computation for 150GHz band) ]
            noise.P_phot[ib2b] = (noise.emissivities * noise.tr_prod * h * self.filter.nu /
                            (np.exp(h * self.filter.nu / k / noise.temperatures[ib2b]) - 1))[ib2b, None] * \
                           noise.g[ib2b]
            noise.P_phot[ib2b] = noise.P_phot[ib2b] * self.detector.efficiency
            noise.NEP_phot2_nobunch[ib2b] = h * self.filter.nu * noise.P_phot[ib2b] * 2
            # note the factor 2 in the definition of the NEP^2
            noise.NEP_phot2[ib2b] = noise.NEP_phot2_nobunch[ib2b] * (1 + noise.P_phot[ib2b] /
                                                         (h * self.filter.nu * noise.g[ib2b]))
            
        if self.debug: self._raise_debug(noise, noise.ib2b)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[ib2b]))
            return {"power": noise.P_phot[ib2b],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[ib2b],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return               

    def NEP_environment(self, noise, names, 
                        return_only = False, sampling = None):

        """
        This method calculates the environment noise.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            names: (noise.names attribute)
                names of the components considered in the instrument model for the noise.  
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2_env" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """

        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        ib2b = noise.ib2b
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            T = noise.temperatures[ib2b]
            b = h * noise.nu_up / k / T
            I1 = quad(funct, 0, b, (4, 1))[0]
            I2 = quad(funct, 0, b, (4, 2))[0]
            K1 = quad(funct, 0, b, (3, 1))[0]

            eff_factor = np.prod(noise.transmissions[(len(names) - 4):]) * \
                         self.detector.efficiency
            noise.P_phot_env = noise.gp[ib2b] * eff_factor * noise.omega_coldstop * noise.S_det * \
                         (k * noise.temperatures[ib2b]) ** 4 / c ** 2 / h ** 3 * K1
            noise.NEP_phot2_env = 4 * noise.omega_coldstop * noise.S_det * \
                            (k * noise.temperatures[ib2b]) ** 5 / c ** 2 / h ** 3 * \
                            eff_factor * (I1 + I2 * eff_factor)
            NEP_phot2_env_nobunch = None
 
        else:##220GHz:
            eff_factor = np.prod(noise.transmissions[(len(names) - 4):]) * \
                         self.detector.efficiency
            g_env = noise.gp[ib2b, None] * noise.S_det * noise.omega_coldstop * (self.filter.nu / c) ** 2 * \
                    noise.sec_beam * noise.dnu
            noise.P_phot_env = (eff_factor * h * self.filter.nu /
                          (np.exp(h * self.filter.nu / k / noise.temperatures[ib2b]) - 1))[ib2b, None] * \
                         g_env
            NEP_phot2_env_nobunch = h * self.filter.nu * noise.P_phot_env * 2
            # note the factor 2 in the definition of the NEP^2
            noise.NEP_phot2_env = NEP_phot2_env_nobunch * (1 + noise.P_phot_env /
                                                           (h * self.filter.nu * g_env))
            if self.debug:
                print("==========================")        
                print("Shape of NEP_phot2_env_nobunch = ", np.shape(NEP_phot2_env_nobunch))
                print("Shape of NEP_phot2_env_nobunch = ", np.shape(noise.P_phot_env))
                print("ib2b", ib2b)
                print("Value used in current version of qubicsoft", NEP_phot2_env_nobunch[ib2b])
                print("==========================")

        if self.debug: self._raise_debug(noise, noise.ib2b,
                                        environment = True)     

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2_env[ib2b]))
            return {"power": noise.P_phot_env,
                    "NEP_phot2_nobunch": NEP_phot2_env_nobunch,
                    "NEP_phot2_env": noise.NEP_phot2_env,
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern),
                    }#"debug_current": NEP_phot2_env_nobunch[ib2b] * (1 + noise.P_phot_env /(h * self.filter.nu * g_env)),
                    #"debug_new": NEP_phot2_env_nobunch * (1 + noise.P_phot_env / (h * self.filter.nu * g_env))}
        else:
            return

    def NEP_combiner(self, noise, 
                    return_only = False, sampling = None):

        """
        This method calculates the noise of the optical combiner (consider 2 mirrors).

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        icomb = noise.icomb
        #150GHz band
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            T = noise.temperatures[icomb]
            b = h * noise.nu_up / k / T
            J1 = quad(funct, 0, b, (4, 1))[0]
            J2 = quad(funct, 0, b, (4, 2))[0]
            L1 = quad(funct, 0, b, (3, 1))[0]
            eta = (noise.emissivities * noise.tr_prod)[icomb] * \
                                        self.detector.efficiency
            noise.P_phot[icomb] = noise.gp[icomb] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * L1 * \
                            noise.S_det * noise.omega_comb * noise.sec_beam
            noise.NEP_phot2[icomb] = 2 * noise.gp[icomb] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                               (J1 + eta * J2) * noise.S_det * noise.omega_comb * noise.sec_beam

        else: #220GHz band
            noise.g[icomb] = noise.gp[icomb] * noise.S_det * noise.omega_comb * (self.filter.nu / c) ** 2 * noise.dnu
            # The combiner emissivity includes the fact that there are 2
            # mirrors
            eta = (noise.emissivities * noise.tr_prod)[icomb] * \
                                        self.detector.efficiency
            noise.P_phot[icomb] = eta * h * self.filter.nu / \
                            (np.exp(h * self.filter.nu / k / noise.temperatures[icomb]) - 1) * noise.g[icomb] 
            noise.NEP_phot2_nobunch[icomb] = h * self.filter.nu * noise.P_phot[icomb] * 2
            noise.NEP_phot2[icomb] = noise.NEP_phot2_nobunch[icomb] * (1 + noise.P_phot[icomb] /
                                                           (h * self.filter.nu * noise.g[icomb]))

        if self.debug: self._raise_debug(noise, noise.icomb)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[icomb]))
            return {"power": noise.P_phot[icomb],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[icomb],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_coldstop(self, noise, 
                    return_only = False, sampling = None):

        """
        This method calculates the noise of the cold stop.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        ics = noise.ics
        #150GHz band
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            T = noise.temperatures[ics]
            b = h * noise.nu_up / k / T
            J1 = quad(funct, 0, b, (4, 1))[0]
            J2 = quad(funct, 0, b, (4, 2))[0]
            L1 = quad(funct, 0, b, (3, 1))[0]
            eta = (noise.emissivities * noise.tr_prod)[ics] * \
                                            self.detector.efficiency
            
            noise.NEP_phot2[ics] = 2 * noise.gp[ics] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                             (J1 + eta * J2) * noise.S_det * noise.omega_coldstop * noise.sec_beam
            noise.P_phot[ics] = noise.gp[ics] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * L1 * \
                          noise.S_det * noise.omega_coldstop * noise.sec_beam

        else:#220GHz band
            eta = noise.emissivities[ics] * noise.tr_prod[ics] * \
                                            self.detector.efficiency
            noise.g[ics] = noise.gp[ics] * noise.S_det * noise.omega_coldstop * (self.filter.nu / c) ** 2 * noise.dnu
            noise.P_phot[ics] =  eta * h * self.filter.nu / \
                          (np.exp(h * self.filter.nu / k / noise.temperatures[ics]) - 1) * noise.g[ics] 
            noise.NEP_phot2_nobunch[ics] = h * self.filter.nu * noise.P_phot[ics] * 2
            noise.NEP_phot2[ics] = noise.NEP_phot2_nobunch[ics] * (1 + noise.P_phot[ics] /
                                                       (h * self.filter.nu * noise.g[ics]))

        if self.debug: self._raise_debug(noise, noise.ics)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[ics]))
            return {"power": noise.P_phot[ics],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[ics],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_dichroic(self, noise, 
                    return_only = False, sampling = None):

        """
        This method calculates the noise of the dichroic. It's only accounted for the FI configuration.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        idic = noise.idic
        if (self.filter.nu <= self.nu1_up) and (self.filter.nu >= self.nu1_down):
            T = noise.temperatures[idic]
            b = h * noise.nu_up / k / T
            J1 = quad(funct, 0, b, (4, 1))[0]
            J2 = quad(funct, 0, b, (4, 2))[0]
            L1 = quad(funct, 0, b, (3, 1))[0]
            eta = (noise.emissivities * noise.tr_prod)[idic] * \
                                                self.detector.efficiency
            noise.g[idic] = noise.gp[idic] * noise.S_det * noise.omega_dichro 
            noise.P_phot[idic] = noise.g[idic] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * L1 * \
                                        noise.sec_beam
            noise.NEP_phot2[idic] = 2 * noise.g[idic] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                              (J1 + eta * J2) * noise.sec_beam
        else: #220GHz
            eta = (noise.emissivities * noise.tr_prod)[idic] * \
                                                self.detector.efficiency
            noise.g[idic] = noise.gp[idic] * noise.S_det * noise.omega_dichro * (noise.nu / c) ** 2 * noise.dnu
            noise.P_phot[idic] = h * noise.nu / \
                           (np.exp(h * noise.nu / k / noise.temperatures[idic]) - 1) * noise.g[idic]
            noise.NEP_phot2_nobunch[idic] = h * noise.nu * noise.P_phot[idic] * 2
            noise.NEP_phot2[idic] = noise.NEP_phot2_nobunch[idic] * (1 + noise.P_phot[idic] /
                                                         (h * noise.nu * noise.g[idic]))

        if self.debug: self._raise_debug(noise, noise.idic)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[idic]))
            return {"power": noise.P_phot[idic],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[idic],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_neutraldensityfilter(self, noise, 
                                return_only = False, sampling = None):

        """
        This method calculates the noise of the neutral density filter for 150GHz band. 
        In the case of the 220GHz the ndf is considered in an independent method called NEP_lastfilters_220.
        
        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        indf = noise.indf

        T = noise.temperatures[indf]
        b = h * noise.nu_up / k / T
        J1 = quad(funct, 0, b, (4, 1))[0]
        J2 = quad(funct, 0, b, (4, 2))[0]
        L1 = quad(funct, 0, b, (3, 1))[0]
        eta = (noise.emissivities * noise.tr_prod)[indf] * \
                                            self.detector.efficiency
        noise.NEP_phot2[indf] = 2 * noise.gp[indf] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                          (J1 + eta * J2) * noise.S_det * np.pi * noise.sec_beam
        noise.P_phot[indf] = noise.gp[indf] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * L1 * \
                       noise.S_det * np.pi * noise.sec_beam

        if self.debug: self._raise_debug(noise, noise.indf)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[indf]))
            return {"power": noise.P_phot[indf],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[indf],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_lowpassedge(self, noise, i, 
                        return_only = False, sampling = None):

        """
        This method calculates the noise of the low pass edge filter.
        In the case of the 220GHz the ndf is considered in an independent method called NEP_lastfilters_220.

        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            i: index for the low pass edge filters (lpe1 or lpe2 attr of noise)
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        T = noise.temperatures[i]
        b = h * noise.nu_up / k / T
        J1 = quad(funct, 0, b, (4, 1))[0]
        J2 = quad(funct, 0, b, (4, 2))[0]
        L1 = quad(funct, 0, b, (3, 1))[0]
        eta = (noise.emissivities * noise.tr_prod)[i] * \
                                            self.detector.efficiency
        noise.NEP_phot2[i] = 2 * noise.gp[i] * eta * (k * T) ** 5 / c ** 2 / h ** 3 * \
                       (J1 + eta * J2) * noise.S_det * np.pi * noise.sec_beam
        noise.P_phot[i] = noise.gp[i] * eta * (k * T) ** 4 / c ** 2 / h ** 3 * L1 * \
                    noise.S_det * np.pi * noise.sec_beam

        if self.debug: self._raise_debug(noise, i)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[i]))
            return {"power": noise.P_phot[i],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[i],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_lastfilter(self, noise, 
                        return_only = False, sampling = None):

        """
        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        ilast = noise.ilast
        T = noise.temperatures[ilast]
        eta = noise.emissivities[ilast] * noise.tr_prod[ilast] * self.detector.efficiency
        noise.P_phot[ilast] = eta * noise.gp[ilast] * noise.S_det * sigma * T ** 4 / 2
        noise.NEP_phot2[ilast] = eta * 2 * noise.gp[ilast] * noise.S_det * np.pi * (k * T) ** 5 \
                           / c ** 2 / h ** 3 * (24.9 + eta * 1.1)        

        if self.debug: self._raise_debug(noise, noise.ilast)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[noise.ilast]))
            return {"power": noise.P_phot[noise.ilast],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[noise.ilast],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return

    def NEP_lpefilter_220(self, noise, i, 
                        return_only = False, sampling = None):

        """
        Arguments:
            noise: parameters for the computation of the noise. It is loaded from
                load_NEP_parameters method
            i: noise component index
            return_only:
                if True, the method returns a dictionary with the components of the noise
                sampled using sampling in Instrument.get_noise() method from pysimulators
                if False, the method just load the components of the photon noise in the noise argument
            sampling:
                qubic.get_sampling(dict) object
        return:
            if return_only --> dictionary with the following keys: 
                "power" --> photon power
                "NEP_phot2" --> NEP squared. shape = (#det,)
                "NEP_array" --> NEP array sampled. shape = (#det,#samples)
                
        """
        #Check if there are a sampling array in case you asked for one component of the photon noise  
        self._raise_sampling_error(return_only, sampling)

        if noise.emissivities[i] == 0.0:
            noise.P_phot[i] = 0.0
            noise.NEP_phot2[i] = 0.0
        else:
            noise.g[i] = noise.gp[i] * noise.S_det * noise.omega_dichro * (self.filter.nu / c) ** 2 * noise.dnu
            noise.P_phot[i] = noise.emissivities[i] * noise.tr_prod[i] * h * self.filter.nu / \
                        (np.exp(h * self.filter.nu / k / noise.temperatures[i]) - 1) * noise.g[i] * \
                        self.detector.efficiency
            noise.NEP_phot2_nobunch[i] = h * self.filter.nu * noise.P_phot[i] * 2
            noise.NEP_phot2[i] = noise.NEP_phot2_nobunch[i] * (1 + noise.P_phot[i] /
                                                   (h * self.filter.nu * noise.g[i]))

        if self.debug: self._raise_debug(noise, i)

        if return_only:
            nep_intern = np.sqrt(np.mean(noise.NEP_phot2[i]))
            return {"power": noise.P_phot[i],
                    "NEP_phot2_nobunch": None,
                    "NEP_phot2": noise.NEP_phot2[i],
                    "NEP_array": Instrument.get_noise(self, sampling, nep = nep_intern)}
        else:
            return
    
    def get_aperture_integration_operator(self):
        """
        Integrate flux density in the telescope aperture.
        Convert signal from W / m^2 / Hz into W / Hz.

        """
        nhorns = np.sum(self.horn.open)
        return HomothetyOperator(nhorns * np.pi * self.horn.radeff ** 2)

    def get_convolution_peak_operator(self, **keywords):
        """
        Return an operator that convolves the Healpix sky by the gaussian
        kernel that, if used in conjonction with the peak sampling operator,
        best approximates the synthetic beam.

        """
        if self.ripples:
            return ConvolutionRippledGaussianOperator(self.filter.nu,
                                                      **keywords)
        fwhm = self.synthbeam.peak150.fwhm * (150e9 / self.filter.nu)
        if "ripples" in keywords.keys():
            del keywords["ripples"]
        return HealpixConvolutionGaussianOperator(fwhm=fwhm, **keywords)

    def get_detector_integration_operator(self):
        """
        Integrate flux density in detector solid angles and take into account
        the secondary beam transmission.

        """
        return QubicInstrument._get_detector_integration_operator(
            self.detector.center, self.detector.area, self.secondary_beam,self.use_file)

    @staticmethod
    def _get_detector_integration_operator(position, area, secondary_beam, use_file):
        """
        Integrate flux density in detector solid angles and take into account
        the secondary beam transmission.

        """
        theta = np.arctan2(
            np.sqrt(np.sum(position[..., :2] ** 2, axis=-1)), position[..., 2])
        phi = np.arctan2(position[..., 1], position[..., 0])
        sr_det = -area / position[..., 2] ** 2 * np.cos(theta) ** 3
        sr_beam = secondary_beam.solid_angle
        sec = secondary_beam(theta, phi)

        if use_file:
            return DiagonalOperator(sr_det / sr_beam, broadcast="rightward")
        else:
            sec = secondary_beam(theta, phi)
            return DiagonalOperator(sr_det / sr_beam * sec, broadcast="rightward")

    def get_detector_response_operator(self, sampling, tau=None):
        """
        Return the operator for the bolometer responses.

        """
        if tau is None:
            tau = self.detector.tau
        sampling_period = sampling.period
        shapein = len(self), len(sampling)
        if sampling_period == 0:
            return IdentityOperator(shapein)
        return ConvolutionTruncatedExponentialOperator(
            tau / sampling_period, shapein=shapein)

    def get_filter_operator(self):
        """
        Return the filter operator.
        Convert units from W/Hz to W.

        """
        if self.filter.bandwidth == 0:
            return IdentityOperator()
        return HomothetyOperator(self.filter.bandwidth)

    def get_hwp_operator(self, sampling, scene):
        """
        Return the rotation matrix for the half-wave plate.

        """
        shape = (len(self), len(sampling))
        if scene.kind == "I":
            return IdentityOperator(shapein=shape)
        if scene.kind == "QU":
            return Rotation2dOperator(-4 * sampling.angle_hwp,
                                      degrees=True, shapein=shape + (2,))
        return Rotation3dOperator("X", -4 * sampling.angle_hwp,
                                  degrees=True, shapein=shape + (3,))

    def get_invntt_operator(self, sampling):
        """
        Return the inverse time-time noise correlation matrix as an Operator.

        """
        return Instrument.get_invntt_operator(
            self, sampling, fknee=self.detector.fknee,
            fslope=self.detector.fslope, ncorr=self.detector.ncorr,
            nep=self.detector.nep)

    def get_polarizer_operator(self, sampling, scene):
        """
        Return operator for the polarizer grid.
        When the polarizer is not present a transmission of 1 is assumed
        for the detectors on the first focal plane and of 0 for the other.
        Otherwise, the signal is split onto the focal planes.

        """
        nd = len(self)
        nt = len(sampling)
        grid = (self.detector.quadrant - 1) // 4

        if scene.kind == "I":
            if self.optics.polarizer:
                return HomothetyOperator(1 / 2)
            # 1 for the first detector grid and 0 for the second one
            return DiagonalOperator(1 - grid, shapein=(nd, nt),
                                    broadcast="rightward")

        if not self.optics.polarizer:
            raise NotImplementedError(
                "Polarized input is not handled without the polarizer grid.")

        z = np.zeros(nd)
        data = np.array([z + 0.5, 0.5 - grid, z]).T[:, None, None, :]
        return ReshapeOperator((nd, nt, 1), (nd, nt)) * \
               DenseBlockDiagonalOperator(data, shapein=(nd, nt, 3))

    def get_projection_operator(self, sampling, scene, verbose=True, interp_projection=False):
        """
        Return the peak sampling operator.
        Convert units from W to W/sr.

        Parameters
        ----------
        sampling : QubicSampling
            The pointing information.
        scene : QubicScene
            The observed scene.
        verbose : bool, optional
            If true, display information about the memory allocation.

        """
        horn = getattr(self, "horn", None)
        primary_beam = self.primary_beam
        #primary_beam = getattr(self, "primary_beam", None)

        if sampling.fix_az:
            rotation = sampling.cartesian_horizontal2instrument
        else:
            rotation = sampling.cartesian_galactic2instrument

        try:
            thetas = self.thetafits
            phis = self.thetafits
            vals = self.thetafits
        except:
            thetas, phis, vals = QubicInstrument._peak_angles(scene, self.filter.nu, self.detector.center, self.synthbeam, horn, primary_beam)
            freqs = 0

        # return QubicInstrument._get_projection_operator(
        #     rotation, scene, self.filter.nu, self.detector.center,
        #     self.synthbeam, horn, primary_beam, self.thetafits, self.phifits, 
        #     self.valfits, self.use_file, self.freqs, interp_projection=interp_projection, verbose=verbose)
        return QubicInstrument._get_projection_operator(
            rotation, scene, self.filter.nu, self.detector.center,
            self.synthbeam, horn, primary_beam, thetas, phis, 
            vals, self.use_file, freqs, interp_projection=interp_projection, verbose=verbose)

    
    @staticmethod
    def _get_projection_operator(
            rotation, scene, nu, position, synthbeam, horn, primary_beam,
            thetafits, phifits, valfits, use_file, freqs, interp_projection=False, verbose=True):

        if use_file == True and interp_projection:
            # Fuse
            ValueError("'use_file == True' case not implemented for the interpolated projection operator.")

        ndetectors = position.shape[0]
        ntimes = rotation.data.shape[0]
        nside = scene.nside

        if use_file == True:
            isfreq=int(np.floor(nu/1e9))
            frq=len(str(freqs[0]))
            
            if (frq <= 4):
                freqid=np.where(freqs == isfreq)
            else:
                freqid=np.where(freqs == nu)
            
            importshape=np.shape(np.shape(thetafits))
            
            if importshape[0] <= 2:
                thetas,phis,vals = thetafits, phifits, valfits

            else:
                thetafits=thetafits[freqid].reshape((np.shape(thetafits)[1],np.shape(thetafits)[2]))
                phifits=phifits[freqid].reshape((np.shape(phifits)[1],np.shape(phifits)[2]))
                valfits=valfits[freqid].reshape((np.shape(valfits)[1],np.shape(valfits)[2]))

                thetas, phis, vals, = thetafits, phifits, valfits
                print("Getting Thetas from Fits File")
    
            thetas, phis, vals = QubicInstrument.remove_significant_peaks(thetas, phis, vals, synthbeam)
        
        else:
            # We get info on synthbeam
            thetas, phis, vals = QubicInstrument._peak_angles(scene, nu, position, synthbeam, horn, primary_beam)
            # phis, thetas, vals = QubicInstrument._peak_angles(scene, nu, position, synthbeam, horn, primary_beam) # test --> not what is needed
            # phis += np.pi/2 # test to rotate the beam around LOS --> still need the mirror beam
            # phis = -phis #- np.pi/2 # test to get the mirrored beam + rotation 90 degrees to mimic data from Moon Salta

        # shape(vals)   : (ndetectors, npeaks)
        # shape(thetas) : (ndetectors, npeaks)

        npeaks = thetas.shape[-1]
        thetaphi = _pack_vector(thetas, phis)  # (ndetectors, npeaks, 2)
        direction = Spherical2CartesianOperator("zenith,azimuth")(thetaphi) 

        e_nf = direction[:, None, :, :]
        if nside > 8192:
            dtype_index = np.dtype(np.int64)
        else:
            dtype_index = np.dtype(np.int32)

        cls = {"I": FSRMatrix,
               "QU": FSRRotation2dMatrix,
               "IQU": FSRRotation3dMatrix}[scene.kind]

        ndims = len(scene.kind)
        nscene = len(scene)
        nscenetot = product(scene.shape[:scene.ndim])
        if nscene != nscenetot and interp_projection:
            # Fuse
            ValueError("'nscene != nscenetot' case not implemented for the interpolated projection operator.")

        if interp_projection:
            print("bilinear interpolation")
            # For each peak position we take the interpolation with the four neighbouring pixels
            nb_interp = 4
        else:
            # Or we just take the pixel pointed at
            nb_interp = 1

        index = np.zeros((ndetectors, ntimes, npeaks, nb_interp))
        if interp_projection:
            # Weights to compute the bilinear interpolation
            weights = np.zeros_like(index)
            # New operator to handle the interpolation
            c2h = Cartesian2HealpixOperator_bilin_interp(nside)
        else:
            c2h = Cartesian2HealpixOperator(nside)

        if nscene != nscenetot:
            table = np.full(nscenetot, -1, dtype_index)
            table[scene.index] = np.arange(len(scene), dtype=dtype_index)

        # We use info on synthbeam + general pointing position to get info on synthbeam pointing positions
        def func_thread(i):
            # e_nf[i] shape: (1, ncolmax, 3)
            # e_ni shape: (ntimes, ncolmax, 3)
            e_ni = rotation.T(e_nf[i].swapaxes(0, 1)).swapaxes(0, 1)
            if nscene != nscenetot:
                np.take(table, c2h(e_ni).astype(int), out=index[i])
            else:
                if interp_projection:
                    res = c2h.get_interpol(e_ni)
                    index[i], weights[i] = np.moveaxis(res[0], [0], [2]), np.moveaxis(res[1], [0], [2])
                else:
                    index[i, ..., 0] = c2h(e_ni)

        with pool_threading() as pool:
            pool.map(func_thread, range(ndetectors))

        for k in range(nb_interp):

            s = cls((ndetectors * ntimes * ndims, nscene * ndims), ncolmax=npeaks,
                dtype=synthbeam.dtype, dtype_index=dtype_index,
                verbose=verbose)
            
            if scene.kind == "I":
                value = s.data.value.reshape(ndetectors, ntimes, npeaks)
                if interp_projection:
                    print("The method 'got_projection_operator' with 'interp_projeciton=True' is not yet checked for scene.kind = 'I'.")
                    value[...] = vals[:, None, :] * weights[:, :, :, k]      # to be checked one day
                else:
                    value[...] = vals[:, None, :]
                shapeout = (ndetectors, ntimes)
            else:
                if str(dtype_index) not in ("int32", "int64") or \
                        str(synthbeam.dtype) not in ("float32", "float64"):
                    raise TypeError(
                        "The projection matrix cannot be created with types: {0} a"
                        "nd {1}.".format(dtype_index, synthbeam.dtype))
                
                if interp_projection:
                    func = "weighted_matrix_rot{0}d_i{1}_r{2}".format(
                        ndims, dtype_index.itemsize, synthbeam.dtype.itemsize)
                
                    getattr(flib.polarization, func)(
                        rotation.data.T, direction.T, s.data.ravel().view(np.int8),
                        vals.T, weights[:, :, :, k].T)
                else:
                    func = "matrix_rot{0}d_i{1}_r{2}".format(
                        ndims, dtype_index.itemsize, synthbeam.dtype.itemsize)

                    getattr(flib.polarization, func)(
                        rotation.data.T, direction.T, s.data.ravel().view(np.int8),
                        vals.T)

                if scene.kind == "QU":
                    shapeout = (ndetectors, ntimes, 2)
                else:
                    shapeout = (ndetectors, ntimes, 3)

            P = 0
            P = ProjectionOperator(s, shapeout=shapeout)
            P.matrix.data.index = index[..., k].reshape(ndetectors * ntimes, npeaks)
            if k == 0:
                total_P = P.copy()
            else:
                total_P = (total_P.__add__(P)).copy()
            
        return total_P

    def get_transmission_operator(self):
        """
        Return the operator that multiplies by the cumulative instrumental
        transmission.
        """
        return DiagonalOperator(
            np.product(self.optics.components["transmission"]) *
            self.detector.efficiency, broadcast="rightward")

    def remove_significant_peaks (thetas, phis, vals, synthbeam):
        
        #now remove insignificant peaks
        vals[~np.isfinite(vals)] = 0
        index = _argsort_reverse(vals)
        thetas = thetas[tuple(index)]
        phis = phis[tuple(index)]
        vals = vals[tuple(index)]
        cumval = np.cumsum(vals, axis=-1)
        imaxs = np.argmax(cumval >= synthbeam.fraction * cumval[:, -1, None],axis=-1) + 1
        imax = max(imaxs)
        # remove additional per-detector non-significant peaks
        # and remove potential NaN in theta, phi
        for idet, imax_ in enumerate(imaxs):
            vals[idet, imax_:] = 0
            thetas[idet, imax_:] = np.pi / 2 #XXX 0 fails in polarization.f90.src (en2ephi and en2etheta_ephi)
            phis[idet, imax_:] = 0

        return thetas, phis, vals
    
    @staticmethod
    def _peak_angles(scene, nu, position, synthbeam, horn, primary_beam):
        """
        Compute the angles and intensity of the synthetic beam peaks which
        accounts for a specified energy fraction.

        """
        theta, phi = QubicInstrument._peak_angles_kmax(
            synthbeam.kmax, horn.spacing, horn.angle, nu, position)
        val = np.array(primary_beam(theta, phi), dtype=float, copy=False)
        val[~np.isfinite(val)] = 0
        index = _argsort_reverse(val)

        theta = theta[tuple(index)]
        phi = phi[tuple(index)]
        val = val[tuple(index)]

        cumval = np.cumsum(val, axis=-1)
        imaxs = np.argmax(cumval >= synthbeam.fraction * cumval[:, -1, None],
                          axis=-1) + 1
        imax = max(imaxs)
        
        # slice initial arrays to discard the non-significant peaks
        theta = theta[:, :imax]
        phi = phi[:, :imax]
        val = val[:, :imax]

        # remove additional per-detector non-significant peaks
        # and remove potential NaN in theta, phi
        for idet, imax_ in enumerate(imaxs):
            val[idet, imax_:] = 0
            theta[idet, imax_:] = np.pi / 2  # XXX 0 fails in polarization.f90.src (en2ephi and en2etheta_ephi)
            phi[idet, imax_:] = 0
        solid_angle = synthbeam.peak150.solid_angle * (150e9 / nu) ** 2
        val *= solid_angle / scene.solid_angle * len(horn)
        return theta, phi, val
    
    # @staticmethod
    def _peak_angles_unsorted(self, scene, nu, position, synthbeam, horn, primary_beam):
        """
        Compute the angles and intensity of the synthetic beam peaks.

        """
        theta, phi = QubicInstrument._peak_angles_kmax(
            synthbeam.kmax, horn.spacing, horn.angle, nu, position)
        val = np.array(primary_beam(theta, phi), dtype=float, copy=False)
        val[~np.isfinite(val)] = 0

        solid_angle = synthbeam.peak150.solid_angle * (150e9 / nu) ** 2
        val *= solid_angle / scene.solid_angle * len(horn)
        return theta, phi, val

    @staticmethod
    def _peak_angles_kmax(kmax, horn_spacing, angle, nu, position):
        """
        Return the spherical coordinates (theta, phi) of the beam peaks,
        in radians up to a maximum diffraction order.
        Parameters
        ----------
        kmax : int, optional
            The diffraction order above which the peaks are ignored.
            For instance, a value of kmax=2 will model the synthetic beam by
            (2 * kmax + 1)**2 = 25 peaks and a value of kmax=0 will only sample
            the central peak.
        horn_spacing : float
            The spacing between horns, in meters.
        nu : float
            The frequency at which the interference peaks are computed.
        position : array of shape (..., 3)
            The focal plane positions for which the angles of the interference
            peaks are computed.
        """
        lmbda = c / nu
        position = -position / np.sqrt(np.sum(position ** 2, axis=-1))[..., None]
        
        if angle != 0:
            angle = np.radians(angle)
            _kx, _ky = np.mgrid[-kmax:kmax + 1, -kmax:kmax + 1]
            kx = _kx * np.cos(angle) - _ky * np.sin(angle)
            ky = _kx * np.sin(angle) + _ky * np.cos(angle)
        else:
            kx, ky = np.mgrid[-kmax:kmax + 1, -kmax:kmax + 1]
        
        nx = position[:, 0, None] - lmbda * kx.ravel() / horn_spacing
        ny = position[:, 1, None] - lmbda * ky.ravel() / horn_spacing
        local_dict = {"nx": nx, "ny": ny}
        theta = ne.evaluate("arcsin(sqrt(nx**2 + ny**2))",
                            local_dict=local_dict)
        phi = ne.evaluate("arctan2(ny, nx)", local_dict=local_dict)
        return theta, phi

    @staticmethod
    def _get_response_A(position, area, nu, horn, secondary_beam, external_A=None, hwp_position=0):
        """
        Phase and transmission from the switches to the focal plane.

        Parameters
        ----------
        position : array-like of shape (..., 3)
            The 3D coordinates where the response is computed [m].
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        secondary_beam : Beam
            The secondary beam.
        external_A : list of tables describing the phase and amplitude at each point of the focal
            plane for each of the horns:
            [0] : array, X coordinates with shape (n) in GRF [m]
            [1] : array, Y coordinates with shape (n) in GRF [m]
            [2] : array, amplitude on X with shape (n, nhorns)
            [3] : array, amplitude on Y with shape (n, nhorns)
            [4] : array, phase on X with shape (n, nhorns) [rad]
            [5] : array, phase on Y with shape (n, nhorns) [rad]
        hwp_position : int
            HWP position from 0 to 7.

        Returns
        -------
        out : complex array of shape (#positions, #horns)
            The phase and transmission from the horns to the focal plane.

        """
        if external_A is None:
            uvec = position / np.sqrt(np.sum(position ** 2, axis=-1))[..., None]
            thetaphi = Cartesian2SphericalOperator("zenith,azimuth")(uvec)
            sr = - area / position[..., 2] ** 2 * np.cos(thetaphi[..., 0]) ** 3
            tr = np.sqrt(secondary_beam(thetaphi[..., 0], thetaphi[..., 1]) *
                         sr / secondary_beam.solid_angle)[..., None]
            const = 2j * np.pi * nu / c
            product = np.dot(uvec, horn[horn.open].center.T)
            return ne.evaluate("tr * exp(const * product)")
        else:
            phi_hwp = np.arange(0, 8) * np.pi / 16
            xx = external_A[0]
            yy = external_A[1]
            ix = np.argmin(np.abs(xx - position[0, 0]))
            jy = np.argmin(np.abs(yy - position[0, 1]))
            Ax = external_A[2][:, ix, jy]
            Ay = external_A[3][:, ix, jy]
            phi_x = external_A[4][:, ix, jy]
            phi_y = external_A[5][:, ix, jy]
            Ex = Ax * (np.cos(phi_x) + 1j * np.sin(phi_x)) * np.cos(2 * phi_hwp[hwp_position])
            Ey = Ay * (np.cos(phi_y) + 1j * np.sin(phi_y)) * np.sin(2 * phi_hwp[hwp_position])
            A = Ex + Ey
            return A

    @staticmethod
    def _get_response_B(theta, phi, spectral_irradiance, nu, horn, primary_beam):
        """
        Return the complex electric amplitude and phase [W^(1/2)] from sources
        of specified spectral irradiance [W/m^2/Hz] going through each horn.

        Parameters
        ----------
        theta : array-like
            The source zenith angle [rad].
        phi : array-like
            The source azimuthal angle [rad].
        spectral_irradiance : array-like
            The source spectral power per unit surface [W/m^2/Hz].
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.

        Returns
        -------
        out : complex array of shape (#horns, #sources)
            The phase and amplitudes from the sources to the horns.

        """
        shape = np.broadcast(theta, phi, spectral_irradiance).shape
        theta, phi, spectral_irradiance = [np.ravel(_) for _ in [theta, phi, spectral_irradiance]]
        uvec = hp.ang2vec(theta, phi)
        source_E = np.sqrt(spectral_irradiance *
                           primary_beam(theta, phi) * np.pi * horn.radeff ** 2)
        const = 2j * np.pi * nu / c
        product = np.dot(horn[horn.open].center, uvec.T)
        out = ne.evaluate("source_E * exp(const * product)")
        return out.reshape((-1,) + shape)

    @staticmethod
    def _get_response(theta, phi, spectral_irradiance, position, area, nu,
                      horn, primary_beam, secondary_beam, external_A=None, hwp_position=0):
        """
        Return the monochromatic complex field [(W/Hz)^(1/2)] related to
        the electric field over a specified area of the focal plane created
        by sources of specified spectral irradiance [W/m^2/Hz]
        Frame used : GRF

        Parameters
        ----------
        theta : array-like
            The source zenith angle [rad].
        phi : array-like
            The source azimuthal angle [rad].
        spectral_irradiance : array-like
            The source spectral_irradiance [W/m^2/Hz].
        position : array-like of shape (..., 3)
            The 3D coordinates where the response is computed, in meters,
            in the GRF frame.
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.
        secondary_beam : Beam
            The secondary beam.
        external_A : list of tables describing the phase and amplitude at each point of the focal
            plane for each of the horns:
            [0] : array, X coordinates with shape (n) in GRF [m]
            [1] : array, Y coordinates with shape (n) in GRF [m]
            [2] : array, amplitude on X with shape (n, nhorns)
            [3] : array, amplitude on Y with shape (n, nhorns)
            [4] : array, phase on X with shape (n, nhorns) [rad]
            [5] : array, phase on Y with shape (n, nhorns) [rad]
        hwp_position : int
            HWP position from 0 to 7.

        Returns
        -------
        out : array of shape (#positions, #sources)
            The complex field related to the electric field over a speficied
            area of the focal plane, in units of (W/Hz)^(1/2).

        """
        A = QubicInstrument._get_response_A(
            position, area, nu, horn, secondary_beam, external_A=external_A, hwp_position=hwp_position)
        B = QubicInstrument._get_response_B(
            theta, phi, spectral_irradiance, nu, horn, primary_beam)
        E = np.dot(A, B.reshape((B.shape[0], -1))).reshape(
            A.shape[:-1] + B.shape[1:])
        return E

    @staticmethod
    def _get_synthbeam(scene, position, area, nu, bandwidth, horn,
                       primary_beam, secondary_beam, synthbeam_dtype=np.float32,
                       theta_max=45, external_A=None, hwp_position=0):
        """
        Return the monochromatic synthetic beam for a specified location
        on the focal plane, multiplied by a given area and bandwidth.
        Frame used : GRF

        Parameters
        ----------
        scene : QubicScene
            The scene.
        position : array-like of shape (..., 3)
            The 3D coordinates where the response is computed, in meters,
            in the GRF frame.
        area : array-like
            The integration area, in m^2.
        nu : float
            The frequency for which the response is computed [Hz].
        bandwidth : float
            The filter bandwidth [Hz].
        horn : PackedArray
            The horn layout.
        primary_beam : Beam
            The primary beam.
        secondary_beam : Beam
            The secondary beam.
        synthbeam_dtype : dtype, optional
            The data type for the synthetic beams (default: float32).
            It is the dtype used to store the values of the pointing matrix.
        theta_max : float, optional
            The maximum zenithal angle above which the synthetic beam is
            assumed to be zero, in degrees.
        external_A : list of tables describing the phase and amplitude at each point of the focal
            plane for each of the horns:
            [0] : array, X coordinates with shape (n) in GRF [m]
            [1] : array, Y coordinates with shape (n) in GRF [m]
            [2] : array, amplitude on X with shape (n, nhorns)
            [3] : array, amplitude on Y with shape (n, nhorns)
            [4] : array, phase on X with shape (n, nhorns) [rad]
            [5] : array, phase on Y with shape (n, nhorns) [rad]
        hwp_position : int
            HWP position from 0 to 7.

        """
        MAX_MEMORY_B = 1e9
        theta, phi = hp.pix2ang(scene.nside, scene.index)
        index = np.where(theta <= np.radians(theta_max))[0]
        nhorn = int(np.sum(horn.open))
        npix = len(index)
        nbytes_B = npix * nhorn * 24
        ngroup = int(np.ceil(nbytes_B / MAX_MEMORY_B))
        out = np.zeros(position.shape[:-1] + (len(scene),),
                       dtype=synthbeam_dtype)
        for s in split(npix, ngroup):
            index_ = index[s]
            sb = QubicInstrument._get_response(
                theta[index_], phi[index_], bandwidth, position, area, nu,
                horn, primary_beam, secondary_beam, external_A=external_A, hwp_position=hwp_position)
            #out[..., index_] = abs2(sb, dtype=synthbeam_dtype)
            out[..., index_] = np.real(sb)**2 + np.imag(sb)**2
        return out

    def get_synthbeam(self, scene, idet=None, theta_max=45, external_A=None, hwp_position=0,
                      detector_integrate=None, detpos=None):
        """
        Return the detector synthetic beams, computed from the superposition
        of the electromagnetic fields.

        The synthetic beam B_d = (B_d,i) of a given detector d is such that
        the power I_d in [W] collected by this detector observing a sky S=(S_i)
        in [W/m^2/Hz] is:
            I_d = (S | B_d) = sum_i S_i * B_d,i.

        Example
        -------
        >>> scene = QubicScene(1024)
        >>> inst = QubicInstrument()
        >>> sb = inst.get_synthbeam(scene, 0)

        The power collected by the bolometers in W, given a sky in W/m²/Hz is:
        >>> sb = inst.get_synthbeam(scene)
        >>> sky = scene.ones()   # [W/m²/Hz]
        >>> P = np.dot(sb, sky)  # [W]

        Parameters
        ----------
        scene : QubicScene
            The scene.
        idet : int, optional
            The detector number. By default, the synthetic beam is computed for
            all detectors.
        theta_max : float, optional
            The maximum zenithal angle above which the synthetic beam is
            assumed to be zero, in degrees.
        external_A : list of tables describing the phase and amplitude at each point of the focal
            plane for each of the horns:
            [0] : array, X coordinates with shape (n) in GRF [m]
            [1] : array, Y coordinates with shape (n) in GRF [m]
            [2] : array, amplitude on X with shape (n, nhorns)
            [3] : array, amplitude on Y with shape (n, nhorns)
            [4] : array, phase on X with shape (n, nhorns) [rad]
            [5] : array, phase on Y with shape (n, nhorns) [rad]
        hwp_position : int
            HWP position from 0 to 7.
        detector_integrate: Optional, number of subpixels in x direction for integration over detectors
            default (None) is no integration => uses the center of the pixel
        detpos: Optional, position in the focal plane at which the Synthesized Beam is desired as np.array([x,y,z])
        

        """
        if detpos is None:
            pos = self.detector.center
        else:
            pos = detpos

        if (idet is not None) and (detpos is None):
            return self[idet].get_synthbeam(scene, theta_max=theta_max, external_A=external_A,
                                            hwp_position=hwp_position, detector_integrate=detector_integrate)[0]
        if detector_integrate is None:
            return QubicInstrument._get_synthbeam(
                scene, pos, self.detector.area, self.filter.nu,
                self.filter.bandwidth, self.horn, self.primary_beam,
                self.secondary_beam, self.synthbeam.dtype, theta_max, external_A=external_A, hwp_position=hwp_position)
        else:
            xmin = np.min(self.detector.vertex[..., 0:1])
            xmax = np.max(self.detector.vertex[..., 0:1])
            ymin = np.min(self.detector.vertex[..., 1:2])
            ymax = np.max(self.detector.vertex[..., 1:2])
            allx = np.linspace(xmin, xmax, detector_integrate)
            ally = np.linspace(ymin, ymax, detector_integrate)
            sb = 0
            for i in range(len(allx)):
                for j in range(len(ally)):
                    pos = self.detector.center
                    pos[0][0] = allx[i]
                    pos[0][1] = ally[j]
                    sb += QubicInstrument._get_synthbeam(
                        scene, pos, self.detector.area, self.filter.nu,
                        self.filter.bandwidth, self.horn, self.primary_beam,
                        self.secondary_beam, self.synthbeam.dtype, theta_max,
                        external_A=external_A, hwp_position=hwp_position) / detector_integrate ** 2
            return sb

    def detector_subset(self, dets):
        subset_inst = copy.deepcopy(self)
        subset_inst.detector = self.detector[dets]
        return subset_inst


def _argsort_reverse(a, axis=-1):
    i = list(np.ogrid[[slice(x) for x in a.shape]])
    i[axis] = a.argsort(axis)[:, ::-1]
    return i


def _pack_vector(*args):
    shape = np.broadcast(*args).shape
    out = np.empty(shape + (len(args),))
    for i, arg in enumerate(args):
        out[..., i] = arg
    return out

class QubicMultibandInstrumentTrapezoidalIntegration:
    """
    The QubicMultibandInstrument class
    Represents the QUBIC multiband features
    as an array of QubicInstrumet objects
    """

    def __init__(self, d):
        """
        filter_nus -- base frequencies array
        filter_relative_bandwidths -- array of relative bandwidths
        center_detector -- bolean, optional
        if True, take only one detector at the centre of the focal plane
            Needed to study the synthesised beam
        """

        self.FRBW = d["filter_relative_bandwidth"]  # initial Full Relative Band Width
        self.d = d.copy()
        d1 = d.copy()

        _, nus_edge150, filter_nus150, deltas150, _, _ = compute_freq(
            150, int(d["nf_sub"] / 2), relative_bandwidth=self.FRBW
        )

        _, nus_edge220, filter_nus220, deltas220, _, _ = compute_freq(
            220, int(d["nf_sub"] / 2), relative_bandwidth=self.FRBW
        )
        
        delta_nu_over_nu_150 = deltas150 / filter_nus150
        delta_nu_over_nu_220 = deltas220 / filter_nus220

        if not d["center_detector"]:
            self.subinstruments = []
            for i in range(len(filter_nus150)):
                if self.d["debug"]:
                    print(
                        f"Integration done with nu = {nus_edge150[i]} GHz with weight {delta_nu_over_nu_150[i]}"
                    )
                d1["filter_nu"] = filter_nus150[i] * 1e9

                if d['debug']:
                    print("setting filter_nu to ",d1["filter_nu"])

                d1["filter_relative_bandwidth"] = delta_nu_over_nu_150[i]
                self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            if d["config"] == "FI": # "TD" has only one band
                for i in range(len(filter_nus220)):
                    if self.d["debug"]:
                        print(
                            f"Integration done with nu = {nus_edge220[i]} GHz with weight {delta_nu_over_nu_220[i]}"
                        )
                    d1["filter_nu"] = filter_nus220[i] * 1e9
                    d1["filter_relative_bandwidth"] = delta_nu_over_nu_220[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            else:
                print("Ignoring band at 220 GHz.")
        else: # Broken code

            self.subinstruments = []
            for i in range(self.nsubbands):
                d1["filter_nu"] = filter_nus150[i] * 1e9
                d1["filter_relative_bandwidth"] = delta_nu_over_nu_150[i]
                q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                q.detector.center = np.array([[0.0, 0.0, -0.3]])
                self.subinstruments.append(q)

            for i in range(self.nsubbands):
                d1["filter_nu"] = filter_nus220[i] * 1e9
                d1["filter_relative_bandwidth"] = delta_nu_over_nu_220[i]
                q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                q.detector.center = np.array([[0.0, 0.0, -0.3]])
                self.subinstruments.append(q)

            """

            "Trapeze" integration
            -------------------------------

            W150 = IntegrationTrapezeOperator(nus_edge150)
            deltas150_trap = np.array(
                [W150.operands[i].todense(shapein=1)[0][0] for i in range(len(nus_edge150))]
            )
            W220 = IntegrationTrapezeOperator(nus_edge220)
            deltas220_trap = np.array(
                [W220.operands[i].todense(shapein=1)[0][0] for i in range(len(nus_edge220))]
            )

            delta_nu_over_nu_150 = deltas150_trap / nus_edge150
            delta_nu_over_nu_220 = deltas220_trap / nus_edge220

            if not d["center_detector"]:
                self.subinstruments = []
                W = IntegrationTrapezeOperator(nus_edge150)
                for i in range(len(nus_edge150)):
                    if self.d["debug"]:
                        print(
                            f"Integration done with nu = {nus_edge150[i]} GHz with weight {delta_nu_over_nu_150[i]}"
                        )
                    # print(nus_edge150)
                    d1["filter_nu"] = nus_edge150[i] * 1e9
                    d1["filter_relative_bandwidth"] = delta_nu_over_nu_150[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]

                W = IntegrationTrapezeOperator(nus_edge220)
                for i in range(len(nus_edge220)):
                    if self.d["debug"]:
                        print(
                            f"Integration done with nu = {nus_edge220[i]} GHz with weight {delta_nu_over_nu_220[i]}"
                        )
                    # print(nus_edge220)
                    d1["filter_nu"] = nus_edge220[i] * 1e9
                    d1["filter_relative_bandwidth"] = delta_nu_over_nu_220[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            else:

                self.subinstruments = []
                W = IntegrationTrapezeOperator(nus_edge150)
                for i in range(self.nsubbands):
                    d1["filter_nu"] = nus_edge150[i] * 1e9
                    d1["filter_relative_bandwidth"] = delta_nu_over_nu_150[i]
                    q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                    q.detector.center = np.array([[0.0, 0.0, -0.3]])
                    self.subinstruments.append(q)

                W = IntegrationTrapezeOperator(nus_edge220)
                for i in range(self.nsubbands):
                    d1["filter_nu"] = nus_edge220[i] * 1e9
                    d1["filter_relative_bandwidth"] = delta_nu_over_nu_220[i]
                    q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                    q.detector.center = np.array([[0.0, 0.0, -0.3]])
                    self.subinstruments.append(q)

            """

    def __getitem__(self, i):
        return self.subinstruments[i]

    def __len__(self):
        return len(self.subinstruments)

    def get_synthbeam(
        self, scene, idet=None, theta_max=45, detector_integrate=None, detpos=None
    ):
        sb = map(
            lambda i: i.get_synthbeam(
                scene,
                idet,
                theta_max,
                detector_integrate=detector_integrate,
                detpos=detpos,
            ),
            self.subinstruments,
        )
        sb = np.array(sb)
        bw = np.zeros(len(self))
        for i in range(len(self)):
            bw[i] = self[i].filter.bandwidth / 1e9
            sb[i] *= bw[i]
        sb = sb.sum(axis=0) / np.sum(bw)
        return sb

    def direct_convolution(self, scene, idet=None):
        synthbeam = [q.synthbeam for q in self.subinstruments]
        for i in range(len(synthbeam)):
            synthbeam[i].kmax = 4
        sb_peaks = map(
            lambda i: QubicInstrument._peak_angles(
                scene,
                self[i].filter.nu,
                self[i][idet].detector.center,
                synthbeam[i],
                self[i].horn,
                self[i].primary_beam,
            ),
            range(len(self)),
        )

        def peaks_to_map(peaks):
            m = np.zeros(hp.nside2npix(scene.nside))
            m[hp.ang2pix(scene.nside, peaks[0], peaks[1])] = peaks[2]
            return m

        sb = map(peaks_to_map, sb_peaks)
        C = [i.get_convolution_peak_operator() for i in self.subinstruments]
        sb = [(C[i])(sb[i]) for i in range(len(self))]
        sb = np.array(sb)
        sb = sb.sum(axis=0)
        return sb

    def detector_subset(self, dets):
        subset_inst = copy.deepcopy(self)
        for i in range(len(subset_inst)):
            subset_inst[i].detector = self[i].detector[dets]
        return subset_inst
    

class QubicMultibandInstrument:
    """
    The QubicMultibandInstrument class
    Represents the QUBIC multiband features 
    as an array of QubicInstrumet objects
    """

    def __init__(self, d):

        """
        filter_nus -- base frequencies array
        filter_relative_bandwidths -- array of relative bandwidths 
        center_detector -- bolean, optional
        if True, take only one detector at the centre of the focal plane
            Needed to study the synthesised beam
        """
        
        self.FRBW = d["filter_relative_bandwidth"]  # initial Full Relative Band Width
        self.d = d.copy()
        d1 = d.copy()

        ### Monochromatic
        if d["nf_sub"] == 1 and d["type_instrument"] != "wide" :
            band = d["filter_nu"]

            Nf, nus_edge220, filter_nus, deltas, Delta, Nbbands = compute_freq(band,
                                                                        d["nf_sub"],
                                                                        d["filter_relative_bandwidth"])
            
            
            self.nsubbands = len(filter_nus)
            if not d["center_detector"]:

                self.subinstruments = []
                for i in range(len(filter_nus)):

                    d1["filter_nu"] = filter_nus[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas[i] / filter_nus[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            else:

                self.subinstruments = []
                for i in range(self.nsubbands):
                    d1["filter_nu"] = filter_nus[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas[i] / filter_nus[i]
                    q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                    q.detector.center = np.array([[0., 0., -0.3]])
                    self.subinstruments.append(q)
        
        elif d["nf_sub"] == 1 and d["type_instrument"] == "wide":

            Nf, nus_edge150, filter_nus150, deltas150, Delta, Nbbands = compute_freq(150,
                                                                        d["nf_sub"],
                                                                        0.25)
            Nf, nus_edge220, filter_nus220, deltas220, Delta, Nbbands = compute_freq(220,
                                                                        d["nf_sub"],
                                                                        0.25)
            
            if not d["center_detector"]:

                self.subinstruments = []
                for i in range(len(filter_nus150)):

                    d1["filter_nu"] = filter_nus150[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas150[i] / filter_nus150[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]

                for i in range(len(filter_nus220)):

                    d1["filter_nu"] = filter_nus220[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas220[i] / filter_nus220[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            else:

                self.subinstruments = []
                for i in range(self.nsubbands):
                    d1["filter_nu"] = filter_nus[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas[i] / filter_nus[i]
                    q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                    q.detector.center = np.array([[0., 0., -0.3]])
                    self.subinstruments.append(q)
        ### Multichromatic
        else:
            Nf, nus_edge150, filter_nus, deltas, Delta, Nbbands = compute_freq(150,
                                                                        d["nf_sub"]-1,
                                                                        0.25)
            Nf, nus_edge220, filter_nus, deltas, Delta, Nbbands = compute_freq(220,
                                                                        d["nf_sub"]-1,
                                                                        0.25)

            if d["type_instrument"] == "wide":
                self.nsubbands = len(filter_nus)
                if not d["center_detector"]:
                    self.subinstruments = []
                    W = IntegrationTrapezeOperator(nus_edge150)
                    for i in range(len(nus_edge150)):
                        if self.d["debug"]:
                            print(f"Integration done with nu = {nus_edge150[i]} GHz with weight {W.operands[i].todense(shapein=1)[0][0]}")

                        d1["filter_nu"] = nus_edge150[i] * 1e9
                        d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge150[i]
                        self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]

                    W = IntegrationTrapezeOperator(nus_edge220)
                    for i in range(len(nus_edge220)):
                        if self.d["debug"]:
                            print(f"Integration done with nu = {nus_edge220[i]} GHz with weight {W.operands[i].todense(shapein=1)[0][0]}")

                        d1["filter_nu"] = nus_edge220[i] * 1e9
                        d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge220[i]
                        self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
                else:
                    self.subinstruments = []
                    W = IntegrationTrapezeOperator(nus_edge150)
                    for i in range(self.nsubbands):
                        d1["filter_nu"] = nus_edge150[i] * 1e9
                        d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge150[i]
                        q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                        q.detector.center = np.array([[0., 0., -0.3]])
                        self.subinstruments.append(q)

                    W = IntegrationTrapezeOperator(nus_edge220)
                    for i in range(self.nsubbands):
                        d1["filter_nu"] = nus_edge220[i] * 1e9
                        d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge220[i]
                        q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                        q.detector.center = np.array([[0., 0., -0.3]])
                        self.subinstruments.append(q)

            else:
                self.nsubbands = len(filter_nus)

                if not d["center_detector"]:
                    if self.d["filter_nu"] == 150e9:
                        self.subinstruments = []
                        W = IntegrationTrapezeOperator(nus_edge150)
                        for i in range(len(nus_edge150)):
                            if self.d["debug"]:
                                print(f"Integration done with nu = {nus_edge150[i]} GHz with weight {W.operands[i].todense(shapein=1)[0][0]}")

                            d1["filter_nu"] = nus_edge150[i] * 1e9
                            d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge150[i]
                            self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]

                    elif self.d["filter_nu"] == 220e9:
                        self.subinstruments = []
                        W = IntegrationTrapezeOperator(nus_edge220)
                        for i in range(len(nus_edge220)):
                            if self.d["debug"]:
                                print(f"Integration done with nu = {nus_edge220[i]} GHz with weight {W.operands[i].todense(shapein=1)[0][0]}")

                            d1["filter_nu"] = nus_edge220[i] * 1e9
                            d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge220[i]
                            self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
                    else: raise TypeError("Wrong band")
                else:
                    if self.d["filter_nu"] == 150e9:
                        self.subinstruments = []
                        W = IntegrationTrapezeOperator(nus_edge150)
                        for i in range(len(nus_edge150)):

                            d1["filter_nu"] = nus_edge150[i] * 1e9
                            d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge150[i]
                            
                            q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                            q.detector.center = np.array([[0., 0., -0.3]])
                            self.subinstruments.append(q)

                    elif self.d["filter_nu"] == 220e9:
                        self.subinstruments = []
                        W = IntegrationTrapezeOperator(nus_edge220)
                        for i in range(len(nus_edge220)):

                            d1["filter_nu"] = nus_edge220[i] * 1e9
                            d1["filter_relative_bandwidth"] = W.operands[i].todense(shapein=1)[0][0]/nus_edge220[i]
                            q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                            q.detector.center = np.array([[0., 0., -0.3]])
                            self.subinstruments.append(q)
                    else: raise TypeError("Wrong band")

            """

            Old integration using rectangle
            -------------------------------

            self.nsubbands = len(filter_nus)
            if not d["center_detector"]:

                self.subinstruments = []
                for i in range(len(filter_nus)):

                    d1["filter_nu"] = filter_nus[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas[i] / filter_nus[i]
                    self.subinstruments += [QubicInstrument(d1, FRBW=self.FRBW)]
            else:

                self.subinstruments = []
                for i in range(self.nsubbands):
                    d1["filter_nu"] = filter_nus[i] * 1e9
                    d1["filter_relative_bandwidth"] = deltas[i] / filter_nus[i]
                    q = QubicInstrument(d1, FRBW=self.FRBW)[0]
                    q.detector.center = np.array([[0., 0., -0.3]])
                    self.subinstruments.append(q)

            """
            
    def __getitem__(self, i):
        return self.subinstruments[i]

    def __len__(self):
        return len(self.subinstruments)

    def get_synthbeam(self, scene, idet=None, theta_max=45, detector_integrate=None, detpos=None):
        sb = map(lambda i: i.get_synthbeam(scene, idet, theta_max,
                                           detector_integrate=detector_integrate, detpos=detpos),
                 self.subinstruments)
        sb = np.array(sb)
        bw = np.zeros(len(self))
        for i in range(len(self)):
            bw[i] = self[i].filter.bandwidth / 1e9
            sb[i] *= bw[i]
        sb = sb.sum(axis=0) / np.sum(bw)
        return sb

    def direct_convolution(self, scene, idet=None):
        synthbeam = [q.synthbeam for q in self.subinstruments]
        for i in range(len(synthbeam)):
            synthbeam[i].kmax = 4
        sb_peaks = map(lambda i: QubicInstrument._peak_angles(scene, self[i].filter.nu,
                                                              self[i][idet].detector.center,
                                                              synthbeam[i],
                                                              self[i].horn,
                                                              self[i].primary_beam),
                       range(len(self)))

        def peaks_to_map(peaks):
            m = np.zeros(hp.nside2npix(scene.nside))
            m[hp.ang2pix(scene.nside,
                         peaks[0],
                         peaks[1])] = peaks[2]
            return m

        sb = map(peaks_to_map, sb_peaks)
        C = [i.get_convolution_peak_operator() for i in self.subinstruments]
        sb = [(C[i])(sb[i]) for i in range(len(self))]
        sb = np.array(sb)
        sb = sb.sum(axis=0)
        return sb

    def detector_subset(self, dets):
        subset_inst = copy.deepcopy(self)
        for i in range(len(subset_inst)):
            subset_inst[i].detector = self[i].detector[dets]
        return subset_inst

