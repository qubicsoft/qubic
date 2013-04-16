# coding: utf-8
from __future__ import division

import healpy as hp
import numpy as np
import qubic_v1
from pysimulators import PointingMatrix, ProjectionInMemoryOperator
from scipy.constants import c, pi

from .operators import HealpixConvolutionGaussianOperator


__all__ = ['QubicInstrument']

class QubicInstrument(object):

    def __init__(self, fwhm_deg=14, focal_length=0.3, nu=150e9, dnu_nu=0,
                 ndetector=1024, detector_size=3e-3, nhorn=400, kappa=1.344,
                 horn_thickness=0.001, nside=256):
        self.init_sky(nside)
        self.init_primary_beam(fwhm_deg)
        self.init_optics(focal_length, nu, dnu_nu)
        self.init_detectors(ndetector, detector_size)
        self.init_horns(nhorn, horn_thickness, kappa)

    def init_sky(self, nside):
        class Sky(object):
            pass
        self.sky = Sky()
        self.sky.npixel = 12 * nside**2
        self.sky.nside = nside

    def init_primary_beam(self, fwhm_deg):
        class PrimaryBeam(object):
            def __init__(self, fwhm_deg):
                self.sigma = np.radians(fwhm_deg) / np.sqrt(8 * np.log(2))
                self.fwhm_deg = fwhm_deg
                self.fwhm_sr = 2 * pi * self.sigma**2
            def __call__(self, theta):
                return np.exp(-theta**2 / (2 * self.sigma**2))

        self.primary_beam = PrimaryBeam(fwhm_deg)

    def init_optics(self, focal_length, nu, dnu_nu):
        class Optics(object):
            pass
        optics = Optics()
        optics.focal_length = focal_length
        optics.nu = nu
        optics.dnu_nu = dnu_nu
        self.optics = optics

    def init_detectors(self, n, size_):
        class Detector(np.recarray):
            pass
        dtype = [('center', [('x', float), ('y', float)])]
        detector = Detector(n, dtype=dtype)

        nx = int(np.sqrt(n))
        if nx**2 != n:
            raise ValueError('Non-square arrays are not handled.')
        a = (nx * np.arange(nx) / (nx-1) - nx * 0.5) * size_
        x, y = np.meshgrid(a, a)
        detector.center.x = x.ravel()
        detector.center.y = y.ravel()
        detector.size_ = size_
        detector.spacing = size_
        self.detector = detector

    def init_horns(self, n, thickness, kappa):
        class Horn(np.recarray):
            pass
        dtype = [('center', [('x', float), ('y', float)])]
        horn = Horn(n, dtype=dtype)

        nx = int(np.sqrt(n))
        if nx**2 != n:
            raise ValueError('Non-square arrays are not handled.')
        lmbda = c / self.optics.nu
        surface = kappa**2 * lmbda**2 / self.primary_beam.fwhm_sr
        radius = np.sqrt(surface / pi) + thickness
        sizex = 2 * radius * nx
        a = -sizex * 0.5 + radius + sizex * np.arange(nx) / nx
        x, y = np.meshgrid(a, a)
        horn.center.x = x.ravel()
        horn.center.y = y.ravel()
        horn.kappa = kappa
        horn.spacing = a[1] - a[0]
        horn.thickness = thickness
        self.horn = horn

    def get_convolution_peak_operator(self, fwhm=np.radians(0.64883707),
                                      **keywords):
        return HealpixConvolutionGaussianOperator(self.sky.nside, fwhm=fwhm,
                                                  **keywords)

    def get_projection_peak_operator(self, pointing, kmax=2):
        matrix = _peak_pointing_matrix(self, kmax, pointing)
        return ProjectionInMemoryOperator(matrix)
        

def _peak_angles(q, kmax):
    """
    Return the spherical coordinates (theta,phi) of the beam peaks, in radians.

    """
    ndetector = len(q.detector)
    lmbda = c / q.optics.nu
    dx = q.horn.spacing
    detvec = np.vstack([-q.detector.center.x,
                        -q.detector.center.y,
                        np.zeros(ndetector) + q.optics.focal_length]).T
    detvec.T[...] /= np.sqrt(np.sum(detvec**2, axis=1))
    
    kx, ky = np.mgrid[-kmax:kmax+1,-kmax:kmax+1]
    nx = detvec[:,0,np.newaxis] - lmbda * kx.ravel() / dx
    ny = detvec[:,1,np.newaxis] - lmbda * ky.ravel() / dx  
    theta = np.arcsin(np.sqrt(nx**2 + ny**2))
    phi = np.arctan2(ny,nx)

    return theta, phi


def _peak_pointing_matrix(q, kmax, pointings):
    pointings = np.atleast_2d(pointings)
    npointing = len(pointings)
    ndetector = len(q.detector)
    npeak = (2 * kmax + 1)**2
    npixel = q.sky.npixel

    pointings = np.radians(pointings)
    theta0, phi0 = _peak_angles(q, kmax)
    weight0 = q.primary_beam(theta0).astype(np.float32)
    weight0 /= np.sum(weight0, axis=-1)[...,None]
    
    peakvec = hp.ang2vec(theta0.ravel(), phi0.ravel())
    shape = theta0.shape

    matrix = PointingMatrix.empty((ndetector, npointing, npeak), npixel, info={})

    for i, p in enumerate(pointings):
        theta, phi, psi = p
        newpeakvec = qubic_v1.qubic.rotateuv(peakvec, theta, phi, psi, inverse=True)
        newtheta, newphi = [a.reshape(shape) for a in hp.vec2ang(newpeakvec)]
        matrix[:,i,:]['index'] = hp.ang2pix(q.sky.nside, newtheta, newphi)
        matrix[:,i,:]['value'] = weight0

    return matrix


