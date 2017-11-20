from __future__ import absolute_import, division, print_function
import numpy as np
from pyoperators import asoperator, IdentityOperator, ReciprocalOperator
from pysimulators.interfaces.healpy import SceneHealpixCMB
from scipy.constants import c, h, k

__all__ = ['QubicScene']


class Atmosphere(object):
    def __init__(self, temperature, emissivity, transmission):
        self.temperature = temperature
        self.emissivity = emissivity
        self.transmission = transmission


class QubicScene(SceneHealpixCMB):
    def __init__(self, d):
        """
        Parameters
        ----------
        nside : int, optional
            The Healpix scene's nside.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        temperature : float, optional
            The CMB temperature used to convert a temperature fluctuation into
            power fluctuation (if absolute is False). The default value is
            taken from Fixsen et al. 2009.
        summer : boolean, optional
            If true, Dome C summer weather conditions are assumed for the
            atmosphere.

        """
        nside=d['nside']
        kind=d['kind']
        absolute=d['absolute']
        temperature=d['temperature']
        summer=d['summer']
        
        
        if summer:
            self.atmosphere = Atmosphere(233., 0.05, 1.)
        else:
            self.atmosphere = Atmosphere(200., 0.015, 1.)
        SceneHealpixCMB.__init__(self, nside, kind=kind, absolute=absolute,
                                 temperature=temperature)
