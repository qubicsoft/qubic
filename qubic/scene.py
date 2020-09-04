from __future__ import absolute_import, division, print_function
import numpy as np
from pyoperators import asoperator, IdentityOperator, ReciprocalOperator
from pysimulators.interfaces.healpy import SceneHealpixCMB
from scipy.constants import c, h, k
import scipy

__all__ = ['QubicScene']


class Atmosphere(object):
    def __init__(self, temperature, emissivity, transmission):
        print('   Using Atmosphere: T={} em={} Trans={}'.format(temperature, emissivity, transmission))
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
        
        #################### Atmosphere emissivity - from JCH analysis of LLAMA site-testing data #############
        #### Atmosphere Temprature
        if d['TemperatureAtmosphere150'] is None:
            d['TemperatureAtmosphere150'] = 270.
        if d['TemperatureAtmosphere220'] is None:
            d['TemperatureAtmosphere220'] = 270.
        fT = scipy.interpolate.interp1d([150., 220.], [d['TemperatureAtmosphere150'], d['TemperatureAtmosphere220']], fill_value="extrapolate")

        #### Atmosphere Emissivity
        if d['EmissivityAtmosphere150'] is None:
            d['EmissivityAtmosphere150'] = 0.081
        if d['EmissivityAtmosphere220'] is None:
            d['EmissivityAtmosphere220'] = 0.138
        fE = scipy.interpolate.interp1d([150., 220.], [d['EmissivityAtmosphere150'], d['EmissivityAtmosphere220']], fill_value="extrapolate")

        self.atmosphere = Atmosphere(fT(d['filter_nu']/1e9), fE(d['filter_nu']/1e9), 1.)
        ########################################################################################################


        ################### Old code - Very wrong - Assumes an optimistic Dome C !!! ###########################
        # if summer:
        #     self.atmosphere = Atmosphere(233., 0.05, 1.)
        # else:
        #     self.atmosphere = Atmosphere(200., 0.015, 1.)
        ########################################################################################################

        SceneHealpixCMB.__init__(self, nside, kind=kind, absolute=absolute,
                                 temperature=temperature)
