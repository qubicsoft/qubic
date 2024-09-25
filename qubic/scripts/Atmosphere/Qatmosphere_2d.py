import numpy as np
import scipy.constants as c
import healpy as hp
import scipy.special as sp
from scipy.integrate import quad
import camb.correlations as cc

from CoolProp import CoolProp as CP
from astropy.cosmology import Planck18

import qubic
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qscene import QubicScene
from qubic.lib.InstrumentModel.Qinstrument import QubicInstrument, compute_freq

from pyoperators import *
from pysimulators import (
    CartesianEquatorial2GalacticOperator,
    CartesianEquatorial2HorizontalOperator,
    CartesianHorizontal2EquatorialOperator,
    CartesianGalactic2EquatorialOperator,
    SamplingHorizontal,
    SphericalEquatorial2GalacticOperator,
    SphericalGalactic2EquatorialOperator,
    SphericalEquatorial2HorizontalOperator,
    SphericalHorizontal2EquatorialOperator)
from pysimulators.interfaces.healpy import Cartesian2HealpixOperator, Spherical2HealpixOperator

#TODO : Adjust rho_0 with PWV
#TODO : Verify conversion into µK_CMB
#TODO : Adjust pressure with density
#TODO : Verify frequency dependence of the atmosphere
#TODO : Verify if the atmosphere is in the same frame as the qubic
#TODO : Add the possibility to have a 3d atmosphere

class Atmsophere:
    
    def __init__(self, params):
        
        ### Import parameters files
        self.params = params
        self.qubic_dict = self.get_qubic_dict()
        
        self.lmax = 3*self.params['nside']-1
        
        ### Build atmsopheric coordinates
        # Cartesian coordinates
        if self.params['h_grid'] == 1:
            # 2d model
            self.altitude = (self.params['h_qubic'] + self.params['altitude_atm_2d']) * np.ones(self.params['h_grid'])
        else:
            # 3d model not yet implemented
            self.altitude = np.linspace(self.params['h_qubic'], self.params['altitude_atm_2d'], self.params['h_grid'])
        self.x_list = np.linspace(-self.params['size_atm'], self.params['size_atm'], self.params['n_grid'])
        self.y_list = np.linspace(-self.params['size_atm'], self.params['size_atm'], self.params['n_grid'])
        # Azimuth / Elevation coordinates
        x, y = np.meshgrid(self.x_list, self.y_list)
        z = np.ones(x.shape) * self.params['altitude_atm_2d']
        self.r, self.el, self.az = self.horizontal_plane_to_azel(x, y, z)
            
        ### Compute temperature and water vapor density
        self.temperature = self.get_temperature_atm(self.altitude)
        self.mean_water_vapor_density = self.get_mean_water_vapor_density(self.altitude)
        
        ### Compute absorption coefficients
        self.integration_frequencies, self.mol_absorption_coeff, self.self_absorption_coeff, self.air_absorption_coeff = self.atm_absorption_coeff()
                
        ### Compute absorption spectrum
        self.abs_spectrum = self.absorption_spectrum()
        self.integrated_abs_spectrum, self.frequencies = self.integrated_absorption_spectrum()
        
        ### Build maps
        self.maps = self.get_maps()
        
    def get_qubic_dict(self, key="in"):
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
            "nf_sub": self.params[f"nsub_{key}"],
            "nf_recon": self.params[f"nrec"],
            "filter_relative_bandwidth": 0.25,
            "npointings": self.params["npointings"],
            "nside": self.params["nside"],
            "MultiBand": True,
            "period": 1,
            "RA_center": 0,
            "DEC_center": -57,
            "filter_nu": 150 * 1e9,
            "noiseless": True,
            "dtheta": 15,
            "nprocs_sampling": 1,
            "photon_noise": False,
            "nhwp_angles": 3,
            #'effective_duration':3,
            "effective_duration150": 3,
            "effective_duration220": 3,
            "filter_relative_bandwidth": 0.25,
            "type_instrument": "two",
            "TemperatureAtmosphere150": None,
            "TemperatureAtmosphere220": None,
            "EmissivityAtmosphere150": None,
            "EmissivityAtmosphere220": None,
            "detector_nep": 4.7e-17,
            "synthbeam_kmax": 1,
        }

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        dict_qubic = qubicDict()
        dict_qubic.read_from_file(dictfilename)

        ### Modify the dictionary
        for i in args.keys():
            dict_qubic[str(i)] = args[i]

        return dict_qubic
        
    def atm_absorption_coeff(self):
        r"""Absorption spectrum.
        
        Method to build the absorption spectrum of the atmopshere using files computed using the am atmospheric model (Paine, 2018).
        The absorption coefficient has two origins: the line-by-line absorption which reprensents the spectral lines, 
        and a continuum absorption coming from collisions between molecules, 
        either :math:`H_2O-H_2O` collisions (self-induced continuum), or collisions between :math`H_2O` and the air (air_induced).

        Spectras for the line-by-line absorption-coefficient were produced using .amc files that looks like that:

        f 130 GHz  250 GHz  0.005 GHz\
        output f  k

        layer\
        P 500 hPa\
        T 280 K\
        column h2o_lines 1 mm_pwv

        And for the continuum absorption:

        f 130 GHz  250 GHz  0.005 GHz\
        output f  k

        layer\
        P 550 hPa\
        T 280 K\
        h 3000 m\
        column h2o_self_continuum 1 mm_pwv

        Then we only need to execute a command line like: am file_name.amc > file_name.out .

        The file 'file_name.out' is created and contains two colomns, one for the frequency, and one for the absorption coefficient. 
        Be careful, am results are in cm and not in m.

        See the am documentation for more details: https://zenodo.org/records/8161272 .

        Returns
        -------
        frequencies : array_like
            Frequencies at which the absorption spectrum is computed.
        mol_absorption_coeff : array_like
            Molecular absorption lines spectrum, from water and dioxygene, in :math:`m^{2}`.
        self_absorption_coeff : array_like
            Self-induced collisions continuum, from water, in :math:`m^{5}`.
        air_absorption_coeff : array_like
            Air_induced collisions continuun, from water, in :math:`m^{5}`.
            
        """
        
        frequencies = []
        mol_absorption_coeff = []
        self_absorption_coeff = []
        air_absorption_coeff = []
        
        ### Initialize atmosppheric properties
        pressure = self.params['pressure_atm']
        temp = self.params['temp_ground']
        pwv = self.params['pwv']
        
        ### Import absorption coefficients from molecular absorption lines
        with open(f'absorption_coefficient/h2o_lines_{pressure}hPa_{temp}K_{pwv}mm.out', 'r') as file:
            for line in file:
                frequencies.append(float(line.split()[0]))
                mol_absorption_coeff.append(float(line.split()[1]) * (1e-2)**2)
                
        ### Import absorption coefficients from self-induced collisions continuum
        with open(f'/home/laclavere/Documents/Thesis/qubic/qubic/scripts/Atmosphere/absorption_coefficient/h2o_self_continuum.out', 'r') as file:
            for line in file:
                self_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        ### Import absorption coefficients from air-induced collisions continuum
        with open(f'/home/laclavere/Documents/Thesis/qubic/qubic/scripts/Atmosphere/absorption_coefficient/h2o_air_continuum.out', 'r') as file:
            for line in file:
                air_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        return frequencies, np.array(mol_absorption_coeff), np.array(self_absorption_coeff), np.array(air_absorption_coeff)
    
    def get_gas_properties(self, params_file = True):
        r"""Gas properties.
        
        Method to compute the properties of the water vapor and the air in the atmosphere.
        It can use parameters given in self.params.yml or be baised on the CoolProp package to compute the water vapor density.

        Parameters
        ----------
        self.params_file : bool, optional
            If True, will use the reference water vapor density given in self.params_file by self.params['rho_0'] (in :math:`g/m^{3}`) to compute the density in :math:`m^{-3}`.
            Else, will use the temperature and pression value to compute it. By default True.

        Returns
        -------
        water_mass : float
            The weight of a :math:`H_2O` molecule in :math:`g`.
        water_vapor_density : float
            The density of water vapor given the atmospheric temperature and pressure in :math:`m^{-3}`.
        air_vapor_density : float
            The density of air vapor given the atmospheric temperature and pressure in :math:`m^{-3}`.
            rho_2d = np.zeros((Nx, Ny))

mean_delta = np.mean(delta_rho)
var_delta = np.var(delta_rho)

rho_2d += mean_water_vapor + sigma_simulated * (delta_rho - mean_delta) / np.sqrt(var_delta)
        
        """        
        
        ### Import physical constants
        temp_atm = self.temperature  # in K
        pressure_atm = self.params['pressure_atm'] * 100  # in Pa

        ### Air properties
        air_molar_mass = CP.PropsSI('MOLARMASS', 'Air')  # in g/mol
        air_mass = air_molar_mass / c.Avogadro * 1e3  # in g
        air_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Air")  # in kg/m-3
        air_density = air_mass_density * 1e3 / air_mass  # in m-3

        ### Water properties
        water_molar_mass = CP.PropsSI('MOLARMASS', 'Water')  # in g/mol
        water_mass = water_molar_mass / c.Avogadro * 1e3  # in g
        
        # Compute water vapor density
        if params_file:
            water_vapor_density = self.mean_water_vapor_density / water_mass  # in m-3
        else:
            water_vapor_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Water")  # in kg/m-3
            water_vapor_density = c.Avogadro * water_vapor_mass_density / water_molar_mass * 1e-3  # in m-3

        return water_mass, water_vapor_density, air_density
        
    
    def absorption_spectrum(self):
        r"""Absorption coefficient.
        
        The coefficient :math:`\alpha_b(\nu)` [:math:`m^2g^{-1}`] is defined by:
        :math:`\alpha_b(\nu) = \frac{1}{m_{H_2O}} \left(k_{lines}(\nu) + n_{H_2O}k_{self}(\nu) + n_{air}k_{air}(\nu)\right)`
        with :math:`m_{H_2O}= 2.992\times 10^{-23} g` the mass of a :math:`H_2O` molecule, math:`k_{lines}` [:math`m^2`] the line-by-line absorption coefficient, 
        :math`k_{self}` and :math`k_{air}` [:math:`m^5`] the self- and air-induced continua, :math:`n_{H_2O}` and :math:`n_{air}` [:maht:`m^{-3}`] the densities of water vapor and air.

        Returns
        -------
        abs_spectrum : array_like
            Atmosphere absorption coefficient, in :math:`m^{2} / g`.
            
        """        

        # Import gas properties
        water_mass, water_vapor_density, air_density = self.get_gas_properties()
                
        # Compute coeff
        abs_spectrum = (self.mol_absorption_coeff + water_vapor_density * self.self_absorption_coeff + air_density * self.air_absorption_coeff) / water_mass 

        return abs_spectrum
    
    def integrated_absorption_spectrum(self, band=150):
        """Integrated absorption spectrum.
        
        Compute the integrated absorption spectrum in a given frequency band, according to the parameters in the self.params.yml file.

        Parameters
        ----------
        band : int, optional
            QUBIC frequency band. Can be either 150 or 220, by default 150

        Returns
        -------
        integrated_abs_spectrum : array_like
            The integrated absorption spectrum in the given frequency band, in :math:`m^{2} / g`.
        nus : array_like
            The frequencies at which the absorption spectrum is computed, in :math:`GHz`.
            
        """        
        
        ### Evaluate the frequency band edges
        freq_min, freq_max = self.integration_frequencies[0], self.integration_frequencies[-1]
        freq_step = (freq_max - freq_min) / (len(self.integration_frequencies) - 1)

        ### Compute the frequency sub-bands within the QUBIC band and their associated indexes
        _, nus_edges, nus, _, _, N_bands = compute_freq(band=band, Nfreq=self.params['nsub_in'], relative_bandwidth=self.qubic_dict['filter_relative_bandwidth'])
        nus_edge_index = (nus_edges - freq_min) / freq_step

        ### Integrate the absorption spectrum over the frequency sub-bands using the trapezoidal method
        integrated_abs_spectrum = np.zeros(N_bands)
        for i in range(N_bands):
            index_inf, index_sup = int(nus_edge_index[i]), int(nus_edge_index[i+1])
            integrated_abs_spectrum[i] = np.trapz(self.abs_spectrum[index_inf:index_sup], 
                                                  x=self.integration_frequencies[index_inf:index_sup])
        
        return integrated_abs_spectrum, nus
    
    def get_mean_water_vapor_density(self, altitude):
        r"""Mean water vapor density.
        
        Compute the mean water vapor density depending on the altitude, using reference water vapor density and water vapor half_height, given in self.params.yml.
        The corresponding equation to compute the mean water vapor density, taken from equation (1) in Morris 2021, is :
        
        .. math::
            \langle \rho(h) \rangle = \rho_0 e^{\left( -log(2).(h - 5190) / h_0 \right)} .

        Parameters
        ----------
        altitude : array_like
            Array containing altitudes at which we want to compute the mean water vapor density.

        Returns
        -------
        mean_water_vapor_density : array_like
            Mean water vapor density in :math:`g/m{3}`.
            
        """  
        
        return self.params['rho_0'] * np.exp(-np.log(2) * (altitude - 5190) / self.params['h_h2o'])
    
    def get_temperature_atm(self, altitude):
        """Temperature.
        
        Compute the temperature of the atmosphere depending on the altitude, using the average ground temperature and a typical height that depend on the observation site.
        The corresponding equation to compute the temperature, taken from equation (13) in Morris 2021, is :
        
        .. math::
            T_{atm}(h) = T_{atm}(0) e^{h / h_T} .

        Parameters
        ----------
        altitude : array_like
            Array containing altitudes at which we want to compute the temperature.

        Returns
        -------
        temperature : array_like
            Temperature in K.
            
        """
        
        return self.params['temp_ground'] * np.exp(- altitude / self.params['h_temp'])
    
    def get_fourier_grid_2d(self):
        """Fourier 2d grid.
        
        Generate a 2d grid of spatial frequencies in Fourier space according to the parameters in the self.params.yml file.

        Returns
        -------
        kx : array_like
            2d array containing the spatial x frequencies in Fourier space, (n_grid, n_grid).
        ky : array_like
            2d array containing the spatial y frequencies in Fourier space, (n_grid, n_grid).
        k_norm : array_like
            2d array containing the norm of the spatial frequencies in Fourier space, (n_grid, n_grid).
            
        """        
        
        ### Generate spatial frequency in Fourier space
        k_distrib_y = np.fft.fftfreq(self.params['n_grid'], d=2*self.params['size_atm']/self.params['n_grid']) * 2*np.pi
        k_distrib_x = np.fft.fftfreq(self.params['n_grid'], d=2*self.params['size_atm']/self.params['n_grid']) * 2*np.pi
        
        ### Build 2d grid and compute the norm of the spatial frequencies
        kx, ky = np.meshgrid(k_distrib_x, k_distrib_y)
        k_norm = np.sqrt(kx**2 + ky**2)
        
        return kx, ky, k_norm
    
    def kolmogorov_spectrum_2d(self, k):
        r"""Kolmogorov 2d spectrum.
        
        Compute the Kolmogorov 2d spectrum, which simulate the power spectrum of the spatial fluctuations of the water vapor density, following the equation :
        
        .. math::
            P(\textbf{k}) = (r_0^{-2} + \lvert \textbf{k} \rvert ^{2})^{-8/6} .
        

        Parameters
        ----------
        k : array_like
            Array containing the spatial frequencies at which we want to compute the Kolmogorov 2d spectrum.

        Returns
        -------
        kolmogorov_spectrum_2d : array_like
            Kolmogorov 2d spectrum.
        """        
        
        return (self.params['correlation_length']**(-2) + np.abs(k)**2)**(-8/6)
    
    def normalized_kolmogorov_spectrum_2d(self, k):
        r"""Normalized Kolmogorov 2d spectrum.
        
        Compute the normalized Kolmogorov 2d spectrum, to ensure :
        
        .. math::
            \int_\textbf{k} P(\textbf{k}) d\textbf{k} = 1 .

        Parameters
        ----------
        k : array_like
            Array containing the spatial frequencies at which we want to compute the normalized Kolmogorov 2d spectrum.

        Returns
        -------
        normalized_kolmogorov_spectrum_2d : array_like
            Normalized Kolmogorov 2d spectrum.
            
        """        
        
        ### Compute the normalization constant
        res, _ = quad(self.kolmogorov_spectrum_2d, np.min(k), np.max(k))
        
        return self.kolmogorov_spectrum_2d(k) / res
    
    def generate_spatial_fluctuations_fourier(self):
        """Spatial 2d fluctuations.
        
        Produce the spatial fluctuations of the water vapor density, by generating random phases in Fourier space, and then computing the inverse Fourier transform.

        Returns
        -------
        delta_rho_x : array_like
            Variation of the water vapor density.
            
        """        
        #! At some point, we will need to normalize these fluctuations using real data. We can maybe use :math:`\sigma_{PWV}` that can be estimated with figure 4 in Morris 2021.
        
        ### Compute the spatial frequencies & power spectrum.
        _, _, k = self.get_fourier_grid_2d()
        kolmogorov_spectrum = self.normalized_kolmogorov_spectrum_2d(k)
        
        ### Generate spatial fluctuations through random phases in Fourier space
        phi = np.random.uniform(0, 2*np.pi, size=(self.params['n_grid'], self.params['n_grid']))
        delta_rho_k = np.sqrt(kolmogorov_spectrum) * np.exp(1j * phi)

        ### Apply inverse Fourier transform to obtain spatial fluctuations in real space
        delta_rho = np.fft.irfft2(delta_rho_k, s=(self.params['n_grid'], self.params['n_grid']))
        
        return delta_rho  
    
    def kolmogorov_correlation_function(self, r, r0):
        """Kolmogorov correlation function.
        
        Compute the Kolmogorov correlation function, which simulate the correlation of the spatial fluctuations of the water vapor density, following the equation :

        .. math::
            D(r) = \frac{2^{2/3}}{\Gamma(1/3)} \left(\frac{r}{r_0}\right)^{1/3} K_{1/3} \left(\frac{r}{r_0}\right) .
            
        We impose that the correlation function is 1 at r = 0.

        Parameters
        ----------
        r : array_like or float
            distance between two points, in meters.
        r0 : array_like or float
            Maximum correlation length, in meters.

        Returns
        -------
        D : array_like
            Correlation function.
            
        """   
             
        return np.where(r==0, 1, 2**(2/3)/sp.gamma(1/3)*(r/r0)**(1/3)*sp.kv(1/3, r/r0))    
    
    def angular_correlation(self, theta, h_atm, r0):
        """Angular Kolmogorov correlation function.
        
        We compute the angular Kolmogorov correlation function, switching the distance between two points to the angle between them on the surface of the sphere
        , using the relation :
        
        .. math::
            r = 2h_{atm} \sin \left(\frac{\theta}{2}\right) ,
            
        where :math:`h_{atm}` is the distance between the atmosphere and our instrument and :math:`\theta` is the angle between the two points.
        

        Parameters
        ----------
        theta : array_like or float
            Angle between two points, in degrees.
        h_atm : array_like or float
            Distance between the atmosphere and our instrument, in meters.

        Returns
        -------
        C : array_like or float
            Angular Kolmogorov correlation function.
            
        """        
        
        r = 2*h_atm*np.sin(np.radians(theta)/2)
        
        return self.kolmogorov_correlation_function(r, r0)
    
    def cl_from_angular_correlation_int(self, l):
        def integrand(cos_theta):
            theta = np.degrees(np.arccos(cos_theta))
            legendre = sp.legendre(l)(cos_theta)
            return self.angular_correlation(theta, self.params['altitude_atm_2d'], self.params['correlation_length']) * legendre
        res, _ = quad(integrand, -1, 1)
        
        return 2 * np.pi * res
    
    def ctheta_2_dell(self, theta_deg, ctheta, lmax, normalization=1):
        ### this is how camb recommends to prepare the x = cos(theta) values for integration
        ### These x values do not contain x=1 so we have. to do this case separately
        x, w = np.polynomial.legendre.leggauss(lmax+1)
        xdeg = np.degrees(np.arccos(x))

        ### We first replace theta=0 by 0 and do that case separately
        myctheta = ctheta.copy()
        myctheta[0] = 0
        ### And now we fill the array that should include polarization (we put zeros there)
        ### with the values of our imput c(theta) interpolated at the x locations
        allctheta = np.zeros((len(x), 4))
        allctheta[:,0] = np.interp(xdeg, theta_deg, myctheta)

        ### Here we call the camb function that does the transform to Cl
        dlth = cc.corr2cl(allctheta, x,  w, lmax)
        ell = np.arange(lmax+1)

        ### the special case x=1 corresponds to theta=0 and add 2pi times c(theta=0) to the Cell
        return ell, dlth[:,0]+ctheta[0]*normalization
    
    def ctheta_2_cell(self, theta_deg, ctheta, lmax, normalization=1):
        
        ### Compute multipole moments and Dl angular power spectrum
        ell, dlth = self.ctheta_2_dell(theta_deg, ctheta, lmax, normalization=normalization)
        
        ### Convert from Dl to Cl
        dl2cl_factor = 2*np.pi / (ell * (ell+1))
        clth = dlth * dl2cl_factor
        
        ### Correct for the special case l=0, as the convertion factor is not valid
        clth[0] = self.cl_from_angular_correlation_int(0)
        
        return ell, clth
    
    def generate_spatial_fluctuation_sphercial_harmonics(self):
            
        ### Compute angular correlation function
        theta = np.linspace(0, 180, self.params['n_grid'])
        ctheta = self.angular_correlation(theta, self.params['altitude_atm_2d'], self.params['correlation_length'])
        
        ### Compute spherical harmonics from angular correlation function
        _, clth = self.ctheta_2_cell(theta, ctheta, self.lmax, normalization=self.params['normalization'])
        
        ### Build fluctuations map
        delta_rho = hp.synfast(clth, nside=self.params['nside'], lmax=self.lmax)
        
        return delta_rho
        
    def get_water_vapor_density_2d_map(self):
        """Water vapor density 2d map.
        
        Get the water vapor density 2d map with simulated fluctuations.

        Returns
        -------
        atm_maps_2d : array_like
            Water vapor density 2d map.
            
        """        
        
        ### Import water vapor density map and fluctuations
        rho = self.get_mean_water_vapor_density(self.altitude)
        if self.params['flat']:
            delta_rho = self.generate_spatial_fluctuations_fourier()
        else:
            delta_rho = self.generate_spatial_fluctuation_sphercial_harmonics()
            
        ### Normalize fluctuations
        normalized_delta_rho = delta_rho * np.sqrt(self.params['sigma_rho'] / np.var(delta_rho))  

        return rho + normalized_delta_rho
        
    def get_maps(self):
        r"""Atmosphere maps.
        
        Get the atmosphere maps in temperature, by using the equation 12 from Morris 2021, that compute the induced temperature in the detector due to the water vapor density :
        
        .. math::
            dT( \teextbf{r}, \nu) = \alpha_b(\nu) \rho(\textbf{r}) T_{atm}(\textbf{r}) dV .
            
        And then, convert it in micro Kelvin CMB.

        Returns
        -------
        temp_maps : array_like
            Temperature maps in micro Kelvin CMB.
            
        """
        #! Does it have sense to compare this temperature with the one from CMB ? We talk about the CMB temperature because it follows a blackbody spectrum, but it is not the case for the atmosphere...
        #! We just have compute the corresponding measured temperature according to physical parameters.
        
        ### Get the water vapor density maps
        water_vapor_density_maps = self.get_water_vapor_density_2d_map()
        print(np.shape(water_vapor_density_maps))
        
        ### Compute the associated temperature maps from the wapor density maps, using the equation 12 from Morris 2021
        temp_maps = self.integrated_abs_spectrum[:, np.newaxis] * self.temperature * water_vapor_density_maps
        print(np.shape(temp_maps))
        
        ### Convert them into micro Kelvin CMB
        temp_maps -= Planck18.Tcmb0.value
        temp_maps *= 1e6
        
        return temp_maps
    
    def horizontal_plane_to_azel(self, x, y, z):
        """Horizontal plane to azimuth and elevation.
        
        Convert the coordinates in the horizontal plane to azimuth and elevation.

        Parameters
        ----------
        x : array_like
            x coordinate.
        y : array_like
            y coordinate.
        z : array_like
            z coordinate.

        Returns
        -------
        az : array_like
            Azimuth coordinate.
        el : array_like
            Elevation coordinate.
            
        """        
        
        r = np.sqrt(x**2 + y**2 + z**2)
        el = np.pi/2 - np.arccos(z/r)
        az = np.arctan2(y, x)
        
        return r, az, el
    
    def get_healpy_atm_maps_2d(self):
        """Healpy 2d atmosphere maps.
        
        Function to project the 2d atmosphere maps in cartesian coordinates, and then project them in spherical coordinates using healpy.

        Returns
        -------
        hp_maps_2d : array_like
            2d healpy maps of the atmosphere.
            
        """        
        
        ### Build list of azimuth and elevation coordinates for each point of the atmosphere
        az_list, el_list = [], []
        for ind_x in range(len(self.x_list)):
            for ind_y in range(len(self.y_list)):
                _, az, el = self.horizontal_plane_to_azel(self.x_list[ind_x], self.y_list[ind_y], self.altitude[0])
                az_list.append(az)
                el_list.append(el) 
        azel_coordinates = np.asarray([az_list, el_list]).T
        
        ### Build rotation operator
        rotation_above_qubic = Cartesian2SphericalOperator('azimuth,elevation')(Rotation3dOperator("ZY'", - self.qubic_dict['latitude'], self.qubic_dict['longitude'])(Spherical2CartesianOperator('azimuth,elevation')))
        
        ### Build healpy projection operator
        rotation_azel2hp = Spherical2HealpixOperator(self.params['nside'], 'azimuth,elevation')
        
        ### Fill the healpy maps with the temperature maps using the operators
        hp_maps_index = rotation_azel2hp(rotation_above_qubic(azel_coordinates)).astype(int)
        hp_maps_2d = np.zeros((len(self.frequencies), hp.nside2npix(self.params['nside'])))
        for ifreq in range(len(self.frequencies)):
            hp_maps_2d[ifreq, hp_maps_index] = self.get_maps()[ifreq].flatten()
        
        return hp_maps_2d
