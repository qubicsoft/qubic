import numpy as np
import scipy.constants as c
import healpy as hp
import scipy.special as sp 
from scipy.integrate import quad
import camb.correlations as cc

from CoolProp import CoolProp as CP
from astropy.cosmology import Planck18

from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qsamplings import equ2gal
from qubic.lib.Instrument.Qinstrument import compute_freq

from pyoperators import *
from pysimulators.interfaces.healpy import Spherical2HealpixOperator

#TODO : Adjust rho_0 with PWV
#TODO : Verify conversion into µK_CMB
#TODO : Adjust pressure with density
#TODO : Verify frequency dependence of the atmosphere
#TODO : Verify if the atmosphere is in the same frame as the qubic
#TODO : Add the possibility to have a 3d atmosphere
#TODO : Verify that get_integrated_absorption_spectrum works as expected, i.e. tha same way as in MM pipeline to build maps

class AtmosphereProperties:
    
    def __init__(self, params):
        
        ### Import parameters files
        self.params = params
        self.qubic_dict = self.get_qubic_dict()

        np.random.seed(self.params['seed'])
        
        ### Build atmsopheric coordinates
        # Cartesian coordinates
        if self.params['h_grid'] == 1:
            # 2d model
            self.altitude = (self.params['h_qubic'] + self.params['altitude_atm_2d']) * np.ones(self.params['h_grid'])
        else:
            # 3d model, not yet implemented
            self.altitude = np.linspace(self.params['h_qubic'], self.params['altitude_atm_2d'], self.params['h_grid'])
        # Importe atmosphere coordinates for the flat atmosphere case
        if self.params['flat']:
            self.x_list = np.linspace(-self.params['size_atm'], self.params['size_atm'], self.params['n_grid'])
            self.y_list = np.linspace(-self.params['size_atm'], self.params['size_atm'], self.params['n_grid'])
            # Azimuth / Elevation coordinates
            x, y = np.meshgrid(self.x_list, self.y_list)
            z = np.ones(x.shape) * self.params['altitude_atm_2d']
            self.r, self.el, self.az = self.horizontal_plane_to_azel(x, y, z)
            
        ### Compute atmosphere temperature and mean water vapor density
        self.temperature = self.get_temperature_atm(self.altitude, self.params['temp_ground'], self.params['h_temp'])
        self.mean_water_vapor_density = self.get_mean_water_vapor_density(self.altitude, self.params['rho_0'], self.params['h_h2o'])
        
        ### Compute absorption coefficients
        self.mol_absorption_coeff, self.self_absorption_coeff, self.air_absorption_coeff, self.integration_frequencies = self.atm_absorption_coeff()
                
        ### Compute absorption spectrum
        self.abs_spectrum = self.absorption_spectrum()
        self.integrated_abs_spectrum, self.frequencies = self.integrated_absorption_spectrum()
        
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
            "dtheta": self.params["dtheta"],
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
            "synthbeam_kmax": self.params['kmax'],
            "synthbeam_fraction": self.params['synthbeam_fraction'],
            "kind":'IQU'
        }

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        dict_qubic = qubicDict()
        dict_qubic.read_from_file(dictfilename)

        ### Modify the dictionary
        for i in args.keys():
            dict_qubic[str(i)] = args[i]

        return dict_qubic
    
    def get_mean_water_vapor_density(self, altitude, rho_0, h_h2o):
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
        
        return rho_0 * np.exp(-np.log(2) * (altitude - 5190) / h_h2o)
    
    def get_temperature_atm(self, altitude, temp_ground, h_T):
        """Temperature.
        
        Compute the temperature of the atmosphere depending on the altitude, using the average ground temperature and a typical height that depend on the observation site.
        The corresponding equation to compute the temperature, taken from equation (13) in Morris 2021, is :
        
        .. math::
            T_{atm}(h) = T_{atm}(0) e^{h / h_T} .

        Parameters
        ----------
        altitude : array_like
            Array containing altitudes at which we want to compute the temperature.
        temp_ground : float
            Average ground temperature in K.
        h_T : float
            Typical height in m.

        Returns
        -------
        temperature : array_like
            Temperature in K.
            
        """
        
        return temp_ground * np.exp(- altitude / h_T)
        
    def atm_absorption_coeff(self):
        r"""Absorption coefficients.
        
        Method to build the absorption spectrum of the atmopshere using files computed using the am atmospheric model (Paine, 2018).
        The absorption coefficient has two origins: the line-by-line absorption which reprensents the spectral lines, 
        and a continuum absorption coming from collisions between molecules, 
        either :math:`H_2O-H_2O` collisions (self-induced continuum), or collisions between :math:`H_2O` and the air (air_induced).

        See the am documentation for more details: https://lweb.cfa.harvard.edu/~spaine/am/ .

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
        
        # Spectras for the line-by-line absorption-coefficient were produced using .amc files that looks like that:

        # f 130 GHz  250 GHz  0.005 GHz
        # output f  k

        # layer
        # P 500 hPa
        # T 280 K
        # column h2o_lines 1 mm_pwv

        # And for the continuum absorption:

        # f 130 GHz  250 GHz  0.005 GHz
        # output f  k

        # layer
        # P 550 hPa
        # T 280 K
        # h 3000 m
        # column h2o_self_continuum 1 mm_pwv

        # Then we only need to execute a command line like: am file_name.amc > file_name.out .

        # The file 'file_name.out' is created and contains two colomns, one for the frequency, and one for the absorption coefficient. 
        # Be careful, am results are in cm and not in m.
        
        frequencies = []
        mol_absorption_coeff = []
        self_absorption_coeff = []
        air_absorption_coeff = []
        
        ### Initialize atmosppheric properties
        pressure = self.params['pressure_atm']
        temp = self.params['temp_ground']
        pwv = self.params['pwv']
        
        ### Import absorption coefficients from molecular absorption lines
        with open(self.params["path_am_files"] + f'h2o_lines_{pressure}hPa_{temp}K_{pwv}mm.out', 'r') as file:
            for line in file:
                frequencies.append(float(line.split()[0]))
                mol_absorption_coeff.append(float(line.split()[1]) * (1e-2)**2)
                
        ### Import absorption coefficients from self-induced collisions continuum
        with open(self.params["path_am_files"] + f'h2o_self_continuum.out', 'r') as file:
            for line in file:
                self_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        ### Import absorption coefficients from air-induced collisions continuum
        with open(self.params["path_am_files"] + f'h2o_air_continuum.out', 'r') as file:
            for line in file:
                air_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        return np.array(mol_absorption_coeff), np.array(self_absorption_coeff), np.array(air_absorption_coeff), frequencies
    
    def get_gas_properties(self, params_file = True):
        r"""Gas properties.
        
        Method to compute the properties of the water vapor and the air in the atmosphere. It uses the CoolProp package (https://github.com/CoolProp/CoolProp) to compute the molar mass of air and water.
        It can use parameters given in self.params.yml or be baised on the CoolProp package to compute the water vapor density.
        
        The pressure as to be defined in the params.yml file. The temperature is computed using the function 'get_temp_atm'.
        
        The molar mass of air is given by the CoolProp package: CP.PropsSI('MOLARMASS', 'Air') kg/mol.
        The molar mass of water is given by the CoolProp package: CP.PropsSI('MOLARMASS', 'Water') kg/mol.
        
        From molar mass, the mass is given by :
        
        .. math::
            m = M / N_A ,
        where :math:`m` is the mass, :math:`M` is the molar mass and :math:`N_A` is the Avogadro constant, given by scipy.constants .
        
        The mass density of air is computed using CoolProp package, from the  pressure and the temperature of the atmosphere.
        Then, the density of air is computed with :
        
        .. math::
            n = \rho / m ,
        where :math:`n` is the density, :math:`\rho` is the mass density and :math:`m` is the mass.

        For the water vapor density, you can follow the same steps by putting the argument params_file = False, 
        or compute it using the mass density computed using 'get_mean_water_vapor_density' and the parameters given in self.params_file.

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
        
        """        
        
        ### Import physical constants
        temp_atm = self.temperature  # in K
        pressure_atm = self.params['pressure_atm'] * 100  # in Pa

        ### Air properties
        air_molar_mass = CP.PropsSI('MOLARMASS', 'Air')  # in kg/mol
        air_mass = air_molar_mass / c.Avogadro * 1e3  # in g
        air_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Air")  # in kg/m-3
        air_density = air_mass_density * 1e3 / air_mass  # in m-3

        ### Water properties
        water_molar_mass = CP.PropsSI('MOLARMASS', 'Water')  # in kg/mol
        water_mass = water_molar_mass / c.Avogadro * 1e3  # in g
        
        ### Compute water vapor density
        if params_file:
            water_vapor_density = self.mean_water_vapor_density / water_mass  # in m-3
        else:
            water_vapor_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Water")  # in kg/m-3
            water_vapor_density = c.Avogadro * water_vapor_mass_density / water_molar_mass * 1e-3  # in m-3

        return water_mass, water_vapor_density, air_density
        
    def absorption_spectrum(self):
        r"""Absorption spectrum.
        
        The coefficient :math:`\alpha_b(\nu)` (:math:`m^2/g`) is defined by:
        
        .. math::
            \alpha_b(\nu) = \frac{1}{m_{H_2O}} \left(k_{lines}(\nu) + n_{H_2O}k_{self}(\nu) + n_{air}k_{air}(\nu)\right) ,
            
        with :math:`m_{H_2O}= 2.992\times 10^{-23} g` the mass of a :math:`H_2O` molecule, :math:`k_{lines}` (:math:`m^2`) the line-by-line absorption coefficient, 
        :math`k_{self}` and :math`k_{air}` (:math:`m^5`) the self- and air-induced continua, :math:`n_{H_2O}` and :math:`n_{air}` (:math:`m^{-3}`) the densities of water vapor and air.

        Returns
        -------
        abs_spectrum : array_like
            Atmosphere absorption coefficient, in :math:`m^{2} / g`.
            
        """        

        ### Import gas properties
        water_mass, water_vapor_density, air_density = self.get_gas_properties()
                
        ### Compute coeff
        abs_spectrum = (self.mol_absorption_coeff + water_vapor_density * self.self_absorption_coeff + air_density * self.air_absorption_coeff) / water_mass 

        return abs_spectrum
    
    def get_integrated_absorption_spectrum(self, band):
        """Integrated absorption spectrum within a band.
        
        Compute the integrated absorption spectrum in a given frequency band, according to the parameters in the self.params.yml file.

        Parameters
        ----------
        band : int
            QUBIC frequency band. Can be either 150 or 220.

        Returns
        -------
        integrated_abs_spectrum : numpy.ndarray
            The integrated absorption spectrum in the given frequency band, in m^2/g.
        nus : numpy.ndarray
            The frequencies at which the absorption spectrum is computed, in GHz.
            
        """
        #! Verify if the integration is made properly !!!
        
        ### Verify the given band
        if band not in [150, 220]:
            raise ValueError("Band must be either 150 or 220 GHz.")

        ### Evaluate the frequency band edges
        freq_min, freq_max = self.integration_frequencies[0], self.integration_frequencies[-1]
        freq_step = (freq_max - freq_min) / (len(self.integration_frequencies) - 1)

        ### Compute the frequency sub-bands within the QUBIC band and their associated indexes
        _, nus_edges, nus, _, _, N_bands = compute_freq(
            band=band, 
            Nfreq=int(self.params['nsub_in']/2), 
            relative_bandwidth=self.qubic_dict['filter_relative_bandwidth']
        )
        nus_edge_index = np.round((nus_edges - freq_min) / freq_step).astype(int)

        ### Integrate the absorption spectrum over the frequency sub-bands using the trapezoidal method
        integrated_abs_spectrum = np.array([
            np.trapz(
                self.abs_spectrum[nus_edge_index[i]:nus_edge_index[i+1]], 
                x=self.integration_frequencies[nus_edge_index[i]:nus_edge_index[i+1]]
            )
            for i in range(N_bands)
        ])
        
        return integrated_abs_spectrum, nus
    
    def integrated_absorption_spectrum(self):
        """Integrated absorption spectrum.
        
        Integrated absorption spectrum in the QUBIC frequency bands: 150 and 220 GHz.

        Returns
        -------
        integrated_abs_spectrum : array_like
            The integrated absorption spectrum in the given frequency band, in :math:`m^{2} / g`.
        nus : array_like
            The frequencies at which the absorption spectrum is computed, in :math:`GHz`.
            
        """        
        
        ### Get the integrated absorption spectrum in the two QUBIC bands : 150 and 220 GHz
        int_abs_spectrum_150, nus_150 = self.get_integrated_absorption_spectrum(band=150)
        int_abs_spectrum_220, nus_220 = self.get_integrated_absorption_spectrum(band=220)
        
        return np.append(int_abs_spectrum_150, int_abs_spectrum_220), np.append(nus_150, nus_220)
    
class AtmosphereMaps(AtmosphereProperties):
    
    def __init__(self, params):
        
        ### Import parameters and the class describing the atmosphere
        self.params = params
        AtmosphereProperties.__init__(self, params)
        
        ### Compute the maximum multipole according to the resolution of the map
        self.lmax = 3*self.params['nside']-1
        
        ### Build water vapor density map
        self.rho_map = self.get_water_vapor_density_2d_map(self.mean_water_vapor_density, flat=self.params['flat'])
        
        ### Build the temperature maps of the atmosphere
        self.atm_temp_maps = self.get_temp_maps(self.rho_map)
        
    def get_fourier_grid_2d(self, n_grid, size_atm):
        """Fourier 2d grid.
        
        Generate a 2d grid of spatial frequencies in Fourier space according to the parameters in the self.params.yml file.

        Parameters
        ----------
        n_grid : int
            Number of grid points in the 2d grid.
        size_atm : float
            Size of the atmosphere in m.
        
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
        k_distrib_y = np.fft.fftfreq(n_grid, d=2*size_atm/n_grid) * 2*np.pi
        k_distrib_x = np.fft.fftfreq(n_grid, d=2*size_atm/n_grid) * 2*np.pi
        
        ### Build 2d grid and compute the norm of the spatial frequencies
        kx, ky = np.meshgrid(k_distrib_x, k_distrib_y)
        k_norm = np.sqrt(kx**2 + ky**2)
        
        return kx, ky, k_norm
    
    def kolmogorov_spectrum_2d(self, k, r0):
        r"""Kolmogorov 2d spectrum.
        
        Compute the Kolmogorov 2d spectrum, which simulate the power spectrum of the spatial fluctuations of the water vapor density, following the equation :
        
        .. math::
            P(\textbf{k}) = (r_0^{-2} + \lvert \textbf{k} \rvert ^{2})^{-8/6} .
        

        Parameters
        ----------
        k : array_like
            Array containing the spatial frequencies at which we want to compute the Kolmogorov 2d spectrum.
        r0 : float
            Maximum spatial coherence length of the water vapor density, in m.

        Returns
        -------
        kolmogorov_spectrum_2d : array_like
            Kolmogorov 2d spectrum.
        """        
        
        return (r0**(-2) + np.abs(k)**2)**(-8/6)
    
    def normalized_kolmogorov_spectrum_2d(self, k, r0):
        r"""Normalized Kolmogorov 2d spectrum.
        
        Compute the normalized Kolmogorov 2d spectrum, to ensure :
        
        .. math::
            \int_\textbf{k} P(\textbf{k}) d\textbf{k} = 1 .

        Parameters
        ----------
        k : array_like
            Array containing the spatial frequencies at which we want to compute the normalized Kolmogorov 2d spectrum.
        r0 : float
            Maximum spatial coherence length of the water vapor density, in m.

        Returns
        -------
        normalized_kolmogorov_spectrum_2d : array_like
            Normalized Kolmogorov 2d spectrum.
            
        """        
        
        ### Compute the normalization constant
        res, _ = quad(self.kolmogorov_spectrum_2d, np.min(k), np.max(k), args=(r0))
        
        return self.kolmogorov_spectrum_2d(k, r0) / res
    
    def generate_spatial_fluctuations_fourier(self, n_grid, size_atm, r0):
        """Spatial 2d fluctuations.
        
        Produce the spatial fluctuations of the water vapor density, by generating random phases in Fourier space, and then computing the inverse Fourier transform.

        Parameters
        ----------
        n_grid : int
            Number of grid points in the 2d grid.
        size_atm : float
            Size of the atmosphere in m.
        r0 : float
            Maximum spatial coherence length of the water vapor density, in m

        Returns
        -------
        delta_rho_x : array_like
            Variation of the water vapor density.
            
        """        
        #! At some point, we will need to normalize these fluctuations using real data. We can maybe use :math:`\sigma_{PWV}` that can be estimated with figure 4 in Morris 2021.
        
        ### Compute the spatial frequencies & power spectrum.
        _, _, k = self.get_fourier_grid_2d(n_grid, size_atm)
        kolmogorov_spectrum = self.normalized_kolmogorov_spectrum_2d(k, r0)

        ### Generate spatial fluctuations through random phases in Fourier space
        phi = np.random.uniform(0, 2*np.pi, size=(self.params['n_grid'], self.params['n_grid']))
        delta_rho_k = np.sqrt(kolmogorov_spectrum) * np.exp(1j * phi)

        ### Apply inverse Fourier transform to obtain spatial fluctuations in real space
        delta_rho = np.fft.ifft2(delta_rho_k, s=(self.params['n_grid'], self.params['n_grid'])).real
        
        return delta_rho  
    
    def kolmogorov_correlation_function(self, r, r0):
        r"""Kolmogorov correlation function.
        
        Compute the Kolmogorov correlation function by applying the inverse Fourier transform to the Kolmogorov 2d spectrum.
        This correlation function can be written as:

        .. math::
            D(r) = \frac{2^{2/3}}{\Gamma(1/3)} \left(\frac{r}{r_0}\right)^{1/3} K_{1/3} \left(\frac{r}{r_0}\right) .
            
        We impose that the correlation function is 1 at r = 0.

        Parameters
        ----------
        r : array_like or float
            Distance between two points, in meters.
        r0 : array_like or float
            Maximum correlation length, in meters.

        Returns
        -------
        D : array_like
            Correlation function.
            
        """   
             
        return np.where(r==0, 1, 2**(2/3)/sp.gamma(1/3)*(r/r0)**(1/3)*sp.kv(1/3, r/r0))    
    
    def angular_correlation(self, theta, h_atm, r0):
        r"""Angular Kolmogorov correlation function.
        
        We compute the angular Kolmogorov correlation function, switching the distance between two points to the angle between them on the surface of the sphere, 
        using the relation :
        
        .. math::
            r = 2h_{atm} \sin \left(\frac{\theta}{2}\right) ,
            
        where :math:`h_{atm}` is the distance between the atmosphere and our instrument and :math:`\theta` is the angle between the two points.
        

        Parameters
        ----------
        theta : array_like or float
            Angle between two points, in degrees.
        h_atm : array_like or float
            Distance between the atmosphere and our instrument, in meters.
        r0 : float
            Maximum correlation length, in meters.

        Returns
        -------
        C : array_like or float
            Angular Kolmogorov correlation function.
            
        """        
        
        ### Compute the distance between two points separeted by the angle theta on the surface of the sphere
        r = 2*h_atm*np.sin(np.radians(theta)/2)
        
        return self.kolmogorov_correlation_function(r, r0)
    
    def cl_from_angular_correlation_int(self, l):
        r"""Angular power spectrum from angular correlation function.
        
        Compute the angular power spectrum from the angular correlation function, using the formula:

        .. math::
            C_{\ell} = 2 \pi \int_{-1}^{1} C(\theta) P_{\ell(\cos \theta) d \cos \theta ,

        where :math:`C(\theta)` is the angular correlation function and :math:`P_l(\cos \theta)` is the Legendre polynomial of order :math:`l`.
        
        WARNING: This function is not very efficient, as it computes the Legendre polynomial of order l for each value of cos(theta). It takes a long time to compute for high values of l.
        It is just used for testing purposes and to compute the case \ell = 0 which is not possible in 'ctheta_2_dell'.

        Parameters
        ----------
        l : int
            Angular multipole order.

        Returns
        -------
        C_l : float
            Angular power spectrum of order l.
            
        """         
        
        #! It should be possible to speed the computation by using the function np.polynomial.legendre.leggauss
        
        ###Compute the integrand for the integral
        def integrand(cos_theta):
            # Compute theta from cos(theta)
            theta = np.degrees(np.arccos(cos_theta))
            
            # Compute the legendre polynomial of order l
            legendre = sp.legendre(l)(cos_theta)
            
            # Return the product of the angular correlation function and the legendre polynomial
            return self.angular_correlation(theta, self.params['altitude_atm_2d'], self.params['correlation_length']) * legendre
        
        ### Integrate over cos(theta)
        res, _ = quad(integrand, -1, 1)
        
        return 2 * np.pi * res
    
    def ctheta_2_dell(self, theta_deg, ctheta, lmax, normalization=1):
        r"""Angular power spectrum from angular correlation function.
        
        Compute the angular power spectrum from the angular correlation function, using the formula:
        
        .. math::
            C_{\ell} = 2 \pi \int_{-1}^{1} C(\theta) P_{\ell(\cos \theta) d \cos \theta ,
        where :math:`C(\theta)` is the angular correlation function and :math:`P_l(\cos \theta)` is the Legendre polynomial of order :math:`l`.
        
        This computation is done using the CAMB library, which is much faster than the 'cl_from_angular_correlation_int' function.

        Parameters
        ----------
        theta_deg : array_like
            Angles between two points on the surface of the sphere, in degrees.
        ctheta : array_like
            Angular correlation function.
        lmax : int
            Maximum angular multipole order.
        normalization : int, optional
            Normalization parameter, by default 1

        Returns
        -------
        dell : array_like
            Angular power spectrum.
            
        """         
        
        ### Compute the Legendre polynomials up to lmax+1, and the theta values
        ### These cos_theta do not contain cos(theta)=1 so we have to do this case separately
        cos_theta, legendre = np.polynomial.legendre.leggauss(lmax+1)
        xdeg = np.degrees(np.arccos(cos_theta))

        ### Replace C(theta=0) by 0
        myctheta = ctheta.copy()
        myctheta[0] = 0
        
        ### Fill the array that should include polarization (we put zeros there) with the values of our imput c(theta) interpolated at the cos_theta locations
        allctheta = np.zeros((len(cos_theta), 4))
        allctheta[:,0] = np.interp(xdeg, theta_deg, myctheta)

        ### Call the camb function that does the transform from C(theta) to Dl
        dlth = cc.corr2cl(allctheta, cos_theta,  legendre, lmax)
        
        ### Compute the multipole moments
        ell = np.arange(lmax+1)

        ### the special case cos(theta)=1 corresponds to theta=0 and add 2pi times c(theta=0) to the Dl
        return ell, dlth[:,0]+ctheta[0]*normalization
    
    def ctheta_2_cell(self, theta_deg, ctheta, lmax, normalization=1):
        r"""Angular power spectrum from angular correlation function.
        
        Compute the angular power spectrum from the angular correlation function, using the function 'ctheta_2_dell',
        and convert the result to :math:`C_{\ell}`.

        Parameters
        ----------
        theta_deg : array_like
            Angles between two points on the surface of the sphere, in degrees.
        ctheta : array_like
            Angular correlation function.
        lmax : int
            Maximum angular multipole order.
        normalization : int, optional
            Normalization parameter, by default 1

        Returns
        -------
        cell : array_like
            Angular power spectrum.
        """        
        
        #! Warning : the Cl computed using CAMB are different from the ones computed using 'cl_from_angular_correlation_int' at large l
        
        ### Compute multipole moments and Dl angular power spectrum
        ell, dlth = self.ctheta_2_dell(theta_deg, ctheta, lmax, normalization=normalization)
        
        ### Convert from Dl to Cl
        dl2cl_factor = 2*np.pi / (ell * (ell+1))
        clth = dlth * dl2cl_factor
        
        ### Correct for the special case l=0, as the convertion factor is not valid
        clth[0] = self.cl_from_angular_correlation_int(0)
        
        return ell, clth
    
    def generate_spatial_fluctuation_sphercial_harmonics(self):
        """Spatial fluctuation map from angular correlation function.
        
        Compute the spatial fluctuation HEALPix map from the angular correlation function, using the function 'ctheta_2_cell'.

        Returns
        -------
        delta : array_like
            Spatial fluctuation map, generated accorging HEALPix formalism.
        """        
            
        ### Compute angular correlation function
        theta = np.linspace(0, 180, self.params['n_theta'])
        ctheta = self.angular_correlation(theta, self.params['altitude_atm_2d'], self.params['correlation_length'])
        
        ### Compute spherical harmonics from angular correlation function
        _, clth = self.ctheta_2_cell(theta, ctheta, self.lmax, normalization=self.params['normalization'])
        
        ### Build fluctuations map
        delta_rho = hp.synfast(clth, nside=self.params['nside'], lmax=self.lmax)
        
        return delta_rho
        
    def get_water_vapor_density_2d_map(self, mean_rho, flat=True):
        """Water vapor density 2d map.
        
        Get the water vapor density 2d map with simulated fluctuations.
        The spatial fluctuations are generated using either the correlation function or the angular correlation function.

        Parameters
        ----------
        mean_rho : array_like
            Mean water vapor density.
        angular : bool, optional
            If True, use the angular correlation function. If False, use the correlation function., by default True

        Returns
        -------
        atm_maps_2d : array_like
            Water vapor density 2d map.
            
        """             
        
        ### Build water vapor density fluctuations
        if flat:
            delta_rho = self.generate_spatial_fluctuations_fourier(self.params['n_grid'], self.params['size_atm'], self.params['correlation_length'])
        else:
            delta_rho = self.generate_spatial_fluctuation_sphercial_harmonics()
            
        ### Normalize fluctuations
        normalized_delta_rho = delta_rho * np.sqrt(self.params['sigma_rho'] / np.var(delta_rho))  

        return mean_rho + normalized_delta_rho
        
    def get_temp_maps(self, maps):
        r"""Atmosphere maps.
        
        Get the atmosphere maps in temperature, by using the equation 12 from Morris 2021, that compute the induced temperature in the detector due to the water vapor density :
        
        .. math::
            dT( \textbf{r}, \nu) = \alpha_b(\nu) \rho(\textbf{r}) T_{atm}(\textbf{r}) dV .
            
        And then, convert it in micro Kelvin CMB.
        
        Parameters
        ----------
        maps : array_like
            Water vapor density 2d map.

        Returns
        -------
        temp_maps : array_like
            Temperature maps in micro Kelvin CMB.
            
        """
        
        ### Compute the associated temperature maps from the wapor density maps, using the equation 12 from Morris 2021
        ###! I assume that the multiplication by the beam profile is done when applying the acquisition operator.
        if len(maps.shape) == 1:
            temp_maps = self.integrated_abs_spectrum[:, np.newaxis] * self.temperature * maps
        else:
            temp_maps = self.integrated_abs_spectrum[:, np.newaxis, np.newaxis] * self.temperature * maps
            
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
    
    def get_azel_coordinates(self):
        
        az_list, el_list = [], []
        for ind_x in range(len(self.x_list)):
            for ind_y in range(len(self.y_list)):
                _, az, el = self.horizontal_plane_to_azel(self.x_list[ind_x], self.y_list[ind_y], self.altitude[0])
                az_list.append(az)
                el_list.append(el) 
        return np.asarray([az_list, el_list]).T
    
    def get_healpy_atm_maps_2d(self, maps, longitude, latitude):
        """Healpy 2d atmosphere maps.
        
        Function to project the 2d atmosphere maps in cartesian coordinates, and then project them in spherical coordinates using healpy.
        By default, the projection is centered on the QUBIC patch (RA=0, DEC=-57).

        Returns
        -------
        hp_maps_2d : array_like
            2d healpy maps of the atmosphere.
            
        """        
        
        ### Build list of azimuth and elevation coordinates for each point of the atmosphere
        azel_coordinates = self.get_azel_coordinates()
        
        ### Build rotation operator
    
        rotation_above_qubic = Cartesian2SphericalOperator('azimuth,elevation')(Rotation3dOperator("ZY'", longitude, 90 - latitude, degrees=True)(Spherical2CartesianOperator('azimuth,elevation')))
        
        ### Build healpy projection operator
        rotation_azel2hp = Spherical2HealpixOperator(self.params['nside'], 'azimuth,elevation')
        
        ### Fill the healpy maps with the temperature maps using the operators
        hp_maps_index = rotation_azel2hp(rotation_above_qubic(azel_coordinates)).astype(int)
        hp_maps_2d = np.zeros((len(self.frequencies), hp.nside2npix(self.params['nside'])))
        for ifreq in range(len(self.frequencies)):
            hp_maps_2d[ifreq, hp_maps_index] = maps[ifreq].flatten()
        
        return hp_maps_2d
