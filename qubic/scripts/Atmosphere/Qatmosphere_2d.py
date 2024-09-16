import numpy as np

import scipy.constants as c
from scipy.integrate import quad
import CoolProp.CoolProp as CP

import qubic

class Atmsophere:
    
    def __init__(self, params):
        
        self.params = params
        self.qubic_dict = qubic.lib.Qdictionary.qubicDict()
        self.qubic_dict['filter_relative_bandwidth'] = 0.25
        
        if self.params['h_grid'] == 1:
            # 2d model
            self.altitude = self.params['altitude_atm_2d']
        else:
            # 3d model not yet implemented
            self.altitude = None
            
        self.temperature = self.temperature_atm(self.altitude)
        self.mean_water_vapor_density = self.mean_water_vapor_density(self.altitude)
        
        self.frequencies, self.mol_absorption_coeff, self.self_absorption_coeff, self.air_absorption_coeff = self.atm_absorption_coeff()
        
        self.abs_spectrum = self.absorption_spectrum()
        self.integrated_abs_spectrum = self.integrated_absorption_spectrum()
        
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
        
        pressure = self.params['pressure_atm']
        temp = self.params['temp_ground']
        pwv = self.params['pwv']
        
        with open(f'absorption_coefficient/h2o_lines_{pressure}hPa_{temp}K_{pwv}mm.out', 'r') as file:
            for line in file:
                frequencies.append(float(line.split()[0]))
                mol_absorption_coeff.append(float(line.split()[1]) * (1e-2)**2)
                
        with open(f'/home/laclavere/Documents/Thesis/qubic/qubic/scripts/Atmosphere/absorption_coefficient/h2o_self_continuum.out', 'r') as file:
            for line in file:
                self_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        with open(f'/home/laclavere/Documents/Thesis/qubic/qubic/scripts/Atmosphere/absorption_coefficient/h2o_air_continuum.out', 'r') as file:
            for line in file:
                air_absorption_coeff.append(float(line.split()[1]) * (1e-2)**5)
                
        return frequencies, np.array(mol_absorption_coeff), np.array(self_absorption_coeff), np.array(air_absorption_coeff)
    
    def get_gas_properties(self, params_file = True):
        r"""Gas properties.
        
        Method to compute the properties of the water vapor and the air in the atmosphere.
        It can use parameters given in params.yml or be baised on the CoolProp package to compute the water vapor density.

        Parameters
        ----------
        params_file : bool, optional
            If True, will use the reference water vapor density given in params_file by self.params['rho_0'] (in :math:`g/m^{3}`) to compute the density in :math:`m^{-3}`.
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
        
        # Import physical constants
        temp_atm = self.temperature                                                             # in K
        pressure_atm = self.params['pressure_atm'] * 100                                            # in Pa
        
        # Air properties
        air_molar_mass = CP.PropsSI('MOLARMASS', 'Air')                                             # in g/mol
        air_mass = air_molar_mass / c.Avogadro * 10**3                                              # in g
        air_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Air")                 # in kg/m-3
        air_density = air_mass_density * 1000 / air_mass                                            # in m-3
        
        # Water properties
        water_molar_mass = CP.PropsSI('MOLARMASS', 'Water')                                         # in g/mol
        water_mass = water_molar_mass / c.Avogadro * 10**3                                          # in g
        if params_file:
            water_vapor_density = self.mean_water_vapor_density / water_mass                                 # in m-3
        else:
            water_vapor_mass_density = CP.PropsSI("D", "T", temp_atm, "P", pressure_atm, "Water")   # in kg/m-3
            water_vapor_density = c.Avogadro * water_vapor_mass_density / water_molar_mass / 1000   # in m-3

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
    
    def integrated_absorption_spectrum(self, band = 150):
        
        freq_max = np.max(self.frequencies)
        freq_min = np.min(self.frequencies)
        freq_step = (freq_max - freq_min) / (len(self.frequencies) - 1)

        _, nus_edges, nus, _, _, N_bands = qubic.lib.Qinstrument.compute_freq(band=band, Nfreq=self.params['nsub_in'], relative_bandwidth=self.qubic_dict['filter_relative_bandwidth'])
        nus_edge_index = (nus_edges - freq_min) / freq_step
        
        integrated_abs_spectrum = []
        for i in range(N_bands):
            index_inf = int(nus_edge_index[i])
            index_sup = int(nus_edge_index[i+1])
            integrated_abs_spectrum.append(np.sum(self.abs_spectrum[index_inf:index_sup]) * 
                                    (self.frequencies[index_sup] - self.frequencies[index_inf])*1e9 / (index_sup - index_inf))
            
        return integrated_abs_spectrum, nus
    
    def mean_water_vapor_density(self, altitude):
        r"""Mean water vapor density.
        
        Compute the mean water vapor density depending on the altitude, using reference water vapor density and water vapor half_height, given in params.yml.
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
    
    def temperature_atm(self, altitude):
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
        
        # Generate spatial frequency in Fourier space
        k_distrib_y = np.fft.fftfreq(self.params['n_grid'], d=2*self.params['size_atm']/self.params['n_grid']) * 2*np.pi
        k_distrib_x = np.fft.fftfreq(self.params['n_grid'], d=2*self.params['size_atm']/self.params['n_grid']) * 2*np.pi
        
        # Build 2d grid
        kx, ky = np.meshgrid(k_distrib_x, k_distrib_y)
        k_norm = np.sqrt(kx**2 + ky**2)
        
        return kx, ky, k_norm
    
    def kolmogorov_spectrum_2d(self, k):
        
        return (self.params['correlation_length']**(-2) + np.abs(k)**2)**(-8/6)
    
    def normalized_kolmogorov_spectrum_2d(self, k):
        
        res, _ = quad(self.kolmogorov_spectrum_2d, np.min(k), np.max(k))
        
        return self.kolmogorov_spectrum_2d(k) / res
    
    def generate_spatial_fluctuations_2d(self):
        #! At some point, we will need to normalize these fluctuations using real data. We can maybe use :math:`\sigma_{PWV}` that can be estimated with figure 4 in Morris 2021.
        
        # Compute the power spectrum
        k = self.get_fourier_grid_2d()[2]
        kolmogorov_spectrum = self.normalized_kolmogorov_spectrum_2d(k)
        
        # Generate spatial fluctuations through random phases in Fourier space
        phi = np.random.uniform(0, 2*np.pi, size=(self.params['n_grid'], self.params['n_grid']))
        delta_rho_k = np.sqrt(kolmogorov_spectrum) * np.exp(1j * phi)

        # Apply inverse Fourier transform to obtain spatial fluctuations
        delta_rho = np.fft.ifft2(delta_rho_k).real
        
        return delta_rho        
        
    def get_water_vapor_density_map(self):
        
        return None