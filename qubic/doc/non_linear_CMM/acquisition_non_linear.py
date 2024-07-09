import qubic
import numpy as np
import matplotlib.pyplot as plt
from pyoperators import *
import healpy as hp
import pysm3
import pysm3.units as u

from Qacquisition import QubicFullBandSystematic, OtherDataParametric


class NonLinearAcquisition:
    def __init__(self, parameters_dict):
        '''
        nside: (int, power of 2) nside of the pixels in the sky from the module healpy.
        nside_beta: (int, power of 2) Same but for the spectral indices of the dust. If nside_beta = 0, reconstructs only on spectral index.
        npointings: (int) Number of random pointings for the acquisition matrix.
        Nsub: (int) Number of frequencies considered in the full band.
        dust_level: (float) Level of the dust. Multiplies the input map of the dust by this factor.
        dust_model: (str) Model of the dust: d0, d1 or d6.
        dust_reconstruction: (bool) If we reconstruct the dust.
        synchrotron_level: (float) Level of the synchrotron. Multiplies the input map of the synchrotron by this factor.
        synchrotron_model: (str) Model of the synchrotron: s0 or s1.
        synchrotron_reconstruction: (bool) If we reconstruct the synchrotron.
        spectrum_modelization: (str) Parametric or blind. Blind not implemented yet.
        frequencies_planck: (array) Planck frequencies in Hz.
        noise_qubic: (float) Level of normal noise in Qubic's TOD.
        noise_planck: (float) Level of normal noise in Planck's TOD.
        planck_coverage_level: (float, between 0 and 1) Relative coverage under which Planck's maps are added to help convergence.
        '''
        # Initializing variables
        self.nside = parameters_dict['nside']
        self.nside_beta = parameters_dict['nside_beta']
        self.npointings = parameters_dict['npointings']
        self.Nsub = parameters_dict['Nsub']
        self.dust_level = parameters_dict['dust_level']
        self.dust_model = parameters_dict['dust_model']
        self.dust_reconstruction = parameters_dict['dust_reconstruction']
        self.synchrotron_level = parameters_dict['synchrotron_level']
        self.synchrotron_model = parameters_dict['synchrotron_model']
        self.synchrotron_reconstruction = parameters_dict['synchrotron_reconstruction']
        #self.spectrum_modelization = parameters_dict['spectrum_modelization']
        self.frequencies_planck = parameters_dict['frequencies_planck']
        self.noise_qubic = parameters_dict['noise_qubic']
        self.noise_planck = parameters_dict['noise_planck']
        self.planck_coverage_level = parameters_dict['planck_coverage_level']

        self.npixel = 12*self.nside**2 # Number of pixels in the sky
        if self.nside_beta:
            self.nbeta = 12*self.nside_beta**2 # Number of spectral indices in the sky
        else:
            self.nbeta = 1
        self.ncomponent = int(self.dust_reconstruction) + int(self.synchrotron_reconstruction) # number of reconstructed components

        _, frequencies150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.Nsub/2)-1, relative_bandwidth=0.25)
        _, frequencies220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.Nsub/2)-1, relative_bandwidth=0.25)
        self.frequencies_qubic = np.concatenate((frequencies150, frequencies220)) * 1e9 # Frequencies of Qubic in Hz
        self.nu0_dust = 353e9 # Reference frequency for the dust
        self.nu0_synchrotron = 150e9 # Reference frequency for the synchrotron #####################################################################

        
        # Testing that variables or well defined
        if (self.nside & (self.nside - 1) != 0) or self.nside <= 0:
            raise Exception("nside should be a power of 2")
        if self.nside_beta > self.nside or (self.nside_beta & (self.nside_beta - 1) != 0) or self.nside_beta < 0:
            raise Exception("nside_beta should be a power of 2 smaller than nside")
        if self.npointings < 1:
            raise Exception("npointings should be greater than 1")
        if self.Nsub < 2 or self.Nsub%2 != 0:
            raise Exception("Nsub should be even and greater than 2")
        if self.dust_level < 0:
            raise Exception("dust_level should be positive")
        if self.dust_reconstruction and self.dust_level == 0:
            raise Exception("You can't reconstruct dust if the dust_level is 0")
        if not self.dust_model in ['d0', 'd1']:
            raise Exception("dust_model should be d0 or d1")
        if self.synchrotron_level < 0:
            raise Exception("synchrotron_level should be positive")
        if self.synchrotron_reconstruction and self.synchrotron_level == 0:
            raise Exception("You can't reconstruct synchrotron if the synchrotron_level is 0")
        if not self.synchrotron_model in ['s0', 's1']:
            raise Exception("synchrotron_model should be s0 or s1")
        if self.noise_qubic < 0.:
            raise Exception("noise_qubic should be a positive number")
        if self.noise_planck < 0.:
            raise Exception("noise_planck should be a positive number")
        if self.planck_coverage_level < 0. or self.planck_coverage_level > 1.:
            raise Exception("planck_coverage_level should be comprised between 0 and 1")
        

        #Ultra Wide Band configuration.
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
        delta_nu_over_nu = 2*delta/nu_ave

        dictname = 'pipeline_demo.dict'
        self.dict = qubic.qubicdict.qubicDict()
        self.dict.read_from_file(dictname)
        self.dict['nside'] = self.nside
        self.dict['npointings'] = self.npointings
        self.dict['nf_sub'] = self.Nsub
        self.dict['filter_nu'] = nu_ave*1e9
        self.dict['filter_relative_bandwidth'] = delta_nu_over_nu
        self.dict['type_instrument'] = 'wide'
        self.dict['MultiBand'] = True
        self.dict['hwp_stepsize'] = 3
        self.dict['synthbeam_kmax'] = 3
        self.dict['synthbeam_fraction'] = 0.95
        self.dict['random_pointing'] = True
        self.dict['repeat_pointing'] = False
        #self.dict['beam_shape'] = 'fitted_beam'

        self.Q = QubicFullBandSystematic(self.dict, Nsub=self.Nsub, Nrec=2, kind='wide')
        self.H_list = self.Q.H # List of the acquisition matrices of Qubic at the different wavelengths.

        # Determines all the pixels that are seen (even poorly) by Qubic
        self.seenpix_qubic = self.H_list[0].T(np.ones(self.H_list[0].shapeout)) != 0.0
        for i in range(len(self.frequencies_qubic)):
            np.logical_or(self.seenpix_qubic, self.H_list[i].T(np.ones(self.H_list[0].shapeout)) != 0.0, out = self.seenpix_qubic)
        self.seenpix_qubic = self.seenpix_qubic[:, 0] # Boolean mask of the pixels of Qubic's patch

        if self.nside_beta:
            self.seenpix_qubic_beta = hp.ud_grade(self.seenpix_qubic, self.nside_beta) # Boolean mask of the spectral indices of Qubic's patch

        self.npixel_patch = np.count_nonzero(self.seenpix_qubic) # Number of pixels in Qubic's patch
        if self.nside_beta:
            self.nbeta_patch = np.count_nonzero(self.seenpix_qubic_beta) # Number of spectral indices in Qubic's patch
        else:
            self.nbeta_patch = 1
        self.component_map_size = 3 * (1 + self.ncomponent) * self.npixel_patch + self.ncomponent * self.nbeta_patch # length of component_map

        # Estimation of the rank of the acquistion matrix
        rank = 1 - (1 - 1/(self.component_map_size) * 992/5)**self.npointings 
        print(f'You are trying to reconstruct {self.component_map_size} parameters.')
        print(f'The analytically estimated rank of the acquisition matrix is {rank*100} %.')
        if rank < 0.999:
            print('The rank is too low! Increase the number of pointings.')
        else:
            print('Make sure you have enough random pointings.')


    def modified_black_body_dust(self, freq, beta):
        '''
        The modified black-body spectrum of the dust. We have: h/(kT) = 2.4 x 10^(-12) Hz^(-1) at T = 20 K.
        '''
        return (np.exp(freq * 2.4e-12) - 1) / (np.exp(self.nu0_dust * 2.4e-12) - 1) * (freq / self.nu0_dust)**beta


    def power_law_synchrotron(self, freq, beta):
        '''
        The power law spectrum of the synchrotron.
        '''
        return (freq / self.nu0_synchrotron)**beta
    
    
    def get_real_sky(self):
        '''
        The components maps and spectral indices that we will try to reconstruct.
        '''
        real_sky = {}

        # CMB
        sky_cmb = pysm3.Sky(nside=self.nside, preset_strings=['c1'], output_unit='uK_CMB')
        sky_cmb = np.array(sky_cmb.get_emission(self.nu0_dust * u.Hz))
        real_sky['cmb'] = sky_cmb.T.copy() # shape (npixel, 3)

        # Dust
        if self.dust_level:
            sky_dust = pysm3.Sky(nside=self.nside, preset_strings=[self.dust_model], output_unit='uK_CMB')
            sky_dust = np.array(sky_dust.get_emission(self.nu0_dust * u.Hz))
            sky_dust_beta = pysm3.Sky(nside=max(self.nside_beta, 1), preset_strings=[self.dust_model], output_unit='uK_CMB')
            beta_dust = np.array(sky_dust_beta.components[0].mbb_index)
            real_sky['dust'] = sky_dust.T * self.dust_level # shape (npixel, 3)
            real_sky['beta_dust'] = beta_dust.copy() # shape nbeta

        # Synchrotron
        if self.synchrotron_level:
            sky_synchrotron = pysm3.Sky(nside=self.nside, preset_strings=[self.synchrotron_model], output_unit='uK_CMB')
            sky_synchrotron = np.array(sky_synchrotron.get_emission(self.nu0_synchrotron * u.Hz))
            sky_synchrotron_beta = pysm3.Sky(nside=max(self.nside_beta, 1), preset_strings=[self.synchrotron_model], output_unit='uK_CMB')
            beta_synchrotron = np.array(sky_synchrotron_beta.components[0].pl_index)
            real_sky['synchrotron'] = sky_synchrotron.T * self.synchrotron_level # shape (npixel, 3)
            real_sky['beta_synchrotron'] = beta_synchrotron.copy() # shape nbeta
            
        return real_sky


    def get_tod(self, real_sky):
        '''
        Creating the TOD
        tod_qubic has shape (ndetectors, npointings)
        tod_planck has shape (len(frequencies_planck) * npixel * 3)
        '''
        # Qubic
        tod_qubic = np.zeros(self.H_list[0].shapeout) # shape (ndetectors, npointings)
        for index, freq in enumerate(self.frequencies_qubic):
            # CMB
            mixed_sky = real_sky['cmb'].copy() # shape (npixel, 3)

            # Dust
            if self.dust_level:
                mbb_beta = self.modified_black_body_dust(freq, real_sky['beta_dust']) # shape nbeta
                if self.dust_model == 'd1':
                    up_grade_mbb_beta = hp.ud_grade(mbb_beta, self.nside) # shape npixel
                    mixed_sky += up_grade_mbb_beta[:, None] * real_sky['dust'] # shape (npixel, 3)
                else: # self.dust_model == 'd0'
                    mixed_sky += mbb_beta * real_sky['dust'] # shape (npixel, 3)

            # Synchrotron
            if self.synchrotron_level:
                pl_beta = self.power_law_synchrotron(freq, real_sky['beta_synchrotron']) # shape nbeta
                if self.synchrotron_model == 's1':
                    up_grade_pl_beta = hp.ud_grade(pl_beta, self.nside) # shape npixel
                    mixed_sky += up_grade_pl_beta[:, None] * real_sky['synchrotron'] # shape (npixel, 3)
                else: # self.synchrotron_model = 's0'
                    mixed_sky += pl_beta * real_sky['synchrotron'] # shape (npixel, 3)

            tod_qubic += self.H_list[index](mixed_sky)
        tod_qubic *= (1 + np.random.normal(0, self.noise_qubic, tod_qubic.shape)) # Add noise

        # Planck
        tod_planck = np.empty((len(self.frequencies_planck), self.npixel, 3))
        for index, freq in enumerate(self.frequencies_planck):
            # CMB
            mixed_sky = real_sky['cmb'].copy() # shape (npixel, 3)

            # Dust
            if self.dust_level:
                mbb_beta = self.modified_black_body_dust(freq, real_sky['beta_dust']) # shape nbeta
                if self.dust_model == 'd1':
                    up_grade_mbb_beta = hp.ud_grade(mbb_beta, self.nside) # shape npixel
                    mixed_sky += up_grade_mbb_beta[:, None] * real_sky['dust'] # shape (npixel, 3)
                else: # self.dust_model == 'd0'
                    mixed_sky += mbb_beta * real_sky['dust'] # shape (npixel, 3)

            # Synchrotron
            if self.synchrotron_level:
                pl_beta = self.power_law_synchrotron(freq, real_sky['beta_synchrotron']) # shape nbeta
                if self.synchrotron_model == 's1':
                    up_grade_pl_beta = hp.ud_grade(pl_beta, self.nside) # shape npixel
                    mixed_sky += up_grade_pl_beta[:, None] * real_sky['synchrotron'] # shape (npixel, 3)
                else: # self.synchrotron_model = 's0'
                    mixed_sky += pl_beta * real_sky['synchrotron'] # shape (npixel, 3)

            tod_planck[index, ...] = mixed_sky
        tod_planck = tod_planck.ravel() # shape (len(frequencies_planck) * npixel * 3)
        tod_planck *= (1 + np.random.normal(0, self.noise_planck, tod_planck.shape)) # Add noise
            
        return tod_qubic, tod_planck


    def component_splitter(self, component_map):
        '''
        Splits the component_map into the different components.
        component_map is an array of shape combining the cmb, dust, synchrotron and thaire spectral indices.
        Returns split_map, a dictionary of the components.
        '''
        split_map = {}
        # CMB
        split_map['cmb'] = component_map[:3*self.npixel_patch].reshape(3, self.npixel_patch).T.copy() # shape (npixel_patch, 3)

        # Dust
        index = 3*self.npixel_patch
        if self.dust_reconstruction:
            split_map['dust'] = component_map[index:index+3*self.npixel_patch].reshape(3, self.npixel_patch).T.copy() # shape (npixel_patch, 3)
            index += 3*self.npixel_patch
            split_map['beta_dust'] = component_map[index:index+self.nbeta_patch].copy() # shape (nbeta_patch)
            index += self.nbeta_patch

        # Synchrotron
        if self.synchrotron_reconstruction:
            split_map['synchrotron'] = component_map[index:index+3*self.npixel_patch].reshape(3, self.npixel_patch).T.copy() # shape (npixel_patch, 3)
            index += 3*self.npixel_patch
            split_map['beta_synchrotron'] = component_map[index:index+self.nbeta_patch].copy() # shape (nbeta_patch)

        return split_map


    def component_combiner(self, split_map):
        '''
        Inverse function of component_splitter.
        Returns a array component_map from a dictionary split_map of the components.
        '''
        # CMB
        component_map = split_map['cmb'].T.ravel().copy()

        # Dust
        if self.dust_reconstruction:
            component_map = np.concatenate((component_map, split_map['dust'].T.ravel(), split_map['beta_dust']))

        # Synchrotron
        if self.synchrotron_reconstruction:
            component_map = np.concatenate((component_map, split_map['synchrotron'].T.ravel(), split_map['beta_synchrotron']))

        return component_map


    def beta_to_pixel(self, beta_map):
        '''
        Takes a vector of shape nbeta_patch and returns a vector of shape npixel_patch.
        Repeats the elements of beta_map over the pixels contained in one super-pixel.
        '''
        if self.nside_beta:
            fullsky_beta_map = np.zeros(self.nbeta) # shape nbeta
            fullsky_beta_map[self.seenpix_qubic_beta] = beta_map.copy()
            fullsky_pixel_map = hp.ud_grade(fullsky_beta_map, self.nside) # shape npixel
            return fullsky_pixel_map[self.seenpix_qubic] # shape npixel_patch
        else: 
            return np.tile(beta_map, self.npixel_patch)


    def pixel_to_beta(self, pixel_map):
        '''
        Takes a vector of shape npixel_patch and returns a vector of shape nbeta_patch.
        Takes the average of the pixels contained in one super-pixel.
        '''
        if self.nside_beta:
            fullsky_pixel_map = np.zeros(self.npixel) # shape npixel
            fullsky_pixel_map[self.seenpix_qubic] = pixel_map.copy()
            fullsky_beta_map = hp.ud_grade(fullsky_pixel_map, self.nside_beta) # shape nbeta
            return fullsky_beta_map[self.seenpix_qubic_beta] # shape nbeta_patch
        else:
            return np.sum(pixel_map) / self.npixel_patch


    def get_mixing_operators(self):
        '''
        The mixing operator A_\nu: giving a component vector, it returns the mixed sky of shape (npixel, 3).
        A_nu(c)[i,:] = (CMB_I_i + mbb(beta_d_i)dust_I_i + pl(beta_s_i)sync_I_i,  CMB_Q_i + mbb(beta_d_i)dust_Q_i + 
            pl(beta_s_i)sync_Q_i,  CMB_U_i + mbb(beta_d_i)dust_U_i + pl(beta_s_i)sync_U_i),
        (each beta is used for multiple pixels).
        Returns the list of the mixing operators at the different wavelengths.
        '''
        def mixing_function(component_map, freq, out):
            split_map = self.component_splitter(component_map)
            out[...] = np.zeros((self.npixel, 3))
            
            # CMB
            out[self.seenpix_qubic, :] = split_map['cmb'].copy() # shape (npixel_patch, 3)

            # dust
            if self.dust_reconstruction:
                mbb_beta = self.modified_black_body_dust(freq, split_map['beta_dust']) # shape nbeta_patch
                up_grade_mbb_beta = self.beta_to_pixel(mbb_beta) # shape npixel_patch
                out[self.seenpix_qubic, :] += up_grade_mbb_beta[:, None] * split_map['dust'] # shape (npixel_patch, 3)

            # synchrotron
            if self.synchrotron_reconstruction:
                pl_beta = self.power_law_synchrotron(freq, split_map['beta_synchrotron']) # shape nbeta_patch
                up_grade_pl_beta = self.beta_to_pixel(pl_beta) # shape npixel_patch
                out[self.seenpix_qubic, :] += up_grade_pl_beta[:, None] * split_map['synchrotron'] # shape (npixel_patch, 3)
        
        Mixing_matrices_qubic = []
        for freq in self.frequencies_qubic:
            Mixing_matrices_qubic.append(Operator(lambda component_map, out, freq=freq : mixing_function(component_map, freq, out), 
                                   shapein=self.component_map_size, shapeout=(self.npixel, 3), dtype='float64'))

        Mixing_matrices_planck = []
        for freq in self.frequencies_planck:
            Mixing_matrices_planck.append(Operator(lambda component_map, out, freq=freq : mixing_function(component_map, freq, out), 
                                   shapein=self.component_map_size, shapeout=(self.npixel, 3), dtype='float64'))
        
        return Mixing_matrices_qubic, Mixing_matrices_planck
    

    def get_jacobien_mixing_operators(self):
        '''
        This operator is the transpose of the Jacobian of the mixing matrix. It is a non-linear operator.
        Returns an operator that can be called to generate the transposed Jacobian operator.
        '''
        # Define the inner operator class
        class Transposed_Jacobian(Operator):
            def __init__(self_jacob, component_map, freq):
                self_jacob.split_map = self.component_splitter(component_map)
                self_jacob.freq = freq
                super().__init__(shapein=(self.npixel,3), shapeout=self.component_map_size, dtype='float64')
            
            def direct(self_jacob, input_map, output):
                split_output = {}
                
                # CMB
                split_output['cmb'] = input_map[self.seenpix_qubic, :].copy() # shape (npixel_patch, 3)

                # dust
                if self.dust_reconstruction:
                    mbb_beta = self.modified_black_body_dust(self_jacob.freq, self_jacob.split_map['beta_dust']) # shape nbeta_patch
                    up_grade_mbb_beta = self.beta_to_pixel(mbb_beta) # shape npixel_patch
                    derive_mbb_beta = mbb_beta * np.log(self_jacob.freq/self.nu0_dust) # shape nbeta_patch

                    split_output['dust'] = up_grade_mbb_beta[:, None] * input_map[self.seenpix_qubic, :] # shape (npixel_patch, 3)
                    split_output['beta_dust'] = self.pixel_to_beta(
                        np.sum(self_jacob.split_map['dust'] * input_map[self.seenpix_qubic, :], 
                               axis=1)) * (self.npixel // self.nbeta) * derive_mbb_beta # shape nbeta_patch

                # synchrotron
                if self.synchrotron_reconstruction:
                    pl_beta = self.power_law_synchrotron(self_jacob.freq, self_jacob.split_map['beta_synchrotron']) # shape nbeta_patch
                    up_grade_pl_beta = self.beta_to_pixel(pl_beta) # shape npixel_patch
                    derive_pl_beta = pl_beta * np.log(self_jacob.freq/self.nu0_synchrotron) # shape nbeta_patch

                    split_output['synchrotron'] = up_grade_pl_beta[:, None] * input_map[self.seenpix_qubic, :] # shape (npixel_patch, 3)
                    split_output['beta_synchrotron'] = self.pixel_to_beta(
                        np.sum(self_jacob.split_map['synchrotron'] * input_map[self.seenpix_qubic, :], 
                               axis=1)) * (self.npixel // self.nbeta) * derive_pl_beta # shape nbeta_patch
                
                output[...] = self.component_combiner(split_output)
        
        # Define the outer operator class
        class Generate_Transposed_Jacobian(Operator):
            def direct(self_jacob, component_map, freq, output):
                # Create the generated operator
                transposed_jacobian = Transposed_Jacobian(component_map, freq)
                # Store the generated operator in the output
                output[...] = transposed_jacobian
        
        return Generate_Transposed_Jacobian()


    def get_noise_inverse_covariance(self):
        '''
        Operators for the inverse covariances matrices of Qubic and Planck.
        The shapein of invN_qubic is (ndetectors, npointings).
        The shapein of invN_planck is (len(frequencies_planck) * npixel * 3).
        Both operators are symmetric.
        A mask operator is also applied to invN_planck, so that Planck's maps are only 
        added where the coverage is under planck_coverage_level.
        '''
        invN_qubic = self.Q.get_invntt_operator()
        
        Planck = OtherDataParametric([int(freq/1e9) for freq in self.frequencies_planck], self.nside, [])
        invN_planck = Planck.get_invntt_operator()
        coverage = self.H_list[0].T(np.ones(self.H_list[0].shapeout))
        mask = np.tile(coverage.ravel(), len(self.frequencies_planck)) / np.max(coverage) < self.planck_coverage_level
        invN_planck = invN_planck * DiagonalOperator(mask) # Planck's maps are only added where the coverage is under planck_coverage_level
        
        return invN_qubic, invN_planck


    def get_preconditioner(self, invN_qubic, invN_planck):
        '''
        We compute an approximation of the inverse of the diagonal of the hessian matrix of chi^2. 
        This is used as a preconditioner for the non-linear PCG. It is very important as the 
        components maps and the spectral indices have a very different behaviour in the PCG. 
        This preconditioner helps making those different parameters more like one another.
        '''
        # Approximation of H.T N^{-1} H for Qubic
        vector = np.ones(self.H_list[0].shapein)
        self.approx_HTNH = np.empty((len(self.H_list), self.npixel_patch)) # shape (Nsub, npixel_patch)
        for index in range(len(self.H_list)):
            self.approx_HTNH[index] = (self.H_list[index].T * invN_qubic * self.H_list[index] * vector)[self.seenpix_qubic, 0] / 50 
            # The factor 50 is a renormalization factor to help the preconditioner of Qubic and of Planck being in the same range.
            # It is purely empirical and could maybe be improved.
        
        def diagonal_qubic(split_map):
            # Preconditioner for Qubic
            split_preconditioner = {}

            # CMB
            split_preconditioner['cmb'] = np.repeat(np.sum(self.approx_HTNH, axis=0)[:, None], 3, axis=1) # shape (npixel_patch, 3)

            # dust
            if self.dust_reconstruction:
                dust_mbb_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch)) # shape (Nsub, npixel_patch)
                derive_dust_mbb_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch)) # shape (Nsub, npixel_patch)
                for index, freq in enumerate(self.frequencies_qubic):
                    dust_mbb = self.beta_to_pixel(self.modified_black_body_dust(freq, split_map['beta_dust']))
                    dust_mbb_squared[index, :] = dust_mbb**2
                    derive_dust_mbb_squared[index, :] = (dust_mbb * np.log(freq/self.nu0_dust))**2
               
                split_preconditioner['dust'] = np.repeat(np.sum(self.approx_HTNH * dust_mbb_squared, axis=0)
                                                         [:, None], 3, axis=1) # shape (npixel_patch, 3)
                split_preconditioner['beta_dust'] = self.pixel_to_beta(
                    np.sum(np.sum(split_map['dust']**2, axis=1) * self.approx_HTNH * derive_dust_mbb_squared, axis=0))

            # synchrotron
            if self.synchrotron_reconstruction:
                synchrotron_pl_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch)) # shape (Nsub, npixel_patch)
                derive_synchrotron_pl_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch)) # shape (Nsub, npixel_patch)
                for index, freq in enumerate(self.frequencies_qubic):
                    synchrotron_pl = self.beta_to_pixel(self.power_law_synchrotron(freq, split_map['beta_synchrotron']))
                    synchrotron_pl_squared[index, :] = synchrotron_pl**2
                    derive_synchrotron_pl_squared[index, :] = (synchrotron_pl * np.log(freq/self.nu0_synchrotron))**2
               
                split_preconditioner['synchrotron'] = np.repeat(np.sum(self.approx_HTNH * synchrotron_pl_squared, axis=0)
                                                                [:, None], 3, axis=1) # shape (npixel_patch, 3)
                split_preconditioner['beta_synchrotron'] = self.pixel_to_beta(
                    np.sum(np.sum(split_map['synchrotron']**2, 
                                  axis=1) * self.approx_HTNH * derive_synchrotron_pl_squared, axis=0)) # shape nbeta_patch

            return self.component_combiner(split_preconditioner)

        # Approximation of invN_planck, has shape (len(frequencies_planck), npixel_patch, 3)
        self.approx_invN_planck = invN_planck(np.ones(invN_planck.shapein)).reshape(
            (len(self.frequencies_planck), self.npixel, 3))[:, self.seenpix_qubic, :]

        def diagonal_planck(split_map):
            # Preconditioner for Planck
            split_preconditioner = {}

            # CMB
            split_preconditioner['cmb'] = np.sum(self.approx_invN_planck, axis=0) # shape (npixel_patch, 3)

            # dust
            if self.dust_reconstruction:
                dust_mbb_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch)) # shape (Nsub, npixel_patch)
                derive_dust_mbb_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch)) # shape (Nsub, npixel_patch)
                for index, freq in enumerate(self.frequencies_planck):
                    dust_mbb = self.beta_to_pixel(self.modified_black_body_dust(freq, split_map['beta_dust']))
                    dust_mbb_squared[index, :] = dust_mbb**2
                    derive_dust_mbb_squared[index, :] = (dust_mbb * np.log(freq/self.nu0_dust))**2
               
                split_preconditioner['dust'] = np.sum(self.approx_invN_planck * dust_mbb_squared[..., None], axis=0) # shape (npixel_patch, 3)
                split_preconditioner['beta_dust'] = self.pixel_to_beta(
                    np.sum(np.sum(split_map['dust']**2 * self.approx_invN_planck, 
                                  axis=2) * derive_dust_mbb_squared, axis=0)) # shape nbeta_patch

            # synchrotron
            if self.synchrotron_reconstruction:
                synchrotron_pl_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch)) # shape (Nsub, npixel_patch)
                derive_synchrotron_pl_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch)) # shape (Nsub, npixel_patch)
                for index, freq in enumerate(self.frequencies_planck):
                    synchrotron_pl = self.beta_to_pixel(self.power_law_synchrotron(freq, split_map['beta_synchrotron']))
                    synchrotron_pl_squared[index, :] = synchrotron_pl**2
                    derive_synchrotron_pl_squared[index, :] = (synchrotron_pl * np.log(freq/self.nu0_synchrotron))**2
               
                split_preconditioner['synchrotron'] = np.sum(self.approx_invN_planck * 
                                                             synchrotron_pl_squared[..., None], axis=0) # shape (npixel_patch, 3)
                split_preconditioner['beta_synchrotron'] = self.pixel_to_beta(
                    np.sum(np.sum(split_map['synchrotron']**2 * self.approx_invN_planck, 
                                  axis=2) * derive_synchrotron_pl_squared, axis=0)) # shape nbeta_patch

            return self.component_combiner(split_preconditioner)

        def hessian_inverse_diagonal(component_map, out):
            # The gradient of the chi^2 is the sum of the one of Qubic and of Planck.
            # Therefore the preconditioner is the inverse of the sum of the preconditioner of Qubic and Planck
            split_map = self.component_splitter(component_map)
            out[...] = 1 / (diagonal_qubic(split_map) + diagonal_planck(split_map))

        return Operator(hessian_inverse_diagonal, shapein=self.component_map_size, shapeout=self.component_map_size, dtype='float64')


    