import qubic
import numpy as np
import matplotlib.pyplot as plt
from pyoperators import *
import healpy as hp
import pysm3
import pysm3.units as u
from time import time
import pickle
import os

from Qacquisition import QubicFullBandSystematic, OtherDataParametric
from non_linear_pcg_preconditioned import non_linear_pcg


class NonLinearCMM:
    def __init__(self, nside, nside_beta, npointings, Nsub, frequencies_planck, noise_qubic, noise_planck, planck_coverage_level):
        '''
        nside: (int, power of 2) nside of the pixels in the sky from the module healpy.
        nside_beta: (int, power of 2) Same but for the spectral indices of the dust.
        npointings: (int) Number of random pointings for the acquisition matrix.
        Nsub: (int) Number of frequencies considered in the full band.
        frequencies_planck: (array) Planck frequencies in Hz.
        noise_qubic: (float) Level of normal noise in Qubic's TOD.
        noise_planck: (float) Level of normal noise in Planck's TOD.
        planck_coverage_level: (float, between 0 and 1) Relative coverage under which Planck's maps are added to help convergence.
        '''
        if (nside & (nside - 1) != 0) or nside <= 0:
            raise Exception("nside should be a power of 2")
        if nside_beta > nside or (nside_beta & (nside_beta - 1) != 0) or nside_beta <= 0:
            raise Exception("nside_beta should be a power of 2 smaller than nside")
        if npointings < 1:
            raise Exception("npointings should be greater than 1")
        if Nsub < 2 or Nsub%2 != 0:
            raise Exception("Nsub should be even and greater than 2")
        if noise_qubic < 0.:
            raise Exception("noise_qubic should be a positive number")
        if noise_planck < 0.:
            raise Exception("noise_planck should be a positive number")
        if planck_coverage_level < 0. or planck_coverage_level > 1.:
            raise Exception("planck_coverage_level should be comprised between 0 and 1")
            
        self.nside = nside
        self.nside_beta = nside_beta
        self.npointings = npointings
        self.Nsub = Nsub
        self.frequencies_planck = frequencies_planck
        self.noise_qubic = noise_qubic
        self.noise_planck = noise_planck
        self.planck_coverage_level = planck_coverage_level

        self.npixel = 12*self.nside**2 # Number of pixels in the sky
        self.nbeta = 12*self.nside_beta**2 # Number of spectral indices in the sky

        _, frequencies150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.Nsub/2)-1, relative_bandwidth=0.25)
        _, frequencies220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.Nsub/2)-1, relative_bandwidth=0.25)
        self.frequencies_qubic = np.concatenate((frequencies150, frequencies220)) * 1e9 # Frequencies of Qubic in Hz
        self.nu0 = self.frequencies_qubic[-1] # Reference frequency

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
        
        self.seenpix_qubic = self.H_list[0].T(np.ones(self.H_list[0].shapeout)) != 0.0
        for i in range(len(self.frequencies_qubic)):
            np.logical_or(self.seenpix_qubic, self.H_list[i].T(np.ones(self.H_list[0].shapeout)) != 0.0, out = self.seenpix_qubic)
        self.seenpix_qubic = self.seenpix_qubic[:, 0] # Boolean mask of the pixels of Qubic's patch
        
        self.seenpix_qubic_beta = hp.ud_grade(self.seenpix_qubic, self.nside_beta) # Boolean mask of the spectral indices of Qubic's patch

        self.npixel_patch = np.count_nonzero(self.seenpix_qubic) # Number of pixels in Qubic's patch
        self.nbeta_patch = np.count_nonzero(self.seenpix_qubic_beta) # Number of spectral indices in Qubic's patch

        # Estimation of the rank of the acquistion matrix
        rank = 1 - (1 - 1/(6*self.npixel_patch+self.nbeta_patch) * 992/5)**self.npointings 
        print(f'You are trying to reconstruct {6*self.npixel_patch+self.nbeta_patch} parameters.')
        print(f'The analytically estimated rank of the acquisition matrix is {rank*100} %.')
        if rank < 0.999:
            print('The rank is too low! Increase the number of pointings.')
        else:
            print('Make sure you have enough random pointings.')

    
    def get_real_sky(self):
        '''
        The components maps and spectral indices that we will try to reconstruct.
        '''
        skycmb = pysm3.Sky(nside=self.nside, preset_strings=['c1'], output_unit='uK_CMB')
        skydust = pysm3.Sky(nside=self.nside, preset_strings=['d1'], output_unit='uK_CMB')
        skycmb = np.array(skycmb.get_emission(self.frequencies_qubic[-1] * u.Hz))
        skydust = np.array(skydust.get_emission(self.frequencies_qubic[-1] * u.Hz))
        skydust_beta = pysm3.Sky(nside=self.nside_beta, preset_strings=['d1'], output_unit='uK_CMB')
        true_beta = np.array(skydust_beta.components[0].mbb_index)
        return np.concatenate((skycmb[0,:], skycmb[1,:], skycmb[2,:], skydust[0,:], skydust[1,:], skydust[2,:], true_beta))

    
    def patch_operators(self):
        '''
        Operator Patch_to_Sky takes a vector (components maps + spectral indices map) on the patch and put it on the full sky 
        with zeros on the pixels that are not observed by Qubic. The operator Sky_to_Patch does the opposite.
        '''
        patch_mask = np.concatenate((np.tile(self.seenpix_qubic, 6), self.seenpix_qubic_beta))

        def patch_to_sky(c, out):
            sky = np.zeros(6*self.npixel+self.nbeta)
            sky[patch_mask] = c
            out[...] = sky

        Patch_to_Sky = Operator(patch_to_sky, shapein=6*self.npixel_patch+self.nbeta_patch, shapeout=6*self.npixel+self.nbeta, dtype='float64')

        def sky_to_patch(c, out):
            out[...] = c[patch_mask]

        Sky_to_Patch = Operator(sky_to_patch, shapein=6*self.npixel+self.nbeta, shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')

        return Patch_to_Sky, Sky_to_Patch

    
    def modified_black_body(self, freq, beta):
        '''
        The modified black-body-spectrum of the dust. We have: h/(kT) = 2.4 x 10^(-12) Hz^(-1) at T = 20 K.
        '''
        return (np.exp(freq * 2.4e-12) - 1) / (np.exp(self.nu0 * 2.4e-12) - 1) * (freq / self.nu0)**beta

    
    def get_mixing_operators(self):
        '''
        The mixing operator A_\nu: giving a vector of shape (6*npixel+nbeta), it returns the mixed sky of shape (npixel, 3).
        A_nu(c)[i,:] = (CMB I_i + f(beta_i)dust I_i,  CMB Q_i + f(beta_i)dust Q_i,  CMB U_i + f(beta_i)dust U_i),
        with f the modified blackbody spectrum, and beta_i the value of the spectral index at pixel i 
        (each beta is used for multiple pixels).
        Returns the list of the mixing operators at the different wavelengths, for Qubic and Planck.
        '''
        def function_A(c, freq, out):
            power_beta = self.modified_black_body(freq, c[6*self.npixel:])
            up_grade_power_beta = hp.ud_grade(power_beta, self.nside)
        
            out[:,0] = c[:self.npixel] + up_grade_power_beta * c[3*self.npixel:4*self.npixel] # I
            out[:,1] = c[self.npixel:2*self.npixel] + up_grade_power_beta * c[4*self.npixel:5*self.npixel] # Q
            out[:,2] = c[2*self.npixel:3*self.npixel] + up_grade_power_beta * c[5*self.npixel:6*self.npixel] # U
        
        A_qubic_list = []
        for freq in self.frequencies_qubic:
            A_qubic_list.append(Operator(lambda c, out, freq=freq : function_A(c, freq, out), 
                                   shapein=6*self.npixel+self.nbeta, shapeout=(self.npixel, 3), dtype='float64'))

        A_planck_list = []
        for freq in self.frequencies_planck:
            A_planck_list.append(Operator(lambda c, out, freq=freq : function_A(c, freq, out), 
                    shapein=6*self.npixel+self.nbeta, shapeout=(self.npixel, 3), dtype='float64'))
            
        return A_qubic_list, A_planck_list

    
    def get_jacobien_mixing_operators(self):
        '''
        This operator is the transpose of the Jacobian of the mixing matrix. It is a non-linear operator.
        Returns an operator that can be called to generate the transposed Jacobian operator.
        '''
        # Define the inner operator class
        class Transposed_Jacobian(Operator):
            def __init__(self1, c, freq):
                self1.c = c
                self1.freq = freq
                super().__init__(shapein=(self.npixel,3), shapeout=6*self.npixel+self.nbeta, dtype='float64')
            
            def direct(self1, input_vector, output):
                power_beta = self.modified_black_body(self1.freq, self1.c[6*self.npixel:])
                derive_power_beta = power_beta * np.log(self1.freq/self.nu0)
                up_grade_power_beta = hp.ud_grade(power_beta, self.nside)

                # CMB
                output[:self.npixel] = input_vector[:,0] # I
                output[self.npixel:2*self.npixel] = input_vector[:,1] # Q
                output[2*self.npixel:3*self.npixel] = input_vector[:,2] # U

                # Dust
                output[3*self.npixel:4*self.npixel] = up_grade_power_beta * input_vector[:,0] # I
                output[4*self.npixel:5*self.npixel] = up_grade_power_beta * input_vector[:,1] # Q
                output[5*self.npixel:6*self.npixel] = up_grade_power_beta * input_vector[:,2] # U

                # Spectral indices
                product = self1.c[3*self.npixel:4*self.npixel] * input_vector[:,0] 
                product += self1.c[4*self.npixel:5*self.npixel] * input_vector[:,1] 
                product += self1.c[5*self.npixel:6*self.npixel] * input_vector[:,2]
                product = hp.ud_grade(product, self.nside_beta) * (self.npixel // self.nbeta)
                output[6*self.npixel:] = derive_power_beta * product
        
        # Define the outer operator class
        class Generate_Transposed_Jacobian(Operator):
            def direct(self1, c, freq, output):
                # Create the generated operator
                transposed_jacobian = Transposed_Jacobian(c, freq)
                # Store the generated operator in the output
                output[...] = transposed_jacobian
        
        return Generate_Transposed_Jacobian()

    
    def get_tod(self, A_qubic_list, A_planck_list, true_c):
        '''
        Creating the TOD
        tod_qubic has shape (ndetectors, npointings)
        tod_planck has shape (len(frequencies_planck) * npixel * 3)
        '''
        tod_qubic = self.H_list[0](A_qubic_list[0](true_c))
        for i in range(1, len(self.frequencies_qubic)):
            tod_qubic += self.H_list[i](A_qubic_list[i](true_c))
        tod_qubic *= (1 + np.random.normal(0, self.noise_qubic, tod_qubic.shape)) # Add noise

        tod_planck = np.empty((len(self.frequencies_planck), self.npixel, 3))
        for i in range(len(self.frequencies_planck)):
            tod_planck[i] = A_planck_list[i](true_c)
        tod_planck = tod_planck.ravel()
        tod_planck *= (1 + np.random.normal(0, self.noise_planck, tod_planck.shape)) # Add noise
            
        return tod_qubic, tod_planck

    
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

    
    def get_grad_chi_squared_operator(self, A_qubic_list, A_planck_list, Patch_to_Sky, Sky_to_Patch, tod_qubic, 
                                      tod_planck, invN_qubic, invN_planck, generate_transposed_jacobian, true_c):
        '''
        The gradient of the chi^2 operator. It is the sum of the of the gradient of the chi^2 of Qubic and of Planck. 
        For Qubic, the gradient is written as follow:
        \nabla \chi^2(\tilde{c}) = \sum (J_{A_\nu}(\tilde{c}))^T H^T N^{-1} \sum H{A_\nu}(\tilde{c}) - \sum (J_{A_\nu}(\tilde{c}))^T H^T N^{-1} d
        '''
        def grad_operator(c, out):
            # Qubic
            sum_H_qubic = self.H_list[0](A_qubic_list[0](Patch_to_Sky(c)))
            for i in range(1, len(self.frequencies_qubic)):
                sum_H_qubic += (self.H_list[i](A_qubic_list[i](Patch_to_Sky(c))))
            sum_H_qubic -= tod_qubic
            sum_H_qubic = invN_qubic(sum_H_qubic)
        
            output_operator = np.empty((), dtype=object)
            generate_transposed_jacobian.direct(Patch_to_Sky(c), self.frequencies_qubic[0], output_operator)
            transposed_jacobian = output_operator.item()
            gradient = transposed_jacobian(self.H_list[0].T(sum_H_qubic))
            for i in range(1, len(self.frequencies_qubic)):
                generate_transposed_jacobian.direct(Patch_to_Sky(c), self.frequencies_qubic[i], output_operator)
                transposed_jacobian = output_operator.item()
                gradient += transposed_jacobian(self.H_list[i].T(sum_H_qubic))
                
            # Planck
            sum_H_planck = np.empty((len(self.frequencies_planck), self.npixel, 3))
            for i in range(len(self.frequencies_planck)):
                sum_H_planck[i, ...] = A_planck_list[i](Patch_to_Sky(c))
            sum_H_planck = sum_H_planck.ravel() - tod_planck
            sum_H_planck = invN_planck(sum_H_planck)
            sum_H_planck = sum_H_planck.reshape((len(self.frequencies_planck), self.npixel, 3))

            for i in range(len(self.frequencies_planck)):
                generate_transposed_jacobian.direct(Patch_to_Sky(c), self.frequencies_planck[i], output_operator)
                transposed_jacobian = output_operator.item()
                gradient += transposed_jacobian(sum_H_planck[i])

            out[...] = Sky_to_Patch(gradient)

        Grad_chi_squared = Operator(grad_operator, shapein=6*self.npixel_patch+self.nbeta_patch, 
                                    shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')

        return Grad_chi_squared


    def get_preconditioner(self, Patch_to_Sky, invN_qubic, invN_planck):
        '''
        We compute an approximation of the inverse of the diagonal of the hessian matrix of chi^2. 
        This is used as a preconditioner for the non-linear PCG. It is very important as the 
        components maps and the spectral indices have a very different behaviour in the PCG. 
        This preconditioner helps making those different parameters more like one another.
        '''
        # Approximation of H.T N^{-1} H for Qubic
        vector = np.ones(self.H_list[0].shapein)
        self.approx_HTNH = np.empty((len(self.H_list), self.npixel_patch)) # has shape (Nsub, npixel_patch)
        for index in range(len(self.H_list)):
            self.approx_HTNH[index] = (self.H_list[index].T * invN_qubic * self.H_list[index] * vector)[self.seenpix_qubic, 0] / 50 
            # The factor 50 is a renormalization factor to help the preconditioner of Qubic and of Planck being in the same range.
            # It is purely empirical and could maybe be improved.
        
        def diagonal_qubic(c):
            # Preconditioner for Qubic
            dust_spectrum_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch))
            derive_dust_spectrum_squared = np.zeros((len(self.frequencies_qubic), self.npixel_patch))
            for index, freq in enumerate(self.frequencies_qubic):
                dust_spectrum = hp.ud_grade(self.modified_black_body(freq, Patch_to_Sky(c)[6*self.npixel:]), self.nside)[self.seenpix_qubic]
                dust_spectrum_squared[index,:] = dust_spectrum**2
                derive_dust_spectrum_squared[index,:] = (dust_spectrum * np.log(freq/self.nu0))**2

            precon = np.empty(6*self.npixel_patch+self.nbeta_patch)
            
            # CMB
            precon[:3*self.npixel_patch] = np.tile(np.sum(self.approx_HTNH, axis=0), 3)
        
            # Dust
            precon[3*self.npixel_patch:6*self.npixel_patch] = np.tile(np.sum(self.approx_HTNH * dust_spectrum_squared, axis=0), 3)
        
            # Spectral indices
            factor1 = c[3*self.npixel_patch:4*self.npixel_patch]**2 # shape (npixel_patch)
            factor1 += c[4*self.npixel_patch:5*self.npixel_patch]**2
            factor1 += c[5*self.npixel_patch:6*self.npixel_patch]**2
            factor1 = factor1 * self.approx_HTNH # shape (len(frequencies_qubic), npixel_patch)
            factor1 *= derive_dust_spectrum_squared
            factor1 = np.sum(factor1, axis=0) # shape (npixel_patch)
            
            downgrader = np.zeros(self.npixel)
            downgrader[self.seenpix_qubic] = factor1
            downgrader = hp.ud_grade(downgrader, self.nside_beta)*(self.npixel//self.nbeta)
            precon[6*self.npixel_patch:] = downgrader[self.seenpix_qubic_beta]

            return precon

        # Approximation of invN_planck, has shape (len(frequencies_planck), 3, npixel_patch)
        self.approx_invN_planck = invN_planck(np.ones(invN_planck.shapein)).reshape(
            (len(self.frequencies_planck), self.npixel, 3)).transpose((0,2,1))[:, :, self.seenpix_qubic]

        def diagonal_planck(c):
            # Preconditioner for Planck
            dust_spectrum_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch))
            derive_dust_spectrum_squared = np.zeros((len(self.frequencies_planck), self.npixel_patch))
            for index, freq in enumerate(self.frequencies_planck):
                dust_spectrum = hp.ud_grade(self.modified_black_body(freq, Patch_to_Sky(c)[6*self.npixel:]), self.nside)[self.seenpix_qubic]
                dust_spectrum_squared[index,:] = dust_spectrum**2
                derive_dust_spectrum_squared[index,:] = (dust_spectrum * np.log(freq/self.nu0))**2
            
            precon = np.empty(6*self.npixel_patch+self.nbeta_patch)
            
            # CMB
            precon[:3*self.npixel_patch] = np.sum(self.approx_invN_planck, axis=0).ravel()
        
            # Dust
            precon[3*self.npixel_patch:6*self.npixel_patch] = np.sum(self.approx_invN_planck * dust_spectrum_squared[:, None, :], axis=0).ravel()
        
            # Spectral indices, factor1 has shape (len(frequencies_planck), npixel_patch)
            factor1 = c[3*self.npixel_patch:4*self.npixel_patch]**2 * self.approx_invN_planck[:, 0, :]
            factor1 += c[4*self.npixel_patch:5*self.npixel_patch]**2 * self.approx_invN_planck[:, 1, :]
            factor1 += c[5*self.npixel_patch:6*self.npixel_patch]**2 * self.approx_invN_planck[:, 2, :]
            factor1 *= derive_dust_spectrum_squared
            factor1 = np.sum(factor1, axis=0) # shape (npixel_patch)

            downgrader = np.zeros(self.npixel)
            downgrader[self.seenpix_qubic] = factor1
            downgrader = hp.ud_grade(downgrader, self.nside_beta)*(self.npixel//self.nbeta)
            precon[6*self.npixel_patch:] = downgrader[self.seenpix_qubic_beta]
        
            return precon

        def hessian_inverse_diagonal(c, out):
            # The gradient of the chi^2 is the sum of the one of Qubic and of Planck.
            # Therefore the preconditioner is the inverse of the sum of the preconditioner of Qubic and Planck
            out[...] = 1 / (diagonal_qubic(c) + diagonal_planck(c))

        return Operator(hessian_inverse_diagonal, shapein=6*self.npixel_patch+self.nbeta_patch, 
                        shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')


    def get_initial_guess(self, Sky_to_Patch, true_c):
        '''
        Set an initial guess for the PCG. For CMB I and dust I, we give the true solution as Planck's measurment
        were good. For CMB Q, U and dust Q, U, we give a map set at 0. FOr the spectral indices, we give a map
        set at 1.53, the mean of the spectral indices on the Qubic's patch.
        '''
        initial_guess = np.empty(6*self.npixel_patch+self.nbeta_patch)
        initial_guess[:self.npixel_patch] = Sky_to_Patch(true_c)[:self.npixel_patch].copy() # CMB I
        initial_guess[self.npixel_patch:3*self.npixel_patch] = np.zeros(2*self.npixel_patch) # CMB Q, U
        initial_guess[3*self.npixel_patch:4*self.npixel_patch] = Sky_to_Patch(true_c)[3*self.npixel_patch:4*self.npixel_patch].copy() # Dust I
        initial_guess[4*self.npixel_patch:6*self.npixel_patch] = np.zeros(2*self.npixel_patch) # Dust Q, U
        initial_guess[6*self.npixel_patch:] = np.ones(self.nbeta_patch)*1.53 # Spectral indicies

        return initial_guess


    def plot_residues(self, folder, residues):
        '''
        Plots the evolution of the relative residues.
        '''
        plt.figure(figsize=(12, 8))
        plt.plot(residues[:,0])
        plt.yscale('log')
        plt.grid(axis='y', linestyle='dotted')
        plt.xlabel('Number of iterations')
        plt.ylabel(r'Relative residue $\frac{||\nabla \chi^2(c_{\beta})||}{||\nabla \chi^2(\vec{0})||}$')
        plt.title('Simultaneous reconstruction of components maps and spectral indices using a non-linear PCG')
        plt.savefig(folder+'residues.pdf')
        plt.close()


    def plot_reconstructed_maps(self, folder, reconstructed_maps, initial_guess, true_c):
        '''
        Plots the true maps, the initial maps, the reconstructed maps, and the difference 
        between the true maps and the reconstructed maps.
        '''
        patch_mask = np.concatenate((np.tile(self.seenpix_qubic, 6), self.seenpix_qubic_beta))
        reconstructed_maps_sky = np.tile(np.nan, 6*self.npixel+self.nbeta)
        reconstructed_maps_sky[patch_mask] = reconstructed_maps
        initial_guess_sky = np.tile(np.nan, 6*self.npixel+self.nbeta)
        initial_guess_sky[patch_mask] = initial_guess
        
        name_list = ['CMB I','CMB Q','CMB U','dust I','dust Q','dust U',r'$\beta$']
        
        plt.figure(figsize=(12, 25))
        for i in range(6):
            hp.gnomview(true_c[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+1), title='True '+name_list[i], rot=qubic.equ2gal(0, -57), 
                        reso=23, cmap='jet', min=np.min(true_c[i*self.npixel:(i+1)*self.npixel][self.seenpix_qubic]), 
                       max=np.max(true_c[i*self.npixel:(i+1)*self.npixel][self.seenpix_qubic]))
            hp.gnomview(initial_guess_sky[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+2), title='Initial '+name_list[i], 
                        rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            hp.gnomview(reconstructed_maps_sky[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+3), title='Reconstructed '+name_list[i], 
                        rot=qubic.equ2gal(0, -57), 
                        reso=23, cmap='jet')
            r = true_c[i*self.npixel:(i+1)*self.npixel] - reconstructed_maps_sky[i*self.npixel:(i+1)*self.npixel]
            sig = np.std(r[self.seenpix_qubic])
            hp.gnomview(r, sub=(7,4,4*i+4), title='Difference '+name_list[i], rot=qubic.equ2gal(0, -57), reso=23, min=-3*sig, max=3*sig, cmap='jet')
        hp.gnomview(true_c[6*self.npixel:], sub=(7,4,4*6+1), title='True '+name_list[6], rot=qubic.equ2gal(0, -57), reso=23, cmap='jet',
                   min=np.min(true_c[6*self.npixel:][self.seenpix_qubic_beta]), max=np.max(true_c[6*self.npixel:][self.seenpix_qubic_beta]))
        hp.gnomview(initial_guess_sky[6*self.npixel:], sub=(7,4,4*6+2), title='Initial '+name_list[6], 
                    rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
        hp.gnomview(reconstructed_maps_sky[6*self.npixel:], sub=(7,4,4*6+3), title='Reconstructed '+name_list[6], 
                    rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
        r = true_c[6*self.npixel:] - reconstructed_maps_sky[6*self.npixel:]
        sig = np.std(r[self.seenpix_qubic_beta])
        hp.gnomview(r, sub=(7,4,4*6+4), title='Difference '+name_list[6], rot=qubic.equ2gal(0, -57), reso=23, min=-3*sig, max=3*sig, cmap='jet')
        plt.savefig(folder+'reconstructed_maps.pdf')
        plt.close()
        


    
    def map_making(self, max_iteration, pcg_tolerance, sigma0, initial_guess=None, verbose=True):
        '''
        Map-making of the components maps and the sepctral indices map on the Qubic's patch through a non-linear PCG.
        max_iteration : (int) Maximum number of iteratiuon of the PCG.
        pcg_tolerance: (float) Tolerance of the PCG.
        sigma0: (float) Initial guess of the size of the step et each iteration of the PCG. If set incorrectly, 
            the PCG will not run. You have to test.
        plot_resuslts: (bool) If True, plots the evolution of the relative residues, and the maps of 
            different components and spectral indices.
        initial_guess: (array of shape (6*npixel_patch+nbeta_patch)) Initial guess for the PCG. If None, the code
            calls the function get_initial_guess.

        Returns:
        residues: (array of shape (max_iteration, 8)) List of the residues for all the parameters together, 
            the CMB I, Q, U maps, the dust I, Q, U maps, and the spectral indices map.
        reconstructed_maps: (array of shape (6*npixel_patch+nbeta_patch)) The reconstructed maps at the end of the PCG.
        '''
        if max_iteration < 1:
            raise Exception("max_iteration should be greater than 1")
        if pcg_tolerance < 0.:
            raise Exception("pcg_tolerance should be positive")
        if sigma0 < 0.:
            raise Exception("sigma0 should be positive")
        
        if verbose:
            print('First, all operators have to be defined.')
        self.true_c = self.get_real_sky()
        self.Patch_to_Sky, self.Sky_to_Patch = self.patch_operators()
        self.A_qubic_list, self.A_planck_list = self.get_mixing_operators()
        self.generate_transposed_jacobian = self.get_jacobien_mixing_operators()
        self.tod_qubic, self.tod_planck = self.get_tod(self.A_qubic_list, self.A_planck_list, self.true_c)
        self.invN_qubic, self.invN_planck = self.get_noise_inverse_covariance()
        self.Grad_chi_squared = self.get_grad_chi_squared_operator(self.A_qubic_list, self.A_planck_list, self.Patch_to_Sky, 
                                                                   self.Sky_to_Patch, self.tod_qubic, self.tod_planck, self.invN_qubic, 
                                                                   self.invN_planck, self.generate_transposed_jacobian, self.true_c)
        self.HessianInverseDiagonal = self.get_preconditioner(self.Patch_to_Sky, self.invN_qubic, self.invN_planck)

        # Check that the gradient is zero at the solution point
        if verbose:
            print(f'The gradient should be zero at the solution point, this is {(self.Grad_chi_squared(self.Sky_to_Patch(self.true_c))==0).all()}.')

        if initial_guess is None:
            self.initial_guess = self.get_initial_guess(self.Sky_to_Patch, self.true_c)
        else:
            self.initial_guess = initial_guess

        
        if verbose:
            print('All operators have been defined, PCG is starting.')

        start = time()
        self.residues = []
        pcg = non_linear_pcg(self.Grad_chi_squared, M=self.HessianInverseDiagonal, conjugate_method='polak-ribiere', 
                             x0=self.initial_guess, tol=pcg_tolerance, sigma_0=sigma0, tol_linesearch=1e-3, maxiter=max_iteration, 
                             residues=self.residues, npixel_patch=self.npixel_patch, nbeta_patch=self.nbeta_patch)
        self.reconstructed_maps = pcg['x']
        self.residues = np.array(self.residues)
        self.residues /= np.linalg.norm(self.Grad_chi_squared(self.initial_guess))
        if verbose:
            print(f'Time taken for PCG: {time()-start} sec')


        if self.noise_qubic == 0.0 and self.noise_planck == 0.0:
            noise_str = '_with_noise_'
        else:
            noise_str = '_noiseless_'
        
        folder = 'nside_'+str(self.nside)+'_beta_'+str(self.nside_beta)+'_npointings_'+str(self.npointings)+'_Nsub_'+str(self.Nsub)
        folder += '_Planck_freq_'+str(len(self.frequencies_planck))+noise_str+'_max_iteration_'+str(max_iteration)+'/'

        os.makedirs(folder, exist_ok=True)
        
        if verbose:
            print('Plotting the residues and the maps.')
        self.plot_residues(folder, self.residues)
        print('The scale of the difference maps is set to ± 3 sigma.')
        self.plot_reconstructed_maps(folder, self.reconstructed_maps, self.initial_guess, self.true_c)

        # Save the results in a dictionnary
        self.result_dictionnary = {
            'residues': self.residues,
            'reconstructed_maps': self.reconstructed_maps,
            'true_c': self.true_c,
            'initial_guess': self.initial_guess,
            'nside': self.nside,
            'nside_beta': self.nside_beta,
            'npointings': self.npointings,
            'Nsub': self.Nsub,
            'frequencies_planck': self.frequencies_planck,
            'noise_qubic': self.noise_qubic,
            'noise_planck': self.noise_planck,
            'planck_coverage_level': self.planck_coverage_level,
            'max_iteration': max_iteration,
            'pcg_tolerance': pcg_tolerance,
            'sigma0': sigma0
        }

        with open(folder+'results.pkl', 'wb') as handle:
            pickle.dump(self.result_dictionnary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #return self.residues, self.reconstructed_maps


self = NonLinearCMM(nside=16, nside_beta=8, npointings=100, Nsub=4, frequencies_planck=[100e9, 143e9, 217e9, 353e9],
                    noise_qubic=0, noise_planck=0, planck_coverage_level=0.2)
self.map_making(max_iteration=10, pcg_tolerance=1e-16, sigma0=1e-3)


