import qubic
import numpy as np
import matplotlib.pyplot as plt
from pyoperators import *
import healpy as hp
import pickle
import os

from acquisition_non_linear import NonLinearAcquisition
from non_linear_pcg_preconditioned import non_linear_pcg


class NonLinearPipeline:
    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict
        self.max_iteration = self.parameters_dict['max_iteration']
        self.pcg_tolerance = self.parameters_dict['pcg_tolerance']
        self.sigma0 = self.parameters_dict['sigma0']
        self.initial_guess = self.parameters_dict['initial_guess']
        self.verbose = self.parameters_dict['verbose']
        self.dust_reconstruction = self.parameters_dict['dust_reconstruction']
        self.synchrotron_reconstruction = self.parameters_dict['synchrotron_reconstruction']
        
        if self.max_iteration < 1:
            raise Exception("max_iteration should be greater than 1")
        if self.pcg_tolerance < 0.:
            raise Exception("pcg_tolerance should be positive")
        if self.sigma0 < 0.:
            raise Exception("sigma0 should be positive")

        self.acquisition = NonLinearAcquisition(self.parameters_dict)

        self.nside = self.acquisition.nside
        self.nside_beta = self.acquisition.nside_beta
        self.npointings = self.acquisition.npointings
        self.Nsub = self.acquisition.Nsub
        self.npixel = self.acquisition.npixel
        self.npixel_patch = self.acquisition.npixel_patch
        self.nbeta = self.acquisition.nbeta
        self.nbeta_patch = self.acquisition.nbeta_patch
        self.frequencies_qubic = self.acquisition.frequencies_qubic
        self.frequencies_planck = self.acquisition.frequencies_planck
        self.component_map_size = self.acquisition.component_map_size
        self.seenpix_qubic = self.acquisition.seenpix_qubic
        self.seenpix_qubic_beta = self.acquisition.seenpix_qubic_beta
        self.component_splitter = self.acquisition.component_splitter
        self.component_combiner = self.acquisition.component_combiner
        self.ncomponent = self.acquisition.ncomponent
        
        if self.verbose:
            print('Generating the acquisition matrices')
        self.H_list = self.acquisition.H_list
        if self.verbose:
            print('Generating the real sky')
        self.real_sky = self.acquisition.get_real_sky()
        if self.verbose:
            print('Generating the TOD of Qubic and Planck')
        self.tod_qubic, self.tod_planck = self.acquisition.get_tod(self.real_sky)
        if self.verbose:
            print('Generating the mixing operators and their transposed Jacobian')
        self.Mixing_matrices_qubic, self.Mixing_matrices_planck = self.acquisition.get_mixing_operators()
        self.Generate_Transposed_Jacobian = self.acquisition.get_jacobien_mixing_operators()
        if self.verbose:
            print('Generating the inverse covariance matrix of Qubic and Planck')
        self.invN_qubic, self.invN_planck = self.acquisition.get_noise_inverse_covariance()
        if self.verbose:
            print('Generating preconditioner')
        self.HessianInverseDiagonal = self.acquisition.get_preconditioner(self.invN_qubic, self.invN_planck)


    def get_grad_chi_squared_operator(self):
        '''
        The gradient of the chi^2 operator. It is the sum of the of the gradient of the chi^2 of Qubic and of Planck. 
        For Qubic, the gradient is written as follow:
        \nabla \chi^2(\tilde{c}) = \sum (J_{A_\nu}(\tilde{c}))^T H^T N^{-1} \sum H{A_\nu}(\tilde{c}) - \sum (J_{A_\nu}(\tilde{c}))^T H^T N^{-1} d
        '''
        def grad_operator(component_map, out):
            # Qubic
            # HAc
            sum_H_qubic = self.H_list[0](self.Mixing_matrices_qubic[0](component_map))
            for i in range(1, len(self.frequencies_qubic)):
                sum_H_qubic += self.H_list[i](self.Mixing_matrices_qubic[i](component_map))
            # HAc - d
            sum_H_qubic -= self.tod_qubic
            # N^-1 (HAc - d)
            sum_H_qubic = self.invN_qubic(sum_H_qubic)

            # J_A H^T N^-1 (HAc - d)
            output_operator = np.empty((), dtype=object)
            self.Generate_Transposed_Jacobian.direct(component_map, self.frequencies_qubic[0], output_operator)
            Transposed_Jacobian = output_operator.item()
            out[...] = Transposed_Jacobian(self.H_list[0].T(sum_H_qubic))
            for i in range(1, len(self.frequencies_qubic)):
                self.Generate_Transposed_Jacobian.direct(component_map, self.frequencies_qubic[i], output_operator)
                Transposed_Jacobian = output_operator.item()
                out[...] += Transposed_Jacobian(self.H_list[i].T(sum_H_qubic))
                
            # Planck
            # Ac
            sum_H_planck = np.empty((len(self.frequencies_planck), self.npixel, 3))
            for i in range(len(self.frequencies_planck)):
                sum_H_planck[i, ...] = self.Mixing_matrices_planck[i](component_map)
            # Ac - d
            sum_H_planck = sum_H_planck.ravel() - self.tod_planck
            # N^-1 (Ac - d)
            sum_H_planck = self.invN_planck(sum_H_planck)
            sum_H_planck = sum_H_planck.reshape((len(self.frequencies_planck), self.npixel, 3))

            # J_A N^-1 (Ac - d)
            for i in range(len(self.frequencies_planck)):
                self.Generate_Transposed_Jacobian.direct(component_map, self.frequencies_planck[i], output_operator)
                Transposed_Jacobian = output_operator.item()
                out[...] += Transposed_Jacobian(sum_H_planck[i])

        Grad_chi_squared = Operator(grad_operator, shapein=self.component_map_size, 
                                    shapeout=self.component_map_size, dtype='float64')

        return Grad_chi_squared


    def get_initial_guess(self):
        '''
        Set an initial guess for the PCG. For CMB I and dust I, we give the true solution as Planck's measurment
        were good. For CMB Q, U and dust Q, U, we give a map set at 0. For the spectral indices, we give a map
        set at 1.54, the mean of the spectral indices on the Qubic's patch.
        '''
        initial_guess = {}

        # CMB
        initial_guess['cmb'] = np.concatenate((self.real_sky['cmb'][self.seenpix_qubic, 0][:, None], 
                                               np.zeros((self.npixel_patch, 2))), axis=1)

        # dust
        if self.dust_reconstruction:
            initial_guess['dust'] = np.concatenate((self.real_sky['dust'][self.seenpix_qubic, 0][:, None], 
                                                    np.zeros((self.npixel_patch, 2))), axis=1)
            initial_guess['beta_dust'] = 1.54 * np.ones(self.nbeta_patch)

        # synchrotron
        if self.synchrotron_reconstruction:
            initial_guess['synchrotron'] = np.concatenate((self.real_sky['synchrotron'][self.seenpix_qubic, 0][:, None], 
                                                           np.zeros((self.npixel_patch, 2))), axis=1)
            initial_guess['beta_synchrotron'] = -3.0 * np.ones(self.nbeta_patch)

        return self.component_combiner(initial_guess)


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


    def plot_reconstructed_maps(self, folder, reconstructed_maps, initial_guess):
        '''
        Plots the true maps, the initial maps, the reconstructed maps, and the difference 
        between the true maps and the reconstructed maps.
        '''
        initial_guess_split = self.component_splitter(initial_guess)
        split_map = self.component_splitter(reconstructed_maps)

        sky_vector = np.tile(np.nan, self.npixel)
        beta_vector = np.tile(np.nan, self.nbeta)
        
        Nrow = 3 * (1 + self.ncomponent) + self.ncomponent
        plt.figure(figsize=(12, 3.5 * Nrow))
        
        # CMB
        name_list = ['CMB I', 'CMB Q', 'CMB U']
        for i, name in enumerate(name_list):
            sky_vector[self.seenpix_qubic] = self.real_sky['cmb'][self.seenpix_qubic, i].copy()
            hp.gnomview(sky_vector, sub=(Nrow,4,4*i+1), title='Input '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            sky_vector[self.seenpix_qubic] = initial_guess_split['cmb'][:, i].copy()
            hp.gnomview(sky_vector, sub=(Nrow,4,4*i+2), title='Initial '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            sky_vector[self.seenpix_qubic] = split_map['cmb'][:, i].copy()
            hp.gnomview(sky_vector, sub=(Nrow,4,4*i+3), title='Reconstructed '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            difference = self.real_sky['cmb'][:, i] - sky_vector
            sig = np.std(difference[self.seenpix_qubic])
            hp.gnomview(difference, sub=(Nrow,4,4*i+4), title='Difference '+name, rot=qubic.equ2gal(0, -57), reso=23, 
                        min=-3*sig, max=3*sig, cmap='jet')
        index = 3

        # dust
        if self.dust_reconstruction:
            name_list = ['dust I', 'dust Q', 'dust U']
            for i, name in enumerate(name_list):
                sky_vector[self.seenpix_qubic] = self.real_sky['dust'][self.seenpix_qubic, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+1), title='Input '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                sky_vector[self.seenpix_qubic] = initial_guess_split['dust'][:, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+2), title='Initial '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                sky_vector[self.seenpix_qubic] = split_map['dust'][:, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+3), title='Reconstructed '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                difference = self.real_sky['dust'][:, i] - sky_vector
                sig = np.std(difference[self.seenpix_qubic])
                hp.gnomview(difference, sub=(Nrow,4,4*(i+index)+4), title='Difference '+name, rot=qubic.equ2gal(0, -57), reso=23, 
                            min=-3*sig, max=3*sig, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = self.real_sky['beta_dust'][self.seenpix_qubic_beta].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+1), title=r'Input $\beta_d$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = initial_guess_split['beta_dust'].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+2), title=r'Initial $\beta_d$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = split_map['beta_dust'].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+3), title=r'Reconstructed $\beta_d$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            difference = self.real_sky['beta_dust'] - beta_vector
            sig = np.std(difference[self.seenpix_qubic_beta])
            hp.gnomview(difference, sub=(Nrow,4,4*(3+index)+4), title=r'Difference $\beta_d$', rot=qubic.equ2gal(0, -57), reso=23, 
                        min=-3*sig, max=3*sig, cmap='jet')
            index += 4

        # synchrotron
        if self.synchrotron_reconstruction:
            name_list = ['synchrotron I', 'synchrotron Q', 'synchrotron U']
            for i, name in enumerate(name_list):
                sky_vector[self.seenpix_qubic] = self.real_sky['synchrotron'][self.seenpix_qubic, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+1), title='Input '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                sky_vector[self.seenpix_qubic] = initial_guess_split['synchrotron'][:, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+2), title='Initial '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                sky_vector[self.seenpix_qubic] = split_map['synchrotron'][:, i].copy()
                hp.gnomview(sky_vector, sub=(Nrow,4,4*(i+index)+3), title='Reconstructed '+name, rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
                difference = self.real_sky['synchrotron'][:, i] - sky_vector
                sig = np.std(difference[self.seenpix_qubic])
                hp.gnomview(difference, sub=(Nrow,4,4*(i+index)+4), title='Difference '+name, rot=qubic.equ2gal(0, -57), reso=23, 
                            min=-3*sig, max=3*sig, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = self.real_sky['beta_synchrotron'][self.seenpix_qubic_beta].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+1), title=r'Input $\beta_s$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = initial_guess_split['beta_synchrotron'].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+2), title=r'Initial $\beta_s$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            beta_vector[self.seenpix_qubic_beta] = split_map['beta_synchrotron'].copy()
            hp.gnomview(beta_vector, sub=(Nrow,4,4*(3+index)+3), title=r'Reconstructed $\beta_s$', rot=qubic.equ2gal(0, -57), reso=23, cmap='jet')
            difference = self.real_sky['beta_synchrotron'] - beta_vector
            sig = np.std(difference[self.seenpix_qubic_beta])
            hp.gnomview(difference, sub=(Nrow,4,4*(3+index)+4), title=r'Difference $\beta_s$', rot=qubic.equ2gal(0, -57), reso=23, 
                        min=-3*sig, max=3*sig, cmap='jet')

        plt.savefig(folder+'reconstructed_maps.pdf')
        plt.close()
        


    
    def map_making(self):
        '''
        Map-making of the components maps and the sepctral indices map on the Qubic's patch through a non-linear PCG.
        max_iteration : (int) Maximum number of iteratiuon of the PCG.
        pcg_tolerance: (float) Tolerance of the PCG.
        sigma0: (float) Initial guess of the size of the step et each iteration of the PCG. If set incorrectly, 
            the PCG will not run. You have to test.
        plot_results: (bool) If True, plots the evolution of the relative residues, and the maps of 
            different components and spectral indices.
        initial_guess: (array of shape (6*npixel_patch+nbeta_patch)) Initial guess for the PCG. If None, the code
            calls the function get_initial_guess.

        Returns:
        residues: (array of shape (max_iteration, 8)) List of the residues for all the parameters together, 
            the CMB I, Q, U maps, the dust I, Q, U maps, and the spectral indices map.
        reconstructed_maps: (array of shape (6*npixel_patch+nbeta_patch)) The reconstructed maps at the end of the PCG.
        '''
        if self.initial_guess is None:
            self.initial_guess = self.get_initial_guess()

        self.Grad_chi_squared = self.get_grad_chi_squared_operator()

        # Check that the gradient is zero at the solution point
        if self.verbose:
            component_solution = self.real_sky['cmb'][self.seenpix_qubic, :].T.ravel().copy()
            if self.dust_reconstruction:
                component_solution = np.concatenate((component_solution, self.real_sky['dust'][self.seenpix_qubic, :].T.ravel(),
                                                    self.real_sky['beta_dust'][self.seenpix_qubic_beta]))
            if self.synchrotron_reconstruction:
                component_solution = np.concatenate((component_solution, self.real_sky['synchrotron'][self.seenpix_qubic, :].T.ravel(),
                                                    self.real_sky['beta_synchrotron'][self.seenpix_qubic_beta]))
            print(f'The gradient should be zero at the solution point, this is {(self.Grad_chi_squared(component_solution)==0).all()}.')

        if self.verbose:
            print('PCG is starting.')

        self.residues = []
        pcg = non_linear_pcg(self.Grad_chi_squared, M=self.HessianInverseDiagonal, conjugate_method='polak-ribiere', 
                             x0=self.initial_guess, tol=self.pcg_tolerance, sigma_0=self.sigma0, tol_linesearch=1e-3, maxiter=self.max_iteration, 
                             residues=self.residues, npixel_patch=self.npixel_patch, nbeta_patch=self.nbeta_patch)
        self.reconstructed_maps = pcg['x']
        self.residues = np.array(self.residues)
        self.residues /= np.linalg.norm(self.Grad_chi_squared(self.initial_guess))
        self.pcg_time = pcg['time']
        if self.verbose:
            print(f'Time taken for PCG: {self.pcg_time} sec')

        
        if self.acquisition.noise_qubic == 0.0 and self.acquisition.noise_planck == 0.0:
            noise_str = '_noiseless_'
        else:
            noise_str = '_with_noise_'

        folder = 'results/'
        folder += 'nside_'+str(self.nside)+'_beta_'+str(self.nside_beta)+'_npointings_'+str(self.npointings)+'_Nsub_'+str(self.Nsub)
        if self.acquisition.dust_level:
            folder += '_dust_lvl_'+str('self.acquisition.dust_level')+'_'+str('self.acquisition.dust_model')
        if self.acquisition.synchrotron_level:
            folder += '_synchrotron_lvl_'+str('self.acquisition.synchrotron_level')+'_'+str('self.acquisition.synchrotron_model')
        folder += noise_str+'_max_iteration_'+str(self.max_iteration)+'/'

        os.makedirs(folder, exist_ok=True)
        
        if self.verbose:
            print('Plotting the residues and the maps.')
        self.plot_residues(folder, self.residues)
        print('The scale of the difference maps is set to ± 3 sigma.')
        self.plot_reconstructed_maps(folder, self.reconstructed_maps, self.initial_guess)

        # Save the results in a dictionnary
        self.result_dictionnary = self.parameters_dict.copy()
        self.result_dictionnary['residues'] = self.residues
        self.result_dictionnary['reconstructed_maps'] = self.reconstructed_maps
        self.result_dictionnary['real_sky'] = self.real_sky
        self.result_dictionnary['initial_guess'] = self.initial_guess
        self.result_dictionnary['pcg_time'] = self.pcg_time

        with open(folder+'results.pkl', 'wb') as handle:
            pickle.dump(self.result_dictionnary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #return self.residues, self.reconstructed_maps


#self = NonLinearCMM(nside=16, nside_beta=8, npointings=100, Nsub=4, frequencies_planck=[100e9, 143e9, 217e9, 353e9],
#                    noise_qubic=0, noise_planck=0, planck_coverage_level=0.2)
#self.map_making(max_iteration=10, pcg_tolerance=1e-16, sigma0=1e-3)


