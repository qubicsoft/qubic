import qubic
import numpy as np
from pyoperators import *
import healpy as hp
import pysm3
import pysm3.units as u
from time import time
import matplotlib.pyplot as plt

from Qacquisition import QubicFullBandSystematic
from non_linear_pcg_preconditioned import non_linear_pcg


class NonLinearCMM:
    def __init__(self, nside, nside_beta, npointings, nf_sub):
        '''
        nside: (int) nside of the pixels in the sky from the module healpy.
        nside_beta: (int) Same but for the spectral indices of the dust.
        npointings: (int) Number of random pointings for the acquisition matrix.
        nf_sub: (int) Number of frequencies considered in the full band.
        '''
        self.nside = nside
        self.nside_beta = nside_beta
        self.npointings = npointings
        self.nf_sub = nf_sub

        self.npixel = 12*self.nside**2
        self.nbeta = 12*self.nside_beta**2

        _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.nf_sub/2)-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.nf_sub/2)-1, relative_bandwidth=0.25)
        self.frequencies = np.concatenate((allnus150, allnus220)) * 1e9

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
        self.dict['nf_sub'] = self.nf_sub
        self.dict['filter_nu'] = nu_ave*1e9
        self.dict['filter_relative_bandwidth'] = delta_nu_over_nu
        self.dict['type_instrument'] = 'wide'
        self.dict['MultiBand'] = True
        self.dict['hwp_stepsize'] = 3
        self.dict['synthbeam_kmax'] = 3
        self.dict['synthbeam_fraction'] = 0.95
        self.dict['random_pointing'] = True
        self.dict['repeat_pointing'] = False

        self.Q = QubicFullBandSystematic(self.dict, Nsub=self.nf_sub, Nrec=2)

        coverage = self.Q.subacqs[0][0].get_coverage()
        self.seenpix_qubic = coverage/coverage.max() > 0.1
        self.seenpix_qubic_beta = hp.ud_grade(self.seenpix_qubic, self.nside_beta)

        self.npixel_patch = np.count_nonzero(self.seenpix_qubic)
        self.nbeta_patch = np.count_nonzero(self.seenpix_qubic_beta)

        rank = 1 - (1 - 1/(6*self.npixel_patch+self.nbeta_patch) * 992/5)**self.npointings
        print(f'You are trying to reconstruct {6*self.npixel_patch+self.nbeta_patch} parameters.')
        print(f'The analytically estimated rank of the acquisition matrix is {rank*100} %.')
        if rank < 0.99:
            print('The ranks is too low! Increase the number of pointings.')
        else:
            print('Make sure you have enough random pointings.')


    def get_acquisition_matrix(self):
        '''
        List of acquisition matrix at the different wavelengths.
        '''
        return self.Q.H

    
    def get_real_sky(self):
        '''
        The components maps and spectral indices that we will try to reconstruct.
        '''
        skycmb = pysm3.Sky(nside=self.nside, preset_strings=['c1'], output_unit='uK_CMB')
        skydust = pysm3.Sky(nside=self.nside, preset_strings=['d1'], output_unit='uK_CMB')
        skycmb = np.array(skycmb.get_emission(self.frequencies[-1] * u.Hz))
        skydust = np.array(skydust.get_emission(self.frequencies[-1] * u.Hz))
        skydust_beta = pysm3.Sky(nside=self.nside_beta, preset_strings=['d1'], output_unit='uK_CMB')
        true_beta = np.array(skydust_beta.components[0].mbb_index)
        return np.concatenate((skycmb[0,:], skycmb[1,:], skycmb[2,:], skydust[0,:], skydust[1,:], skydust[2,:], true_beta))

    
    def patch_operators(self, true_c):
        '''
        Operator Patch_to_Sky takes a vector (components maps + spectral indices map) on the patch and put it on the full sky 
        with Planck data on the pixels that are not observed by Qubic. The operator Sky_to_Patch does the opposite.
        '''
        patch_mask = np.concatenate((np.tile(self.seenpix_qubic, 6), self.seenpix_qubic_beta))

        def patch_to_sky(c, out):
            sky = true_c.copy()
            sky[patch_mask] = c
            out[...] = sky

        Patch_to_Sky = Operator(patch_to_sky, shapein=6*self.npixel_patch+self.nbeta_patch, shapeout=6*self.npixel+self.nbeta, dtype='float64')

        def sky_to_patch(c, out):
            out[...] = c[patch_mask]

        Sky_to_Patch = Operator(sky_to_patch, shapein=6*self.npixel+self.nbeta, shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')

        return Patch_to_Sky, Sky_to_Patch

    
    def modified_black_body(self, freq, beta):
        '''
        The modified black-body-spectrum of the dust. We have: h/(kT) = 2.4 x 10^(-12) Hz^(-1) with T = 20 K.
        '''
        nu0 = self.frequencies[-1]
        return (np.exp(freq * 2.4e-12) - 1) / (np.exp(nu0 * 2.4e-12) - 1) * (freq/nu0)**beta

    
    def get_mixing_operators(self):
        '''
        The mixing operator A_\nu: giving a vector of shape (6*npixel+nbeta), it returns the mixed sky of shape (npixel, 3).
        A_nu(c)[i,:] = (CMB I_i + f(beta_i)dust I_i,  CMB Q_i + f(beta_i)dust Q_i,  CMB U_i + f(beta_i)dust U_i),
        with f the modified blackbody spectrum, and beta_i the value of the spectral index at pixel i 
        (each beta is used for multiple pixels).
        Returns the list of the mixing operators at the different wavelengths.
        '''
        def function_A(c, freq, out):
            power_beta = self.modified_black_body(freq, c[6*self.npixel:])
            up_grade_power_beta = hp.ud_grade(power_beta, self.nside)
        
            out[:,0] = c[:self.npixel] + up_grade_power_beta * c[3*self.npixel:4*self.npixel]
            out[:,1] = c[self.npixel:2*self.npixel] + up_grade_power_beta * c[4*self.npixel:5*self.npixel]
            out[:,2] = c[2*self.npixel:3*self.npixel] + up_grade_power_beta * c[5*self.npixel:6*self.npixel]
        
        A_list = []
        for freq in self.frequencies:
            A_list.append(Operator(lambda c, out, freq=freq : function_A(c, freq, out), 
                                   shapein=6*self.npixel+self.nbeta, shapeout=(self.npixel, 3), dtype='float64'))
            
        return A_list

    
    def get_jacobien_mixing_operators(self):
        '''
        This operator is the transpose of the Jacobian of the mixing matrix. It is a non-linear operator.
        Returns an operator that can be called to generate the transposed Jacobian operator.
        '''
        # Define the inner operator class
        class Transposed_Jacobian(Operator):
            def __init__(self1, c, freq, **keywords):
                self1.c = c
                self1.freq = freq
                super().__init__(shapein=(self.npixel,3), shapeout=6*self.npixel+self.nbeta, dtype='float64', **keywords)
            
            def direct(self1, input_vector, output):
                
                power_beta = self.modified_black_body(self1.freq, self1.c[6*self.npixel:])
                derive_power_beta = power_beta * np.log(self1.freq/self.frequencies[-1])
                
                output[:self.npixel] = input_vector[:,0]
                output[self.npixel:2*self.npixel] = input_vector[:,1]
                output[2*self.npixel:3*self.npixel] = input_vector[:,2]
        
                up_grade_power_beta = hp.ud_grade(power_beta, self.nside)
                output[3*self.npixel:4*self.npixel] = up_grade_power_beta * input_vector[:,0]
                output[4*self.npixel:5*self.npixel] = up_grade_power_beta * input_vector[:,1]
                output[5*self.npixel:6*self.npixel] = up_grade_power_beta * input_vector[:,2]
            
                product = self1.c[3*self.npixel:4*self.npixel]*input_vector[:,0] 
                product += self1.c[4*self.npixel:5*self.npixel]*input_vector[:,1] 
                product += self1.c[5*self.npixel:6*self.npixel]*input_vector[:,2]
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

    
    def get_tod(self, H_list, A_list, true_c):
        '''
        Creating the TOD
        '''
        tod = H_list[0](A_list[0](true_c))
        for i in range(1, len(self.frequencies)):
            tod += H_list[i](A_list[i](true_c))
            
        return tod

    
    def get_grad_chi_squared_operator(self, H_list, A_list, Patch_to_Sky, Sky_to_Patch, tod, generate_transposed_jacobian, true_c):
        '''
        The gradient of $\chi^2$ operator. We have:
        $$\nabla\chi^2(\tilde{c}) = \sum (J_{A_\nu}(\tilde{c}))^TH^TN^{-1} \sum H{A_\nu}(\tilde{c}) - \sum (J_{A_\nu}(\tilde{c}))^TH^TN^{-1}d$$
        '''
        def grad_operator(c, out):
            W = H_list[0](A_list[0](Patch_to_Sky(c)))
            for i in range(1, len(self.frequencies)):
                W += (H_list[i](A_list[i](Patch_to_Sky(c))))
            W -= tod ###### invN(W-tod)
        
            output_operator = np.empty((), dtype=object)
            generate_transposed_jacobian.direct(Patch_to_Sky(c), self.frequencies[0], output_operator)
            transposed_jacobian = output_operator.item()
            X = transposed_jacobian(H_list[0].T(W))
            for i in range(1, len(self.frequencies)):
                generate_transposed_jacobian.direct(Patch_to_Sky(c), self.frequencies[i], output_operator)
                transposed_jacobian = output_operator.item()
                X += transposed_jacobian(H_list[i].T(W))
            
            out[...] = Sky_to_Patch(X)
        
        Grad_chi_squared = Operator(grad_operator, shapein=6*self.npixel_patch+self.nbeta_patch, 
                                    shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')

        return Grad_chi_squared


    def get_preconditioner(self, H_list, Patch_to_Sky):
        '''
        We compute an approximation of the inverse of the diagonal of the hessian matrix of chi^2. 
        This is used as a preconditioner for the non-linear PCG. It is very important as the 
        components maps and the spectral indices have a very different behaviour in the PCG. 
        This preconditioner helps making those different parameters more like one another.
        '''
        def hessian_inverse_diagonal(c, out):
            sky_c = Patch_to_Sky(c)
            dust_spectrum_squared = np.zeros((len(self.frequencies),self.npixel_patch))
            derive_dust_spectrum_squared = np.zeros((len(self.frequencies),self.npixel_patch))
            for index, freq in enumerate(self.frequencies):
                dust_spectrum = hp.ud_grade(self.modified_black_body(freq, sky_c[6*self.npixel:]), self.nside)[self.seenpix_qubic]
                dust_spectrum_squared[index,:] = dust_spectrum**2
                derive_dust_spectrum_squared[index,:] = (dust_spectrum * np.log(freq/self.frequencies[-1]))**2

            vector = np.ones(H_list[0].shapein)
            approx_hth = np.empty((len(H_list), self.npixel_patch)) # has shape (nf_sub, npixel_patch)
            for index in range(len(H_list)):
                approx_hth[index] = (H_list[index].T * H_list[index] * vector)[self.seenpix_qubic, 0]
            
            # CMB
            out[:3*self.npixel_patch] = np.tile(1/np.sum(approx_hth, axis=0), 3)
        
            # Dust
            out[3*self.npixel_patch:6*self.npixel_patch] = np.tile(1/np.sum(approx_hth * dust_spectrum_squared, axis=0), 3)
        
            # Spectral indices
            factor1 = c[3*self.npixel_patch:4*self.npixel_patch]**2
            factor1 += c[4*self.npixel_patch:5*self.npixel_patch]**2
            factor1 += c[5*self.npixel_patch:6*self.npixel_patch]**2
            factor1 = factor1 * approx_hth # has shape (frequencies, npixel_patch)
            factor1 *= derive_dust_spectrum_squared
            factor1 = np.sum(factor1, axis=0) # shape (npixel_patch)
            
            downgrader = np.zeros(self.npixel)
            downgrader[self.seenpix_qubic] = factor1
            downgrader = hp.ud_grade(downgrader, self.nside_beta)*(self.npixel//self.nbeta)
            out[6*self.npixel_patch:] = 1/downgrader[self.seenpix_qubic_beta]
        
        return Operator(hessian_inverse_diagonal, shapein=6*self.npixel_patch+self.nbeta_patch, shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')


    def get_initial_guess(self, Sky_to_Patch, true_c):
        '''
        Set an initial guess for the PCG. For CMB I and dust I, we give the true solution as Planck's measurment
        were good. For CMB Q, U and dust Q, U, we give a map set at 0. FOr the spectral indices, we give a map
        set at 1.53, the mean of the spectral indices on the Qubic's patch.
        '''
        x0 = np.empty(6*self.npixel_patch+self.nbeta_patch)
        x0[:self.npixel_patch] = Sky_to_Patch(true_c)[:self.npixel_patch].copy()
        x0[self.npixel_patch:3*self.npixel_patch] = np.zeros(2*self.npixel_patch)
        x0[3*self.npixel_patch:4*self.npixel_patch] = Sky_to_Patch(true_c)[3*self.npixel_patch:4*self.npixel_patch].copy()
        x0[4*self.npixel_patch:6*self.npixel_patch] = np.zeros(2*self.npixel_patch)
        x0[6*self.npixel_patch:] = np.ones(self.nbeta_patch)*1.53

        return x0


    def plot_residues(self, residues):
        '''
        Plots the evolution of the relative residues.
        '''
        plt.plot(residues[:,0])
        plt.yscale('log')
        plt.grid(axis='y', linestyle='dotted')
        plt.xlabel('Number of iterations')
        plt.ylabel(r'Relative residue $\frac{||\nabla \chi^2(c_{\beta})||}{||\nabla \chi^2(\vec{0})||}$')
        plt.title('Simultaneous reconstruction of components maps and spectral indices using a non-linear PCG')
        plt.show()


    def plot_reconstructed_maps(self, x, x0sky, true_c):
        '''
        Plots the true maps, the initial maps, the reconstructed maps, and the difference 
        between the true maps and the reconstructed maps.
        '''
        name_list = ['CMB I','CMB Q','CMB U','dust I','dust Q','dust U',r'$\beta$']
        
        plt.figure(figsize=(12, 25))
        for i in range(6):
            hp.gnomview(true_c[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+1), title='True '+name_list[i], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
            hp.gnomview(x0sky[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+2), title='Initial '+name_list[i], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
            hp.gnomview(x[i*self.npixel:(i+1)*self.npixel], sub=(7,4,4*i+3), title='Reconstructed '+name_list[i], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
            r = true_c[i*self.npixel:(i+1)*self.npixel] - x[i*self.npixel:(i+1)*self.npixel]
            sig = np.std(r[self.seenpix_qubic])
            hp.gnomview(r, sub=(7,4,4*i+4), title='Difference '+name_list[i], rot=qubic.equ2gal(0, -57),reso=18, min=-2*sig, max=2*sig, cmap='jet')
        hp.gnomview(true_c[6*self.npixel:], sub=(7,4,4*6+1), title='True '+name_list[6], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
        hp.gnomview(x0sky[6*self.npixel:], sub=(7,4,4*6+2), title='Initial '+name_list[6], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
        hp.gnomview(x[6*self.npixel:], sub=(7,4,4*6+3), title='Reconstructed '+name_list[6], rot=qubic.equ2gal(0, -57),reso=18, cmap='jet')
        r = true_c[6*self.npixel:] - x[6*self.npixel:]
        sig = np.std(r[self.seenpix_qubic_beta])
        hp.gnomview(r, sub=(7,4,4*6+4), title='Difference '+name_list[6], rot=qubic.equ2gal(0, -57),reso=18, min=-2*sig, max=2*sig, cmap='jet')
        plt.show()
        


    
    def map_making(self, max_iteration, tol, sigma0, plot_results=True, x0=None, verbose=True):
        '''
        Map-making of the components maps and the sepctral indices map on the Qubic's patch through a non-linear PCG.
        max_iteration : (int) Maximum number of iteratiuon of the PCG.
        tol: (float) Tolerance of the PCG.
        sigma0: (float) Initial guess of the size of the step et each iteration of the PCG. If set incorrectly, 
            the PCG will not run. You have to test.
        plot_resuslts: (bool) If True, plots the evolution of the relative residues, and the maps of 
            different components and spectral indices.
        x0: (array of shape (6*npixel_patch+nbeta_patch)) Initial guess for the PCG. If None, the code
            calls the function get_initial_guess.

        Returns:
        residues: (array of shape (max_iteration, 8)) List of the residues for all the parameters together, 
            the CMB I, Q, U maps, the dust I, Q, U maps, and the spectral indices map.
        x: (array of shape (6*npixel_patch+nbeta_patch)) The reconstructed maps at the end of the PCG.
        '''
        if verbose:
            print('First, all operators have to be defined.')
        self.H_list = self.get_acquisition_matrix()
        self.true_c = self.get_real_sky()
        self.Patch_to_Sky, self.Sky_to_Patch = self.patch_operators(self.true_c)
        self.A_list = self.get_mixing_operators()
        self.generate_transposed_jacobian = self.get_jacobien_mixing_operators()
        self.tod = self.get_tod(self.H_list, self.A_list, self.true_c)
        self.Grad_chi_squared = self.get_grad_chi_squared_operator(self.H_list, self.A_list, self.Patch_to_Sky, self.Sky_to_Patch, self.tod, self.generate_transposed_jacobian, self.true_c)
        self.HessianInverseDiagonal = self.get_preconditioner(self.H_list, self.Patch_to_Sky)

        # Check that the gradient is zero at the solution point
        if verbose:
            print(f'The gradient should be zero at the solution point, this is {(self.Grad_chi_squared(self.Sky_to_Patch(self.true_c))==0).all()}.')

        if x0 is None:
            self.x0 = self.get_initial_guess(self.Sky_to_Patch, self.true_c)
        else:
            self.x0 = x0

        
        if verbose:
            print('All operators have been defined, PCG is starting.')

        start = time()
        residues = []
        pcg = non_linear_pcg(self.Grad_chi_squared, M=self.HessianInverseDiagonal, conjugate_method='polak-ribiere', x0=self.x0, tol=tol, sigma_0=sigma0, tol_linesearch=1e-3, maxiter=max_iteration, residues=residues, npixel_patch=self.npixel_patch, nbeta_patch=self.nbeta_patch)
        x = pcg['x']
        residues = np.array(residues)
        residues /= np.linalg.norm(self.Grad_chi_squared(self.x0))
        if verbose:
            print(f'Time taken for PCG: {time()-start} sec')

        
        if plot_results:
            if verbose:
                print('Plotting the residues and the maps.')
            self.plot_residues(residues)
            print('The scale of the difference maps is set to ± 3 sigma.')
            self.plot_reconstructed_maps(self.Patch_to_Sky(x), self.Patch_to_Sky(self.x0), self.true_c)

        return residues, x






    """ #Attempt at defining a preconditioner, but is to slow to compute.
    def get_coverage(self, H_list):
        '''
        Computation of the coverage at each frequency and for I, Q and U. The coverage of pixel i is the sum over 
        the column i of the operator H of the squares of the elements:
        Cov[\nu, i] = \sum_{\text{det}\times\text{samplings}} (H_\nu [\text{det}\times\text{samplings}, i])^2
        '''
        Cov = np.empty((len(self.frequencies), 3*self.npixel_patch))
        mixed_map_mask = np.concatenate([self.seenpix_qubic,self.seenpix_qubic,self.seenpix_qubic])
        
        for i in range(3*self.npixel_patch):
            patch_vector = np.zeros((self.npixel_patch,3))
            patch_vector[i%self.npixel_patch, i//self.npixel_patch] = 1
            basis_vector = np.zeros((self.npixel,3))
            basis_vector[self.seenpix_qubic, :] = patch_vector
            for freq_index in range(len(self.frequencies)):
                vector_i = H_list[freq_index](basis_vector)
                vector_i = vector_i.ravel()
                Cov[freq_index, i] = np.dot(vector_i, vector_i)
            patch_vector[i%self.npixel_patch, i//self.npixel_patch] = 0
            
        return Cov


    def get_preconditioner(self, Patch_to_Sky, Cov):
        '''
        We compute an approximation of the inverse of the diagonal of the hessian matrix of chi^2. 
        This is used as a preconditioner for the non-linear PCG. It is very important as the 
        components maps and the spectral indices have a very different behaviour in the PCG. 
        This preconditioner helps making those different parameters more like one another.

        For that, we suppose that H^TN^{-1}H is essentialy diagonal and that this diagonal is 
        approximatly the coverage, this means neglecting the effect of N^{-1}.
        '''
        def hessian_inverse_diagonal(c, out):
            sky_c = Patch_to_Sky(c)
            dust_spectrum_squared = np.zeros((len(self.frequencies),self.npixel_patch))
            derive_dust_spectrum_squared = np.zeros((len(self.frequencies),self.npixel_patch))
            for index, freq in enumerate(self.frequencies):
                dust_spectrum = hp.ud_grade(self.modified_black_body(freq, sky_c[6*self.npixel:]), self.nside)[self.seenpix_qubic]
                dust_spectrum_squared[index,:] = dust_spectrum**2
                derive_dust_spectrum_squared[index,:] = (dust_spectrum * np.log(freq/self.frequencies[-1]))**2
        
            # CMB
            out[:3*self.npixel_patch] = 1/np.sum(Cov, axis=0)
        
            # Dust
            out[3*self.npixel_patch:6*self.npixel_patch] = 1/np.sum(np.concatenate((dust_spectrum_squared,
                                                                        dust_spectrum_squared,dust_spectrum_squared),1)*Cov, axis=0)
        
            # Spectral indices
            # factor 1 has shape (frequencies, npixel_patch)
            factor1 = c[3*self.npixel_patch:4*self.npixel_patch]**2 * Cov[:,:self.npixel_patch]
            factor1 += c[4*self.npixel_patch:5*self.npixel_patch]**2 * Cov[:,self.npixel_patch:2*self.npixel_patch]
            factor1 += c[5*self.npixel_patch:6*self.npixel_patch]**2 * Cov[:,2*self.npixel_patch:3*self.npixel_patch]
            factor1 *= derive_dust_spectrum_squared
            factor1 = np.sum(factor1, axis=0) # shape (npixel_patch)
            
            downgrader = np.zeros(self.npixel)
            downgrader[self.seenpix_qubic] = factor1
            downgrader = hp.ud_grade(downgrader, self.nside_beta)*(self.npixel//self.nbeta)
            out[6*self.npixel_patch:] = 1/downgrader[self.seenpix_qubic_beta]
        
        return Operator(hessian_inverse_diagonal, shapein=6*self.npixel_patch+self.nbeta_patch, shapeout=6*self.npixel_patch+self.nbeta_patch, dtype='float64')
    """














