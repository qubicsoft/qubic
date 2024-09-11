# Vary general importations
import numpy as np
import healpy as hp
import yaml
import os

# Qubic importations
import qubic
from qubic import QubicSkySim as qss
from qubic.lib import Qsamplings as Qs
from qubic.lib.Qacquisition import *
from qubic.lib.Qnoise import *
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

# Local importations
from simtools.mpi_tools import *
#from simtools.noise_timeline import *
from simtools.foldertools import *
from solver.cg import *
import qubic.lib.Qcomponent_model as c

class PresetSims:


    """
    
    Instance to initialize the Components Map-Making. It reads the `params.yml` file to define QUBIC acquisitions.
    
    Arguments : 
    ===========
        - comm    : MPI common communicator (define by MPI.COMM_WORLD).
        - seed    : Int number for CMB realizations.
        - it      : Int number for noise realizations.
        - verbose : bool, Display message or not.
    
    """

    def __init__(self, comm, seed, seed_noise, verbose=True):
        
        self.verbose = verbose
        self.seed_noise = seed_noise
        ### MPI common arguments
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        if self.verbose:
            self._print_message('========= Initialization =========')

        ### Open parameters file
        if self.verbose:
            self._print_message('    => Reading parameters file')
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
            
        ### Define seed for CMB generation and noise
        self.params['CMB']['seed'] = seed
        
        ### Define tolerance of the rms variations
        self.rms_tolerance = self.params['MapMaking']['pcg']['noise_rms_variation_tolerance']
        self.ites_rms_tolerance = self.params['MapMaking']['pcg']['ites_to_converge']
        self.rms_plot = np.zeros((1, 2))
        
        ### Get job id for plots
        self.job_id = os.environ.get('SLURM_JOB_ID')

        ### Create folder for saving data and figures
        if self.rank == 0:
            if self.params['save'] != 0:
                #print(self.params['CMB']['seed'])
                self.params['foldername'] = f"{self.params['Foregrounds']['type']}_{self.params['Foregrounds']['model_d']}_{self.params['MapMaking']['qubic']['type']}_" + self.params['foldername']
                create_folder_if_not_exists(self.params['foldername'])
            if self.params['Plots']['maps'] == True or self.params['Plots']['conv_beta'] == True:
                create_folder_if_not_exists(f'jobs/{self.job_id}/I')
                create_folder_if_not_exists(f'jobs/{self.job_id}/Q')
                create_folder_if_not_exists(f'jobs/{self.job_id}/U')
                create_folder_if_not_exists(f'jobs/{self.job_id}/allcomps')
        
        
        ### QUBIC dictionary
        if self.verbose:
            self._print_message('    => Reading QUBIC dictionary')
        self.dict = self._get_dict()

        ### Skyconfig
        self.skyconfig_in = self._get_sky_config(key='in')
        self.skyconfig_out = self._get_sky_config(key='out')
        
        ### Define model for reconstruction
        if self.verbose:
            self._print_message('    => Creating model')
            
        self.comps_in, self.comps_name_in = self._get_components_fgb(key='in')
        self.comps_out, self.comps_name_out = self._get_components_fgb(key='out')

        ### Center of the QUBIC patch
        self.center = Qs.equ2gal(self.dict['RA_center'], self.dict['DEC_center'])

        ### External frequencies
        self.external_nus = self._get_external_nus()
        
        ### Joint acquisition
        if self.params['Foregrounds']['CO_in']:
            self.nu_co = self.params['Foregrounds']['nu0_co']
        else:
            self.nu_co = None
        
        if self.verbose:
            self._print_message('    => Creating acquisition')
            
        ### Joint acquisition for QUBIC operator
        self.joint_in = JointAcquisitionComponentsMapMaking(self.dict, 
                                                         self.params['MapMaking']['qubic']['type'], 
                                                         self.comps_in, 
                                                         self.params['MapMaking']['qubic']['nsub'],
                                                         self.external_nus,
                                                         self.params['MapMaking']['planck']['nintegr'],
                                                         nu_co=self.nu_co,
                                                         ef150=self.params['MapMaking']['qubic']['duration_150'],
                                                         ef220=self.params['MapMaking']['qubic']['duration_220'])
        
        if self.params['MapMaking']['qubic']['nsub'] == self.params['MapMaking']['qubic']['nsub_out']:
            self.joint_out = JointAcquisitionComponentsMapMaking(self.dict, 
                                                         self.params['MapMaking']['qubic']['type'], 
                                                         self.comps_out, 
                                                         self.params['MapMaking']['qubic']['nsub'],
                                                         self.external_nus,
                                                         self.params['MapMaking']['planck']['nintegr'],
                                                         nu_co=self.nu_co,
                                                         H=self.joint_in.qubic.H,
                                                         ef150=self.params['MapMaking']['qubic']['duration_150'],
                                                         ef220=self.params['MapMaking']['qubic']['duration_220'])
        else:
            self.joint_out = JointAcquisitionComponentsMapMaking(self.dict, 
                                                         self.params['MapMaking']['qubic']['type'], 
                                                         self.comps_out, 
                                                         self.params['MapMaking']['qubic']['nsub_out'],
                                                         self.external_nus,
                                                         self.params['MapMaking']['planck']['nintegr'],
                                                         nu_co=self.nu_co,
                                                         H=None,
                                                         ef150=self.params['MapMaking']['qubic']['duration_150'],
                                                         ef220=self.params['MapMaking']['qubic']['duration_220'])
        
        ### Compute coverage map
        self.coverage = self.joint_out.qubic.coverage
        self.pixmax = np.where(self.coverage == self.coverage.max())[0][0]
        
        self.seenpix_qubic = self.coverage/self.coverage.max() > 0
        self.seenpix_BB = self.coverage/self.coverage.max() > 0.3
        #self.seenpix_analysis = self.coverage/self.coverage.max() > 0.2
        self.seenpix = self.coverage/self.coverage.max() > self.params['MapMaking']['planck']['thr']
        self.coverage_cut = self.coverage.copy()
        self.coverage_cut[~self.seenpix] = 1
        self.fsky = self.seenpix.astype(float).sum() / self.seenpix.size
        #print(self.coverage.size, self.fsky)
        #stop
        self.seenpix_plot = self.coverage/self.coverage.max() > self.params['MapMaking']['planck']['thr']
        if self.params['Foregrounds']['nside_fit'] != 0:
            self.seenpix_beta = hp.ud_grade(self.seenpix, self.params['Foregrounds']['nside_fit'])
        
        ### Compute true components
        if self.verbose:
            self._print_message('    => Creating components')
        self.components_in, self.components_conv_in, _ = self._get_components(self.skyconfig_in)
        self.components_out, self.components_conv_out, self.components_iter = self._get_components(self.skyconfig_out)
        
        ### Get input spectral index
        if self.verbose:
            self._print_message('    => Reading spectral indices')
        self._get_beta_input()
        
        ### Mask for weight Planck data
        self.mask = np.ones(12*self.params['MapMaking']['qubic']['nside']**2)
        self.mask[self.seenpix] = self.params['MapMaking']['planck']['kappa']
        
        self.mask_beta = np.ones(12*self.params['MapMaking']['qubic']['nside']**2)

        if self.params['Foregrounds']['nside_fit'] != 0:
            self.coverage_beta = self.get_coverage()#hp.ud_grade(self.seenpix_qubic, self.params['Foregrounds']['nside_fit'])#self.get_coverage()
        else:
            self.coverage_beta = None
        
        C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['planck']['fwhm_kappa'], lmax=2*self.params['MapMaking']['qubic']['nside'])
        self.mask = C(self.mask)
        self.mask_beta = C(self.mask_beta)
        
        pixsnum_seenpix = np.where(self.seenpix)[0]
        centralpix = hp.ang2pix(self.params['MapMaking']['qubic']['nside'], self.center[0],self.center[1],lonlat=True)
        self.angmax = np.max(qss.get_angles(centralpix,pixsnum_seenpix,self.params['MapMaking']['qubic']['nside']))
        
        ### Inverse noise-covariance matrix
        self.invN = self.joint_out.get_invntt_operator(mask=self.mask)
        self.invN_beta = self.joint_out.get_invntt_operator(mask=self.mask_beta)
        
        ### Preconditionning
        self._get_preconditionner()
        
        ### Convolutions
        self._get_convolution()
        
        ### Get observed data
        if self.verbose:
            self._print_message('    => Getting observational data')
        self._get_tod()
        
        ### Compute initial guess for PCG
        if self.verbose:
            self._print_message('    => Initialize starting point')
        self._get_x0() 
        
        if self.verbose:
            self.display_simulation_configuration()   
    def _get_preconditionner(self):
        
        if self.params['Foregrounds']['nside_fit'] == 0:
            conditionner = np.ones((len(self.comps_out), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
        else:
            conditionner = np.zeros((3, 12*self.params['MapMaking']['qubic']['nside']**2, len(self.comps_out)))
            
        for i in range(conditionner.shape[0]):
            for j in range(conditionner.shape[2]):
                conditionner[i, self.seenpix_qubic, j] = 1/self.coverage[self.seenpix_qubic]
                
        if len(self.comps_name_out) > 2:
            if self.params['Foregrounds']['nside_fit'] == 0:
                conditionner[2:, :, :] = 1
            else:
                conditionner[:, :, 2:] = 1
                
        if self.params['MapMaking']['planck']['fixpixels']:
            conditionner = conditionner[:, self.seenpix_qubic, :]
            
        if self.params['MapMaking']['planck']['fixI']:
            conditionner = conditionner[:, :, 1:]
        
        self.M = get_preconditioner(conditionner)
    def display_simulation_configuration(self):
        
        if self.rank == 0:
            print('******************** Configuration ********************\n')
            print('    - Sky In :')
            print(f"        CMB : {self.params['CMB']['cmb']}")
            print(f"        Dust : {self.params['Foregrounds']['Dust_in']} - {self.params['Foregrounds']['model_d']}")
            print(f"        Synchrotron : {self.params['Foregrounds']['Synchrotron_in']} - {self.params['Foregrounds']['model_s']}")
            print(f"        CO : {self.params['Foregrounds']['CO_in']}\n")
            print('    - Sky Out :')
            print(f"        CMB : {self.params['CMB']['cmb']}")
            print(f"        Dust : {self.params['Foregrounds']['Dust_out']} - {self.params['Foregrounds']['model_d']}")
            print(f"        Synchrotron : {self.params['Foregrounds']['Synchrotron_out']} - {self.params['Foregrounds']['model_s']}")
            print(f"        CO : {self.params['Foregrounds']['CO_out']}\n")
            if self.params['Foregrounds']['type'] == 'parametric':
                print(f"    - Parametric :")
                print(f"        Nside_pix : {self.params['Foregrounds']['nside_pix']}")
                print(f"        Nside_fit : {self.params['Foregrounds']['nside_fit']}\n")
            elif self.params['Foregrounds']['type'] == 'blind':
                print(f"    - Blind\n")
            else:
                raise TypeError(f"{self.params['Foregrounds']['type']} method is not yet implemented")
            print('    - QUBIC :')
            print(f"        Npointing : {self.params['MapMaking']['qubic']['npointings']}")
            print(f"        Nsub : {self.params['MapMaking']['qubic']['nsub']}")
            print(f"        Ndet : {self.params['MapMaking']['qubic']['ndet']}")
            print(f"        Npho150 : {self.params['MapMaking']['qubic']['npho150']}")
            print(f"        Npho220 : {self.params['MapMaking']['qubic']['npho220']}")
            print(f"        RA : {self.params['MapMaking']['sky']['RA_center']}")
            print(f"        DEC : {self.params['MapMaking']['sky']['DEC_center']}")
            if self.params['MapMaking']['qubic']['type'] == 'two':
                print(f"        Type : Dual Bands")
            else:
                print(f"        Type : Ultra Wide Band")
            print(f"        MPI Tasks : {self.size}")
    def _angular_distance(self, pix):
        
        
        theta1, phi1 = hp.pix2ang(self.params['Foregrounds']['nside_fit'], pix)
        pixmax = hp.vec2pix(uvcenter, lonlat=True)
        thetamax, phimax = hp.pix2ang(self.params['Foregrounds']['nside_fit'], pixmax)
        #center = qubic.equ2gal(self.params['MapMaking']['sky']['RA_center'], self.params['MapMaking']['sky']['DEC_center'])
        
        dist = hp.rotator.angdist((thetamax, phimax), (theta1, phi1))
        return dist
    def get_coverage(self):
        
        center = qubic.equ2gal(self.params['MapMaking']['sky']['RA_center'], self.params['MapMaking']['sky']['DEC_center'])
        uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
        uvpix = np.array(hp.pix2vec(self.params['Foregrounds']['nside_fit'], np.arange(12*self.params['Foregrounds']['nside_fit']**2)))
        ang = np.arccos(np.dot(uvcenter, uvpix))
        indices = np.argsort(ang)
        
        mask = np.zeros(12*self.params['Foregrounds']['nside_fit']**2)
        #sorted_indices = np.argsort(_ang_dist)
        okpix = indices[:self.params['Foregrounds']['npix_fit']]
        mask[okpix] = 1
        #uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
        #uvpix = np.array(hp.pix2vec(self.params['Foregrounds']['nside_fit'], np.arange(12*self.params['Foregrounds']['nside_fit']**2)))
        #ang = np.arccos(np.dot(uvcenter, uvpix))
        #indices = np.argsort(ang)
        #okpix = ang < -1
        #okpix[indices[0:int(fsky * 12*nside**2)]] = True
        #mask = np.zeros(12*nside**2)
        #
        return mask
    def _get_noise(self):

        """
        
        Method to define QUBIC noise, can generate Dual band or Wide Band noise by following :

            - Dual Band : n = [Ndet + Npho_150, Ndet + Npho_220]
            - Wide Band : n = [Ndet + Npho_150 + Npho_220]

        """

        if self.params['MapMaking']['qubic']['type'] == 'wide':
            noise = QubicWideBandNoise(self.dict, 
                                       self.params['MapMaking']['qubic']['npointings'], 
                                       detector_nep=self.params['MapMaking']['qubic']['detector_nep'],
                                       duration=np.mean([self.params['MapMaking']['qubic']['duration_150'], self.params['MapMaking']['qubic']['duration_220']]))
        else:
            noise = QubicDualBandNoise(self.dict, 
                                       self.params['MapMaking']['qubic']['npointings'], 
                                       detector_nep=self.params['MapMaking']['qubic']['detector_nep'],
                                       duration=[self.params['MapMaking']['qubic']['duration_150'], self.params['MapMaking']['qubic']['duration_220']])

        return noise.total_noise(self.params['MapMaking']['qubic']['ndet'], 
                                 self.params['MapMaking']['qubic']['npho150'], 
                                 self.params['MapMaking']['qubic']['npho220'],
                                 seed_noise=self.seed_noise).ravel()
    def _get_U(self):
        if self.params['Foregrounds']['nside_fit'] == 0:
            U = (
                ReshapeOperator((len(self.comps_name) * sum(self.seenpix_qubic) * 3), (len(self.comps_name), sum(self.seenpix_qubic), 3)) *
                PackOperator(np.broadcast_to(self.seenpix_qubic[None, :, None], (len(self.comps_name), self.seenpix_qubic.size, 3)).copy())
                ).T
        else:
            U = (
                ReshapeOperator((3 * len(self.comps_name) * sum(self.seenpix_qubic)), (3, sum(self.seenpix_qubic), len(self.comps_name))) *
                PackOperator(np.broadcast_to(self.seenpix_qubic[None, :, None], (3, self.seenpix_qubic.size, len(self.comps_name))).copy())
            ).T
        return U
    def _get_tod(self):

        """
        
        Method to define fake observational data from QUBIC. It includes astrophysical foregrounds contamination using `self.beta` and systematics using `self.g`.
        We generate also fake observational data from external experiments. We generate data in the following way : d = H . A . c + n

        Be aware that the data used the MPI communication to use several cores. Full data are stored in `self.TOD_Q_BAND_ALL` where `self.TOD_Q` is a part
        of all the data. The multiprocessing is done by divide the number of detector per process.
        
        """

        self._get_input_gain()
        self.H = self.joint_in.get_operator(beta=self.beta_in, Amm=self.Amm_in, gain=self.g, fwhm=self.fwhm)
        #self.Ho = self.joint_out.get_operator(beta=self.beta_out, Amm=self.Amm_out, gain=self.g, fwhm=self.fwhm)
        
        if self.rank == 0:
            seed_pl = np.random.randint(10000000)
        else:
            seed_pl = None
            
        seed_pl = self.comm.bcast(seed_pl, root=0)
        
        ne = self.joint_in.external.get_noise(seed=seed_pl) * self.params['MapMaking']['planck']['level_planck_noise']
        nq = self._get_noise()
        
        self.TOD_Q = (self.H.operands[0])(self.components_in[:, :, :]) + nq
        #self.TOD_Qo = (self.Ho.operands[0])(self.components_out[:, :, :])
        self.TOD_E = (self.H.operands[1])(self.components_in[:, :, :]) + ne
        
        #plt.figure()
        #plt.plot(self.TOD_Q - self.TOD_Qo)
        #plt.plot(self.TOD_Qo)
        #plt.plot()
        #plt.savefig('tod.png')
        #plt.close()    
        #stop   
        ### Reconvolve Planck data toward QUBIC angular resolution
        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
            _r = ReshapeOperator(self.TOD_E.shape, (len(self.external_nus), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
            maps_e = _r(self.TOD_E)
            C = HealpixConvolutionGaussianOperator(fwhm=self.joint_in.qubic.allfwhm[-1], lmax=2*self.params['MapMaking']['qubic']['nside'])
            for i in range(maps_e.shape[0]):
                maps_e[i] = C(maps_e[i])
            #maps_e[:, ~self.seenpix, :] = 0.
            
            self.TOD_E = _r.T(maps_e)

        self.TOD_obs = np.r_[self.TOD_Q, self.TOD_E] 
    def extra_sed(self, nus, correlation_length):

        
        #if self.rank == 0:
        #    seed = np.random.randint(10000000)
        #else:
        #    seed = None
            
        #seed = self.comm.bcast(seed, root=0)
        np.random.seed(1)
        extra = np.ones(len(nus))
        if self.params['Foregrounds']['model_d'] != 'd6':
            return np.ones(len(nus))
        else:
            for ii, i in enumerate(nus):
                rho_covar, rho_mean = pysm3.models.dust.get_decorrelation_matrix(353.00000001 * u.GHz, 
                                           np.array([i]) * u.GHz, 
                                           correlation_length=correlation_length*u.dimensionless_unscaled)
                #print(i, rho_covar, rho_mean)
                rho_covar, rho_mean = np.array(rho_covar), np.array(rho_mean)
                extra[ii] = rho_mean[:, 0] + rho_covar @ np.random.randn(1)
            #print(extra)
            #stop
            return extra
    def _get_Amm(self, comps, comp_name, nus, beta_d_true=None, beta_s_true=None, init=False):
        if beta_d_true is None:
            beta_d_true = 1.54
        if beta_s_true is None:
            beta_s_true = -3
        nc = len(comps)
        nf = len(nus)
        A = np.zeros((nf, nc))
        
        if self.params['Foregrounds']['model_d'] == 'd6' and init == False:
            extra = self.extra_sed(nus, self.params['Foregrounds']['l_corr'])
        else:
            extra = np.ones(len(nus))

        for inu, nu in enumerate(nus):
            for j in range(nc):
                if comp_name[j] == 'CMB':
                    A[inu, j] = 1.
                elif comp_name[j] == 'Dust':
                    A[inu, j] = comps[j].eval(nu, np.array([beta_d_true]))[0][0] * self.params['Foregrounds']['Ad'] * extra[inu]
                elif comp_name[j] == 'Synchrotron':
                    #print(comps[j].eval(nu), comps[j].eval(nu).shape)
                    A[inu, j] = comps[j].eval(nu)
        return A
    def _spectral_index_mbb(self, nside):

        """
        
        Method to define input spectral indices if the d1 model is used for thermal dust description.
        
        """

        sky = pysm3.Sky(nside=nside, preset_strings=['d1'])
        return np.array(sky.components[0].mbb_index)
    def _spectral_index_pl(self, nside):

        """
        
        Method to define input spectral indices if the s1 model is used for synchrotron description.
        
        """

        sky = pysm3.Sky(nside=nside, preset_strings=['s1'])
        return np.array(sky.components[0].pl_index)
    def _get_beta_input(self):

        """
        
        Method to define the input spectral indices. If the model is d0, the input is 1.54, if not the model assumes varying spectral indices across the sky
        by calling the previous method. In this case, the shape of beta is (Nbeta, Ncomp).
        
        """
        
        self.nus_eff_in = np.array(list(self.joint_in.qubic.allnus) + list(self.joint_in.external.allnus))
        self.nus_eff_out = np.array(list(self.joint_out.qubic.allnus) + list(self.joint_out.external.allnus))
        
        if self.params['Foregrounds']['type'] == 'parametric':
            self.Amm_in = None
            self.Amm_out = None
            if self.params['Foregrounds']['model_d'] == 'd0':
                self.beta_in = np.array([float(i._REF_BETA) for i in self.comps_in[1:]])
                self.beta_out = np.array([float(i._REF_BETA) for i in self.comps_out[1:]])
            elif self.params['Foregrounds']['model_d'] == 'd1':
                self.beta_in = np.zeros((12*self.params['Foregrounds']['nside_pix']**2, len(self.comps_in)-1))
                self.beta_out = np.zeros((12*self.params['Foregrounds']['nside_fit']**2, len(self.comps_out)-1))
                for iname, name in enumerate(self.comps_name_in):
                    if name == 'Dust':
                        self.beta_in[:, iname-1] = self._spectral_index_mbb(self.params['Foregrounds']['nside_pix'])
                    elif name == 'CMB':
                        pass
                    elif name == 'Synchrotron':
                        self.beta_in[:, iname-1] = self._spectral_index_pl(self.params['Foregrounds']['nside_pix'])
                    else:
                        raise TypeError(f'{name} is not implemented..')
                
                for iname, name in enumerate(self.comps_name_out):
                    if name == 'Dust':
                        self.beta_out[:, iname-1] = self._spectral_index_mbb(self.params['Foregrounds']['nside_fit'])
                    elif name == 'CMB':
                        pass
                    elif name == 'Synchrotron':
                        self.beta_out[:, iname-1] = self._spectral_index_pl(self.params['Foregrounds']['nside_fit'])
                    else:
                        raise TypeError(f'{name} is not implemented..')
            elif self.params['Foregrounds']['model_d'] == 'd6':
                
                self.Amm_in = self._get_Amm(self.comps_in, self.comps_name_in, self.nus_eff_in, init=False)
                #self.Amm_in[:2*self.joint_in.qubic.Nsub] = self._get_Amm(self.comps_in, self.comps_name_in, self.nus_eff, init=False)[:2*self.joint_in.qubic.Nsub]
                #print(self.Amm_in.shape)
                #stop
                #self.Amm_in[2*self.joint_in.qubic.Nsub:] = self._get_Amm(self.comps_in, self.comps_name_in, self.nus_eff, init=True)[2*self.joint_in.qubic.Nsub:]
                self.Amm_out = self._get_Amm(self.comps_out, self.comps_name_out, self.nus_eff_out, init=False)
                self.beta_in = np.array([float(i._REF_BETA) for i in self.comps_in[1:]])
                self.beta_out = np.array([float(i._REF_BETA) for i in self.comps_out[1:]])
                
        elif self.params['Foregrounds']['type'] == 'blind':
            self.beta_in = np.array([float(i._REF_BETA) for i in self.comps_in[1:]])
            self.beta_out = np.array([float(i._REF_BETA) for i in self.comps_out[1:]])
            self.Amm_in = self._get_Amm(self.comps_in, self.comps_name_in, self.nus_eff_in, init=False)
            self.Ammtrue = self._get_Amm(self.comps_out, self.comps_name_out, self.nus_eff_out, init=False)

            self.Amm_in[len(self.joint_in.qubic.allnus):] = self._get_Amm(self.comps_in, self.comps_name_in, self.nus_eff_in, init=True)[len(self.joint_in.qubic.allnus):]
            self.Amm_out = self._get_Amm(self.comps_out, self.comps_name_out, self.nus_eff_out, init=True)
            #print(self.Amm_in)
            #print(self.Amm_out)
            #stop
        else:
            raise TypeError(f"method {self.params['Foregrounds']['type']} is not yet implemented..")
        
        '''
        if self.params['Foregrounds']['Dust_in'] is False:
            self.beta = np.array([])
            self.Amm = None
        elif self.params['Foregrounds']['model_d'] == 'd0':
            self.Amm = self._get_Amm(self.nus_eff)
            if self.params['Foregrounds']['nside_fit'] == 0:
                self.beta = np.array([float(i._REF_BETA) for i in self.comps_in[1:]])
            else:
                self.beta = np.array([[1.54]*(12*self.params['Foregrounds']['nside_pix']**2)]).T
        elif self.params['Foregrounds']['model_d'] == 'd1':
            self.Amm = None
            self.beta = np.array([self._spectral_index()]).T
        elif self.params['Foregrounds']['model_d'] == 'd6':
            self.Amm = self._get_Amm(self.nus_eff, init=True)
            self.Amm[:2*self.joint_in.qubic.Nsub] = self._get_Amm(self.nus_eff[:2*self.joint_in.qubic.Nsub], 
                                                                    beta_d_true=self.params['Foregrounds']['beta_d_init'], 
                                                                    beta_s_true=self.params['Foregrounds']['beta_s_init'], 
                                                                    init=False)
            self.beta = np.array([i._REF_BETA for i in self.comps_in[1:]])
        else:
            raise TypeError(f"model {self.params['Foregrounds']['model_d']} is not yet implemented..")
        '''    
    def _get_components(self, skyconfig):

        """
        
        Read configuration dictionary which contains every components adn the model. 
        
        Example : d = {'cmb':42, 'dust':'d0', 'synchrotron':'s0'}

        The CMB is randomly generated fram specific seed. Astrophysical foregrounds come from PySM 3.
        
        """
        
        components = np.zeros((len(skyconfig), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
        components_conv = np.zeros((len(skyconfig), 12*self.params['MapMaking']['qubic']['nside']**2, 3))
        
        if self.params['MapMaking']['qubic']['convolution'] or self.params['MapMaking']['qubic']['fake_convolution']:
            C = HealpixConvolutionGaussianOperator(fwhm=self.joint_in.qubic.allfwhm[-1], lmax=2*self.params['MapMaking']['qubic']['nside'])
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)
            
        mycls = give_cl_cmb(r=self.params['CMB']['r'], 
                            Alens=self.params['CMB']['Alens'])

        for k, kconf in enumerate(skyconfig.keys()):
            if kconf == 'cmb':

                np.random.seed(skyconfig[kconf])
                cmb = hp.synfast(mycls, self.params['MapMaking']['qubic']['nside'], verbose=False, new=True).T
                components[k] = cmb.copy()
                components_conv[k] = C(cmb).copy()
            
            elif kconf == 'dust':
                
                
                sky=pysm3.Sky(nside=self.params['MapMaking']['qubic']['nside'], 
                              preset_strings=[self.params['Foregrounds']['model_d']], 
                              output_unit="uK_CMB")
                
                sky.components[0].mbb_temperature = 20*sky.components[0].mbb_temperature.unit
                map_dust = np.array(sky.get_emission(self.params['Foregrounds']['nu0_d'] * u.GHz, None).T * \
                                  utils.bandpass_unit_conversion(self.params['Foregrounds']['nu0_d']*u.GHz, None, u.uK_CMB))
                components[k] = map_dust.copy()
                components_conv[k] = C(map_dust).copy()
                    

            elif kconf == 'synchrotron':

                sky = pysm3.Sky(nside=self.params['MapMaking']['qubic']['nside'], 
                                preset_strings=[self.params['Foregrounds']['model_s']], 
                                output_unit="uK_CMB")
                
                map_sync = np.array(sky.get_emission(self.params['Foregrounds']['nu0_s'] * u.GHz, None).T * \
                                utils.bandpass_unit_conversion(self.params['Foregrounds']['nu0_s'] * u.GHz, None, u.uK_CMB)) * self.params['Foregrounds']['As']
                components[k] = map_sync.copy() 
                components_conv[k] = C(map_sync).copy()
                
            elif kconf == 'coline':
                
                m = hp.ud_grade(hp.read_map('data/CO_line.fits') * 10, self.params['MapMaking']['qubic']['nside'])
                mP = polarized_I(m, self.params['MapMaking']['qubic']['nside'])
                myco = np.zeros((12*self.params['MapMaking']['qubic']['nside']**2, 3))
                myco[:, 0] = m.copy()
                myco[:, 1:] = mP.T.copy()
                components[k] = myco.copy()
                components_conv[k] = C(myco).copy()
            else:
                raise TypeError('Choose right foreground model (d0, s0, ...)')
        
        if self.params['Foregrounds']['nside_fit'] == 0:
            components_iter = components.copy()
        else:
            components = components.T.copy()
            components_iter = components.copy() 
        
        if self.params['MapMaking']['qubic']['noise_only']:
            components *= 0
            components_conv *= 0
            components_iter *= 0
        
        return components, components_conv, components_iter
    def _get_ultrawideband_config(self):
        

        """
        
        Method to simply define Ultra Wide Band configuration.
        
        """
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
    
        return nu_ave, 2*delta/nu_ave
    def _get_dict(self):
    
        """

        Method to define and modify the QUBIC dictionary.
        
        """

        nu_ave, delta_nu_over_nu = self._get_ultrawideband_config()

        args = {'npointings':self.params['MapMaking']['qubic']['npointings'], 
                'nf_recon':1, 
                'nf_sub':self.params['MapMaking']['qubic']['nsub'], 
                'nside':self.params['MapMaking']['qubic']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['MapMaking']['sky']['RA_center'], 
                'DEC_center':self.params['MapMaking']['sky']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'kind':'IQU',
                'config':'FI',
                'verbose':False,
                'dtheta':self.params['MapMaking']['qubic']['dtheta'],
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':self.params['MapMaking']['qubic']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(self.params['MapMaking']['qubic']['detector_nep']), 
                'synthbeam_kmax':self.params['MapMaking']['qubic']['synthbeam_kmax']}

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubicDict()
        d.read_from_file(dictfilename)
        for i in args.keys():
        
            d[str(i)] = args[i]

    
        return d
    def _get_sky_config(self, key):
        
        """
        
        Method to define sky model used by PySM to generate fake sky. It create dictionary like :

                sky = {'cmb':42, 'dust':'d0'}
        
        """
        
        sky = {}
        for ii, i in enumerate(self.params.keys()):

            if i == 'CMB':
                if self.params['CMB']['cmb']:
                    sky['cmb'] = self.params['CMB']['seed']
            else:
                for jj, j in enumerate(self.params['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == f'Dust_{key}':
                        if self.params['Foregrounds'][j]:
                            sky['dust'] = self.params['Foregrounds']['model_d']
                    elif j == f'Synchrotron_{key}':
                        if self.params['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['Foregrounds']['model_s']
                    elif j == f'CO_{key}':
                        if self.params['Foregrounds'][j]:
                            sky['coline'] = 'co2'
        return sky
    def _get_components_fgb(self, key):

        """
        
        Method to define sky model taken form FGBuster code. Note that we add `COLine` instance to define monochromatic description.

        """

        comps = []
        comps_name = []

        if self.params['CMB']['cmb']:
            comps += [c.CMB()]
            comps_name += ['CMB']
            
        if self.params['Foregrounds'][f'Dust_{key}']:
            comps += [c.Dust(nu0=self.params['Foregrounds']['nu0_d'], temp=self.params['Foregrounds']['temp'])]
            comps_name += ['Dust']

        if self.params['Foregrounds'][f'Synchrotron_{key}']:
            comps += [c.Synchrotron(nu0=self.params['Foregrounds']['nu0_s'], beta_pl=-3)]
            comps_name += ['Synchrotron']

        if self.params['Foregrounds'][f'CO_{key}']:
            comps += [c.COLine(nu=self.params['Foregrounds']['nu0_co'], active=False)]
            comps_name += ['CO']
        
        return comps, comps_name
    def _get_external_nus(self):


        """
        
        Method to create python array of external frequencies by reading the `params.yml` file.

        """

        allnus = [30, 44, 70, 100, 143, 217, 353]
        external = []
        for inu, nu in enumerate(allnus):
            if self.params['MapMaking']['planck'][f'{nu:.0f}GHz']:
                external += [nu]

        return external
    def _get_convolution(self):

        """
        
        Method to define all agular resolutions of the instrument at each frequencies. `self.fwhm` are the real angular resolution and `self.fwhm_recon` are the 
        beams used for the reconstruction. 
        
        """
        
        if self.params['MapMaking']['qubic']['convolution']:
            self.fwhm_recon = np.sqrt(self.joint_in.qubic.allfwhm**2 - np.min(self.joint_in.qubic.allfwhm)**2)
            self.fwhm = self.joint_in.qubic.allfwhm
        elif self.params['MapMaking']['qubic']['fake_convolution']:
            self.fwhm_recon = None
            self.fwhm = np.ones(len(self.joint_in.qubic.allfwhm)) * self.joint_in.qubic.allfwhm[-1]
        else:
            self.fwhm_recon = None
            self.fwhm = None      
    def _get_input_gain(self):

        """
        
        Method to define gain detector of QUBIC focal plane. It is a random generation following normal law. Note that `self.g` contains gain for the i-th process
        that contains few detectors, all the gain are stored in `self.G`.
        
        """
        
        np.random.seed(None)
        if self.params['MapMaking']['qubic']['type'] == 'wide':
            self.g = np.random.normal(1, self.params['MapMaking']['qubic']['sig_gain'], self.joint_in.qubic.ndets)
            #self.g = np.random.uniform(1, 1 + self.params['MapMaking']['qubic']['sig_gain'], self.joint_in.qubic.ndets)#np.random.random(self.joint_in.qubic.ndets) * self.params['MapMaking']['qubic']['sig_gain'] + 1
            #self.g /= self.g[0]
        else:
            self.g = np.random.normal(1, self.params['MapMaking']['qubic']['sig_gain'], (self.joint_in.qubic.ndets, 2))
            #self.g = np.random.uniform(1, 1 + self.params['MapMaking']['qubic']['sig_gain'], (self.joint_in.qubic.ndets, 2))#self.g = np.random.random((self.joint_in.qubic.ndets, 2)) * self.params['MapMaking']['qubic']['sig_gain'] + 1
            #self.g /= self.g[0]
        #print(self.g)
        #stop
        self.G = join_data(self.comm, self.g)
        if self.params['MapMaking']['qubic']['fit_gain']:
            g_err = 0.2
            self.g_iter = np.random.uniform(self.g - g_err/2, self.g + g_err/2, self.g.shape)
            #self.g_iter = self.g + np.random.normal(0, self.g*0.2, self.g.shape)
            #self.g_iter = self.g + np.random.normal(0, self.g*0.2, self.g.shape)
        else:
            self.g_iter = np.ones(self.g.shape)
        self.Gi = join_data(self.comm, self.g_iter)
        self.allg = np.array([self.g_iter])
    def _get_x0(self):

        """
        
        Method to define starting point of the convergence. The argument 'set_comp_to_0' multiply the pixels values by a given factor. You can decide 
        to convolve also the map by a beam with an fwhm in radians.
        
        """
            
        if self.rank == 0:
            seed = np.random.randint(100000000)
        else:
            seed = None
        
        seed = self.comm.bcast(seed, root=0)
        np.random.seed(seed)
        
        if self.params['Foregrounds']['type'] == 'parametric':
            if self.params['Foregrounds']['nside_fit'] == 0:
                self.beta_iter = self.beta_out.copy()
                self.beta_iter += np.random.normal(0., self.params['MapMaking']['initial']['sig_beta_x0'], len(self.beta_iter))
                #self.beta_iter[0] = 1.54
                self.Amm_iter = None
                self.allAmm_iter = None
        
            else:
                self.Amm_iter = None
                self.allAmm_iter = None
                self.beta_iter = self.beta_out.copy()
                _index_seenpix_beta = np.where(self.coverage_beta == 1)[0]
                self.beta_iter[_index_seenpix_beta, 0] += np.random.normal(0, 
                                                                   self.params['MapMaking']['initial']['sig_beta_x0'], 
                                                                   _index_seenpix_beta.shape)
                
        elif self.params['Foregrounds']['type'] == 'blind':
            if self.params['Foregrounds']['Dust_out']:
                beta_d_init = self.params['Foregrounds']['beta_d_init']
            else:
                beta_d_init = None
            
            if self.params['Foregrounds']['Synchrotron_out']:
                beta_s_init = self.params['Foregrounds']['beta_s_init']
            else:
                beta_s_init = None
            self.beta_iter = np.array([1.54])
            self.Amm_iter = self.Amm_out.copy()
            self.allAmm_iter = np.array([self.Amm_iter]) 
            
        else:
            raise TypeError(f"{self.params['Foregrounds']['type']} is not yet implemented")

        if self.params['Foregrounds']['nside_fit'] == 0:
            self.allbeta = np.array([self.beta_iter])
            ### Constant spectral index -> maps have shape (Ncomp, Npix, Nstk)
            for i in range(len(self.comps_out)):
                if self.comps_name_out[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_cmb_to_0']
                    self.components_iter[i, self.seenpix, 1:] *= self.params['MapMaking']['initial']['qubic_patch_cmb']

                elif self.comps_name_out[i] == 'Dust':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_dust_to_0']
                    self.components_iter[i, self.seenpix, 1:] *= self.params['MapMaking']['initial']['qubic_patch_dust']

                elif self.comps_name_out[i] == 'Synchrotron':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_sync_to_0']
                    self.components_iter[i, self.seenpix, 1:] *= self.params['MapMaking']['initial']['qubic_patch_sync']

                elif self.comps_name_out[i] == 'CO':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[i] = C(self.components_iter[i] + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[i].shape)) * self.params['MapMaking']['initial']['set_co_to_0']
                    self.components_iter[i, self.seenpix, 1:] *= self.params['MapMaking']['initial']['qubic_patch_co']
                else:
                    raise TypeError(f'{self.comps_name_out[i]} not recognize')

        else:
            self.allbeta = np.array([self.beta_iter])
            ### Varying spectral indices -> maps have shape (Nstk, Npix, Ncomp)
            for i in range(len(self.comps_out)):
                if self.comps_name_out[i] == 'CMB':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    #print(self.components_iter.shape)
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T).T * self.params['MapMaking']['initial']['set_cmb_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_cmb']
                    
                elif self.comps_name_out[i] == 'Dust':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_dust_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_dust']
                    
                elif self.comps_name_out[i] == 'Synchrotron':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_sync_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_sync']
                    
                elif self.comps_name_out[i] == 'CO':
                    C = HealpixConvolutionGaussianOperator(fwhm=self.params['MapMaking']['initial']['fwhm_x0'], lmax=2*self.params['MapMaking']['qubic']['nside'])
                    self.components_iter[:, :, i] = C(self.components_iter[:, :, i].T + np.random.normal(0, self.params['MapMaking']['initial']['sig_map'], self.components_iter[:, :, i].T.shape)).T * self.params['MapMaking']['initial']['set_co_to_0']
                    self.components_iter[:, self.seenpix, i] *= self.params['MapMaking']['initial']['qubic_patch_co']
                else:
                    raise TypeError(f'{self.comps_name_out[i]} not recognize') 
        
        if self.params['MapMaking']['qubic']['noise_only']:
            self.components_iter *= 0
        #self.components_iter[:, ~self.seenpix_qubic, :] = 0           
    def _print_message(self, message):

        """
        
        Method that display a `message` only for the first rank because of MPI multiprocessing.
        
        """
        
        if self.rank == 0:
            print(message)

