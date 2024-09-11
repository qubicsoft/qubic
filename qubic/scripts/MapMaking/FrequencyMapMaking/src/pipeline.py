# Very general packages 
import numpy as np
import yaml
import pickle
import time
import healpy as hp
import os

# Qubic libraries
import qubic
from qubic.lib import Qacquisition as acq
from qubic.lib import Qnoise as Qn
from qubic.lib import Qsamplings
from qubic.lib import Qqubicdict
from qubic import NamasterLib as nam
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pyoperators import MPI

# Local importation
from model.models import *
from likelihood.likelihood import *
from plots.plotter import *
from mapmaking.planck_timeline import *
from model.externaldata import *
from tools.foldertools import *
from fgb.component_model import *
from tools.cg import pcg
from spectrum.spectra import Spectrum



def save_pkl(name, d):
    with open(name, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
__all__ = ['PipelineFrequencyMapMaking', 
           'PipelineEnd2End']
  
class PipelineFrequencyMapMaking:

    """
    
    Instance to reconstruct frequency maps using QUBIC abilities.
    
    Parameters :
    ------------
        - comm : MPI communicator
        - file : str to create folder for data saving
    
    """
    
    def __init__(self, comm, file):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.file = file
        self.externaldata = PipelineExternalData(file)
        self.externaldata.run()
        self.externaldata_noise = PipelineExternalData(file, noise_only=True)
        self.externaldata_noise.run()
        #print(self.externaldata.maps.shape)
        #stop
        self.job_id = os.environ.get('SLURM_JOB_ID')
        
        ### Initialize plot instance
        self.plots = PlotsMM(self.params)

        
        self.center = Qsamplings.equ2gal(self.params['QUBIC']['RA_center'], self.params['QUBIC']['DEC_center'])
        self.fsub = int(self.params['QUBIC']['nsub'] / self.params['QUBIC']['nrec'])

        ### MPI common arguments
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        ### Sky
        self.dict, self.dict_mono = self.get_dict()
        self.skyconfig = self._get_sky_config()
        
        ### Joint acquisition
        self.joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nrec'], self.params['QUBIC']['nsub'])
        self.planck_acquisition143 = acq.PlanckAcquisition(143, self.joint.qubic.scene)
        self.planck_acquisition217 = acq.PlanckAcquisition(217, self.joint.qubic.scene)
        self.nus_Q = self._get_averaged_nus()

        ### Joint acquisition for TOD making
        self.joint_tod = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nsub'], self.params['QUBIC']['nsub'])

        ### Coverage map
        self.coverage = self.joint.qubic.subacqs[0].get_coverage()
        covnorm = self.coverage / self.coverage.max()
        self.seenpix = covnorm > self.params['QUBIC']['covcut']
        self.fsky = self.seenpix.astype(float).sum() / self.seenpix.size
        self.coverage_cut = self.coverage.copy()
        self.coverage_cut[~self.seenpix] = 1

        self.seenpix_for_plot = covnorm > 0
        self.mask = np.ones(12*self.params['Sky']['nside']**2)
        self.mask[self.seenpix] = self.params['QUBIC']['kappa']
        
        
        ### Angular resolutions
        self.targets, self.allfwhm = self._get_convolution()
        
        self.external_timeline = ExternalData2Timeline(self.skyconfig, 
                                                       self.joint.qubic.allnus, 
                                                       self.params['QUBIC']['nrec'], 
                                                       nside=self.params['Sky']['nside'], 
                                                       corrected_bandpass=self.params['QUBIC']['bandpass_correction'])

        ### Define reconstructed and TOD operator
        self._get_H()
        ### Inverse noise covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)

        ### Noises
        
        seed_noise_planck = self._get_random_value()
        print('seed_noise_planck', seed_noise_planck)
        
        self.noise143 = self.planck_acquisition143.get_noise(seed_noise_planck) * self.params['Data']['level_planck_noise']
        self.noise217 = self.planck_acquisition217.get_noise(seed_noise_planck+1) * self.params['Data']['level_planck_noise']

        if self.params['QUBIC']['type'] == 'two':
            qubic_noise = Qn.QubicDualBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])
        elif self.params['QUBIC']['type'] == 'wide':
            qubic_noise = Qn.QubicWideBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])

        self.noiseq = qubic_noise.total_noise(self.params['QUBIC']['ndet'], 
                                       self.params['QUBIC']['npho150'], 
                                       self.params['QUBIC']['npho220'],
                                       seed_noise=seed_noise_planck).ravel()

    def _get_random_value(self):
        
        np.random.seed(None)
        if self.rank == 0:
            seed = np.random.randint(10000000)
        else:
            seed = None
            
        seed = self.comm.bcast(seed, root=0)
        return seed
    def _get_H(self):
        
        """
        
        Method to compute QUBIC operators.
        
        """
        
        self.H = self.joint.get_operator(fwhm=self.targets)
        self.Htod = self.joint_tod.get_operator(fwhm=self.allfwhm)
        self.Hqtod = self.joint_tod.qubic.get_operator(fwhm=self.allfwhm)  
    def _get_averaged_nus(self):
        
        """
        
        Method to average QUBIC frequencies.

        """
        
        nus_eff = []
        for i in range(self.params['QUBIC']['nrec']):
            nus_eff += [np.mean(self.joint.qubic.allnus[i*self.fsub:(i+1)*self.fsub])]
        
        return np.array(nus_eff)
    def _get_sky_config(self):
        
        """
        
        Method that read `params.yml` file and create dictionary containing sky emission such as :
        
                    d = {'cmb':seed, 'dust':'d0', 'synchrotron':'s0'}
        
        Note that the key denote the emission and the value denote the sky model using PySM convention. For CMB, seed denote the realization.
        
        """
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):
            #print(ii, i)

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb']:
                    if self.params['QUBIC']['seed'] == 0:
                        if self.rank == 0:
                            seed = np.random.randint(10000000)
                        else:
                            seed = None
                        seed = self.comm.bcast(seed, root=0)
                    else:
                        seed = self.params['QUBIC']['seed']
                    print(f'Seed of the CMB is {seed} for rank {self.rank}')
                    sky['cmb'] = seed
                
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = self.params['QUBIC']['dust_model']
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['QUBIC']['sync_model']

        return sky
    def get_ultrawideband_config(self):
        
        """
        
        Method that pre-compute UWB configuration.

        """
        
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
    
        return nu_ave, 2*delta/nu_ave
    def get_dict(self):
    
        """
        
        Method to modify the qubic dictionary.
        
        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {'npointings':self.params['QUBIC']['npointings'], 
                'nf_recon':self.params['QUBIC']['nrec'], 
                'nf_sub':self.params['QUBIC']['nsub'], 
                'nside':self.params['Sky']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['QUBIC']['RA_center'], 
                'DEC_center':self.params['QUBIC']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'dtheta':self.params['QUBIC']['dtheta'],
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':self.params['QUBIC']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(self.params['QUBIC']['detector_nep']), 
                'synthbeam_kmax':self.params['QUBIC']['synthbeam_kmax']}
        
        args_mono = args.copy()
        args_mono['nf_recon'] = 1
        args_mono['nf_sub'] = 1

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = Qqubicdict.qubicDict()
        d.read_from_file(dictfilename)
        dmono = d.copy()
        for i in args.keys():
        
            d[str(i)] = args[i]
            dmono[str(i)] = args_mono[i]

    
        return d, dmono
    def _get_convolution(self):

        """
        
        Method to define expected QUBIC angular resolutions (radians) as function of frequencies.

        """
        
        ### Define FWHMs
        if self.params['QUBIC']['convolution']:
            allfwhm = self.joint.qubic.allfwhm
            targets = np.array([])
            for irec in range(self.params['QUBIC']['nrec']):
                targets = np.append(targets, np.sqrt(allfwhm[irec*self.fsub:(irec+1)*self.fsub]**2 - np.min(allfwhm[irec*self.fsub:(irec+1)*self.fsub])**2))
                #targets = np.sqrt(allfwhm**2 - np.min(allfwhm)**2)
        else:
            targets = None
            allfwhm = None

        return targets, allfwhm
    def get_input_map(self):
        m_nu_in = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))

        for i in range(self.params['QUBIC']['nrec']):
            m_nu_in[i] = np.mean(self.external_timeline.m_nu[i*self.fsub:(i+1)*self.fsub], axis=0)
        
        return m_nu_in    
    def _get_tod(self, noise=False):

        """
        
        Method that compute observed TODs with TOD = H . s + n with H the QUBIC operator, s the sky signal and n the instrumental noise.

        """
        
        if noise:
            factor = 0
        else:
            factor = 1
        if self.params['QUBIC']['type'] == 'wide':
            if self.params['QUBIC']['nrec'] != 1:
                TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                for irec in range(int(self.params['QUBIC']['nrec']/2)):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(factor * self.external_timeline.maps[irec] + self.noise143)

                for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(factor * self.external_timeline.maps[irec] + self.noise217)
            else:
                TOD_PLANCK = np.zeros((2*self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.allfwhm[-1])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)

                TOD_PLANCK[0] = C(factor * self.external_timeline.maps[0] + self.noise143)
                TOD_PLANCK[1] = C(factor * self.external_timeline.maps[0] + self.noise217)

            TOD_PLANCK = TOD_PLANCK.ravel()
            TOD_QUBIC = self.Hqtod(factor * self.external_timeline.m_nu).ravel() + self.noiseq
            TOD = np.r_[TOD_QUBIC, TOD_PLANCK]

        else:

            sh_q = self.joint.qubic.ndets * self.joint.qubic.nsamples
            TOD_QUBIC = self.Hqtod(factor * self.external_timeline.m_nu).ravel() + self.noiseq

            TOD_QUBIC150 = TOD_QUBIC[:sh_q].copy()
            TOD_QUBIC220 = TOD_QUBIC[sh_q:].copy()

            TOD = TOD_QUBIC150.copy()
    
            TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
            for irec in range(int(self.params['QUBIC']['nrec']/2)):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                TOD = np.r_[TOD, C(factor * self.external_timeline.maps[irec] + self.noise143).ravel()]

            TOD = np.r_[TOD, TOD_QUBIC220.copy()]
            for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                TOD = np.r_[TOD, C(factor * self.external_timeline.maps[irec] + self.noise217).ravel()]

        self.m_nu_in = self.get_input_map()

        return TOD
    def _barrier(self):

        """
        
        Method to introduce comm.Barrier() function if MPI communicator is detected.
        
        """
        if self.comm is None:
            pass
        else:
            self.comm.Barrier()
    def print_message(self, message):
        
        """
        
        Method to print message only on rank 0 if MPI communicator is detected. It display simple message if not.
        
        """
        
        if self.comm is None:
            print(message)
        else:
            if self.rank == 0:
                print(message)
    def _get_preconditionner(self):
        
        conditionner = np.ones((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
            
        for i in range(conditionner.shape[0]):
            for j in range(conditionner.shape[2]):
                conditionner[i, :, j] = 1/self.coverage_cut
                
        return acq.get_preconditioner(conditionner)
    def _pcg(self, d, x0):

        '''
        
        Solve the map-making equation iteratively :     H^T . N^{-1} . H . x = H^T . N^{-1} . d

        The PCG used for the minimization is intrinsequely parallelized (e.g see PyOperators).
        
        '''


        A = self.H.T * self.invN * self.H
        b = self.H.T * self.invN * d
        #print(self.params)
        ### Preconditionning
        #M = acq.get_preconditioner(np.ones(12*self.params['Sky']['nside']**2))
        M = self._get_preconditionner()
        #print("PRECONDITIONNER")

        ### PCG
        start = time.time()
        solution_qubic_planck = pcg(A=A, 
                                    b=b, 
                                    comm=self.comm,
                                    x0=x0, 
                                    M=M, 
                                    tol=self.params['PCG']['tol'], 
                                    disp=True, 
                                    maxiter=self.params['PCG']['maxiter'], 
                                    create_gif=self.params['PCG']['gif'], 
                                    center=self.center, 
                                    reso=self.params['QUBIC']['dtheta'], 
                                    seenpix=self.seenpix,
                                    jobid=self.job_id)

        self._barrier()

        if self.params['PCG']['gif']:
            do_gif(f'gif_convergence_{self.job_id}', solution_qubic_planck['nit'], self.job_id)

        if self.params['QUBIC']['nrec'] == 1:
            solution_qubic_planck['x']['x'] = np.array([solution_qubic_planck['x']['x']])
        end = time.time()
        execution_time = end - start
        self.print_message(f'Simulation done in {execution_time:.3f} s')

        return solution_qubic_planck['x']['x']
    def save_data(self, name, d):

        """
        
        Method to save data using pickle convention.
        
        """
        
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):

        """
        
        Method to run the whole pipeline from TOD generation from sky reconstruction by reading `params.yml` file.
        
        """
        
        self.print_message('\n=========== Map-Making ===========\n')

        ### Get simulated data
        self.TOD = self._get_tod(noise=False)
        self.n = self._get_tod(noise=True)

        ### Wait for all processes
        self._barrier()

        ### Solve map-making equation
        self.s_hat = self._pcg(self.TOD, x0=self.m_nu_in)
        self.s_hat[:, ~self.seenpix, :] = 0
        
        dict_solution = {'maps':self.s_hat, 'nus':self.nus_Q, 'coverage':self.coverage, 'center':self.center, 'maps_in':self.m_nu_in, 'parameters':self.params}
        if self.params['QUBIC']['do_noise_only']:
            self.s_hat_noise = self._pcg(self.n, x0=self.m_nu_in*0)
            self.s_hat_noise[:, ~self.seenpix, :] = 0
            dict_solution['maps_noise'] = self.s_hat_noise
        
        ### Plots and saving
        if self.rank == 0:
            
            self.save_data(self.file, dict_solution)
            self.externaldata.run(fwhm=self.params['QUBIC']['convolution'], noise=True)
            
            self.external_maps = self.externaldata.maps.copy()
            self.external_maps[:, ~self.seenpix, :] = 0
            
            self.external_maps_noise = self.externaldata_noise.maps.copy()
            self.external_maps_noise[:, ~self.seenpix, :] = 0
            
            if len(self.externaldata.external_nus) != 0:
                self.s_hat = np.concatenate((self.s_hat, self.external_maps), axis=0)
                self.s_hat_noise = np.concatenate((self.s_hat_noise, self.external_maps_noise), axis=0)
                self.nus_Q = np.array(list(self.nus_Q) + list(self.externaldata.external_nus))
            
            
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=0, nsig=3, fwhm=0.0048, name='signal')
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=1, nsig=3, fwhm=0.0048, name='signal')
            self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=2, nsig=3, fwhm=0.0048, name='signal')
            
            self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=0, nsig=3, fwhm=0.0048, name='noise')
            self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=1, nsig=3, fwhm=0.0048, name='noise')
            self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=2, nsig=3, fwhm=0.0048, name='noise') 

        self._barrier()   


class PipelineEnd2End:

    """

    Wrapper for End-2-End pipeline. It added class one after the others by running method.run().

    """

    def __init__(self, comm):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.comm = comm
        self.job_id = os.environ.get('SLURM_JOB_ID')
        
        create_folder_if_not_exists(self.comm, f'allplots_{self.job_id}')

        self.job_id = os.environ.get('SLURM_JOB_ID')
        if self.comm.Get_rank() == 0:
            if not os.path.isdir(self.params['path_out'] + 'maps/'):
                os.makedirs(self.params['path_out'] + 'maps/')
            if not os.path.isdir(self.params['path_out'] + 'spectrum/'):
                os.makedirs(self.params['path_out'] + 'spectrum/')
        self.file = self.params['path_out'] + 'maps/' + self.params['Data']['datafilename'] + f'_{self.job_id}.pkl'
        self.file_spectrum = self.params['path_out'] + 'spectrum/' + 'spectrum_' + self.params['Data']['datafilename']+f'_{self.job_id}.pkl'
        #print(self.file)
        #print(self.file_spectrum)
        #stop
        ### Initialization
        self.mapmaking = PipelineFrequencyMapMaking(self.comm, self.file)
        
        
    
    def main(self):

        ### Execute Frequency Map-Making
        self.mapmaking.run() 
        
        ### Execute spectrum
        if self.mapmaking.rank == 0:
            self.spectrum = Spectrum(self.params, self.mapmaking)
            self.spectrum.run(self.file_spectrum)

        

        
        

