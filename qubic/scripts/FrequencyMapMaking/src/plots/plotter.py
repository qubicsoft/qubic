import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

class PlotsMM:

    def __init__(self, params):

        self.params = params
        self.stk = ['I', 'Q', 'U']

    def plot_FMM(self, m_in, m_out, center, seenpix, nus, job_id, figsize=(10, 8), istk=1, nsig=3, name='signal'):
        
        m_in[:, ~seenpix, :] = hp.UNSEEN
        m_out[:, ~seenpix, :] = hp.UNSEEN

        
        plt.figure(figsize=figsize)

        k=1
        for i in range(self.params['QUBIC']['nrec']):
            
            hp.gnomview(m_in[i, :, istk], rot=center, reso=15, cmap='jet', 
                        min = - nsig * np.std(m_out[0, seenpix, istk]), 
                        max = nsig * np.std(m_out[0, seenpix, istk]), 
                        sub=(self.params['QUBIC']['nrec'], 3, k),
                        title=r'Input - $\nu$ = '+f'{nus[i]:.0f} GHz')
            hp.gnomview(m_out[i, :, istk], rot=center, reso=15, cmap='jet', 
                        min = - nsig * np.std(m_out[0, seenpix, istk]), 
                        max = nsig * np.std(m_out[0, seenpix, istk]), 
                        sub=(self.params['QUBIC']['nrec'], 3, k+1),
                        title=r'Output - $\nu$ = '+f'{nus[i]:.0f} GHz')
            
            res = m_in[i, :, istk] - m_out[i, :, istk]
            res[~seenpix] = hp.UNSEEN
            
            hp.gnomview(res, rot=center, reso=15, cmap='jet', min = - nsig * np.std(m_out[0, seenpix, istk]), max = nsig * np.std(m_out[0, seenpix, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+2))

            k+=3
        plt.savefig(f'allplots_{job_id}/frequency_maps_{self.stk[istk]}_{name}.png')
        plt.close()

    def plot_FMM_mollview(self, m_in, m_out, nus, job_id, figsize=(10, 8), istk=1, nsig=3, fwhm=0):

        C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        plt.figure(figsize=figsize)

        k=1
        for i in range(self.params['QUBIC']['nrec']):
            hp.mollview(C(m_in[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k),
                        title=r'Input - $\nu$ = '+f'{nus[i]:.0f} GHz')

            hp.mollview(C(m_out[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+1),
                        title=r'Output - $\nu$ = '+f'{nus[i]:.0f} GHz')

            hp.mollview(C(m_in[i, :, istk]) - C(m_out[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+2))

            k+=3
        plt.savefig(f'allplots_{job_id}/frequency_maps_{self.stk[istk]}_moll.png')
        plt.close()