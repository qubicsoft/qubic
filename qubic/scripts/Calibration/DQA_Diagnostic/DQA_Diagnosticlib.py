##############################################################################
#Authors: E. Manzan
#Date: 2023-03-08 
#Library dedicated at performing an overall analysis of the quality of the focal plane data

import qubic
import qubicpack
from qubicpack.qubicfp import qubicfp
from qubic import demodulation_lib as dl

from tqdm import tqdm
from qubic import selfcal_lib as scal
from qubic import progress_bar
import qubic.fibtools as ft

import numpy as np
from pylab import *
import glob
import pickle

from matplotlib import rc
rc('figure',figsize=(5,5))
rc('font',size=16)

class Diagnostic:

    def __init__(self, a_list, data_date, sat_thr=0.01, upper_satval = 4.19*1e6):
        '''
        This Class takes in:
        - a_list: a qubic.focalplane object. Either a single qubicfp or a list of qubicfp.
        - sat_thr: float, fraction of time to be used as threshold for saturation. If the saturation time is > sat_thr : detector is saturated. Default is 0.01 (1%)
        - upper_satval: float, positive ADU value ccorresponding to saturation, Default is 4.19*1e6
        
        The methods perform the following type of diagnostic:
        - focal plane display
        - timeline acquisition
        - saturation detection
        - flux jumps detection (TO BE IMPLEMENTED)
        - power spectral density 
        - coadded maps (TO BE IMPLEMENTED)
        '''
        #initialize the qubicfp object
        self.a = a_list
        #FluxJumps.__init__(a)
        
        #save dataset name (single string or list of strings)
        self.data_date = data_date
        print()
        print('#######################################')
        print('Initialize diagnostic class for the dataset : \n {}'.format(self.data_date))
        
        #save saturation threshold
        self.sat_thr = sat_thr
        print('Saturation time threshold is : {} %'.format(self.sat_thr*100))
        #save ADU value corresponding to saturation
        self.upper_satval = upper_satval
        self.lower_satval = - upper_satval
        print('ADU saturation values: {} and {}'.format(self.upper_satval, self.lower_satval))
        
        #dummy variable for the detector color, to be set later
        self.colors = 0
        
        #load in the tods from self.a
        print('##########Loading in the TODs##########')
        if isinstance(self.data_date, str): 
            #there's only one qubicfc (dataset) to load
            self.timeaxis, self.tod = self.a.tod()
        else:
            #there's more than one qubicfc (dataset) to load
            self.timeaxis, _= self.a[0].tod()
            self.num_datasets = len(self.a)
            print('There are {} datasets'.format(self.num_datasets))
            tods = []
            for i in range(self.num_datasets):
                _, tod_tmp = self.a[i].tod()
                print(tod_tmp.shape)
                tods.append(tod_tmp)
                print(len(tods))
            self.tod = np.array(tods)
        print('Timeline : ', self.timeaxis.shape)
        print('ADU : ', self.tod.shape)
        print('#######################################')

    def do_saturation_diagnostic(self, adu, data_date_full, do_plot=True):
        '''
        This function detects saturated TESs, prints out the saturation percentage, (optionally) plots the focal plane timeline with saturated detectors in red (thermo TES are in black, the others in green) and returns:
        - tes_to_use : array of Bool; True if TES is NOT saturated, False if TES is saturated or a thermometer
        - colors : array of strings; 'g' if the TES is NOT saturated, 'r' if TES is saturated, 'k' if TES is a thermometer
        Input:
        - adu: (array) the detectors ADU timeline
        - data_date_full : string with the label of the dataset
        - do_plot: if True, plots the focal plane
        '''
    
        #upper_satval = 4.19*1e6
        #lower_satval = -4.19*1e6

        frac_sat_time = np.zeros(256)
        tes_to_use = np.ones(256, dtype=bool)

        colors = np.ones(256, dtype=str) 

        for i in range(256):
            mask1 = adu[i] > self.upper_satval
            mask2 = adu[i] < self.lower_satval
            frac_sat_time[i] = (np.sum(mask1)+np.sum(mask2))/len(adu[i])

        saturated_tes = (frac_sat_time >= self.sat_thr)
        fraction_saturated_tes = np.sum(saturated_tes) / 256
        fraction_not_saturated_tes = np.sum(~saturated_tes) / 256

        tes_to_use[saturated_tes] = False
        colors[saturated_tes] = 'r'
        colors[~saturated_tes] = 'g'

        thermo_tes_idx = np.array([3,35,67,99])
        tes_to_use[thermo_tes_idx] = False
        tes_to_use[thermo_tes_idx+128] = False
        colors[thermo_tes_idx] = 'k'
        colors[thermo_tes_idx+128] = 'k'

        print('{:.2f}% TES are saturated; {:.2f}% TES are not saturated'.format(fraction_saturated_tes*100, fraction_not_saturated_tes*100))

        #plot
        if do_plot:
            suptitle_to_use = '{} : FP timeline. {:.2f}% TES are saturated; {:.2f}% TES are not saturated'.format(data_date_full,
                                                                                                    fraction_saturated_tes*100,
                                                                                                    fraction_not_saturated_tes*100)
            savefig_path_and_filename = './Focal_plane_saturation_{}'.format(data_date_full)
            self.plot_focal_plane(self.timeaxis, adu, colors, plot_suptitle=suptitle_to_use, path_and_filename=savefig_path_and_filename)

        return tes_to_use, colors
        

    def tes_saturation_state(self, do_plot=True):
        '''
        Wrapper to the -do_saturation_diagnostic- function, i.e. the saturation detection function. 
        If there is more than one dataset, it calls the -do_saturation_diagnostic- for each dataset.
        '''
        print()
        print('-----> Saturation detection')
        if isinstance(self.data_date, str): #there's only one dataset
            self.tes_to_use, self.colors = self.do_saturation_diagnostic(self.tod, self.data_date, do_plot) 
        else: #there is more than one dataset. loop over all of them
            self.tes_to_use = np.ones((self.num_datasets, 256), dtype=str)
            self.colors = np.ones((self.num_datasets, 256), dtype=str) 
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.tes_to_use[j,:], self.colors[j,:] = self.do_saturation_diagnostic(self.tod[j], self.data_date[j], do_plot)
                
        return
    
    
    def hf_noise_estimate_and_spectrum(self, time, adu, data_date_full, colors_to_use, do_plot=True):
        '''
        This function computes the power spectral density of the entire focal plane and plots it (optional).
        It returns:
        - estimate: (float) estimated white noise level
        - spectrum_density: (array) power spectral density of each TESs 
        - freq_f: (array) frequencies used for PSD computation 
        Input:
        - time: (array) timestamps
        - adu: (array) the detectors ADU timeline 
        - data_date_full : string with the label of the dataset
        - colors_to_use : array of string, containing the color (state) of each detector
        - do_plot: if True, plots the focal plane
        '''
        sh = np.shape(adu)
        if len(sh) == 1:
            adu = np.reshape(adu, (1, len(adu)))
            ndet = 1
        else:
            ndet = sh[0]
        estimate = np.zeros(ndet)
        spectrum_density = []
        for i in range(ndet):
            spectrum_f, freq_f = ft.power_spectrum(time, adu[i, :], rebin=True)
            spectrum_density.append(spectrum_f)
            mean_level = np.mean(spectrum_f[np.abs(freq_f) > (3*np.max(freq_f) / 4)])
            samplefreq = 1. / (time[1] - time[0])
            estimate[i] = (np.sqrt(mean_level * samplefreq / 2))
        
        if do_plot:
            #if self.colors doesn't exist, instantiate it and then do plot
            if not isinstance(self.colors, np.ndarray):
                #instantiate color array
                self.tes_saturation_state(do_plot=False)
            #do plot
            self.plot_focal_plane(freq_f, np.array(spectrum_density), colors_to_use, plot_suptitle='{}: power spectrum'.format(data_date_full),
                             path_and_filename='./{}_powerspectrum'.format(data_date_full), the_yscale='log')
        
        return estimate, np.array(spectrum_density), freq_f
    
    def tes_noise_estimate_and_spectrum(self, do_plot=True):
        '''
        Wrapper to the -hf_noise_estimate- function, i.e. the power spectral density computation. 
        If there is more than one dataset, it calls the -hf_noise_estimate- for each dataset.
        '''
        print('-----> Power spectrum computation ')
        if isinstance(self.data_date, str): #there's only one dataset
            self.noise_level, self.spectrum_density, self.frequency = self.hf_noise_estimate_and_spectrum(self.timeaxis, self.tod, self.data_date, self.colors, do_plot)
        else: #there is more than one dataset. loop over all of them 
            noise_level = []
            spectrum_density = []
            freq = []
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                noise_level_tmp, spectrum_density_tmp, freq_tmp  = self.hf_noise_estimate_and_spectrum(self.timeaxis, self.tod[j], self.data_date[j], self.colors[j], do_plot)
                noise_level.append(noise_level_tmp)
                spectrum_density.append(spectrum_density_tmp)
                freq.append(freq_tmp)
           
            self.noise_level = np.array(noise_level)
            self.spectrum_density = np.array(spectrum_density)
            self.frequency = np.array(freq)
        return
    
    
    def plot_focal_plane(self, x_data, tes_y_data, colors_plot, plot_suptitle=None, path_and_filename=None, the_xscale='linear', the_yscale='linear'):
        '''
        This function plots some data (timeline, spectrum) over the entire qubic focal plane.
        Input: 
        - x_data: an array of data running on the x axis (time, frequencies, etc.)
        - y_data: an array of data running on the y axis (ADU, power, etc.)
        - colors_plot: an array of 256 (or number of detectors) elements. Each element is a string with the color of that detector ('g' for good TESs, 'r' for saturation, etc.)
        - plot_suptitle (optional): string with the title of the plot
        - path_and_filename (optional): string with namefile of the plot. The function will save a png and a pdf version of the FP
        - the_xscale and the_yscale : 'linear' or 'log' (default is linear) ; to choose the scale of the x and y axes
        '''
        dictfilename = 'global_source_oneDet.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        d['synthbeam'] = 'CalQubic_Synthbeam_Calibrated_Multifreq_FI.fits'
        q = qubic.QubicInstrument(d)

        figure(figsize=(30, 30))
        bar=progress_bar(256, 'Display focal plane')
        #add title
        if plot_suptitle is not None:
            suptitle(plot_suptitle, x=0.5, y=1.0, fontsize=12)
        #define FP grid
        x=np.linspace(-0.0504, -0.0024, 17)
        y=np.linspace(-0.0024, -0.0504, 17)
        X, Y = np.meshgrid(x, y)

        allTES=np.arange(1, 129, 1)
        good_tes = allTES
        coord_thermo=np.array([17*11+1, 17*12+1, 17*13+1, 17*14+1, 275, 276, 277, 278])
        bgcol = 'w'
        k=0
        k_thermo=0
        #plot FP
        for j in [1, 2]:
            for i in good_tes:

                if np.sum(i == np.array([4,36,68,100])) != 0:
                    place_graph=coord_thermo[k_thermo]
                    k_thermo+=1
                else:
                    xtes, ytes, FP_index, index_q= scal.TES_Instru2coord(TES=i, ASIC=j, q=q, frame='ONAFP', verbose=False)
                    ind=np.where((np.round(xtes, 4) == np.round(X, 4)) & (np.round(ytes, 4) == np.round(Y, 4)))
                    place_graph=ind[0][0]*17+ind[1][0]+1
                num_tes=i
                if j == 2:
                    num_tes+=128
                idx_tes = num_tes - 1

                rc('font',size=6)
                subplot(17,17, place_graph)
                #############################
                #### Do your plot here ######
                if x_data.ndim==1:
                    plot(x_data, tes_y_data[idx_tes, :], color=colors_plot[idx_tes])
                else:
                    plot(x_data[idx_tes], tes_y_data[idx_tes, :], color=colors_plot[idx_tes])
                #############################            
                xscale(the_xscale)
                yscale(the_yscale)
                annotate('{}'.format(num_tes), xy=(0, 0),  xycoords='axes fraction', fontsize=9, color='black',
                     fontstyle='italic', fontweight='bold', xytext=(0.05,0.85),backgroundcolor=bgcol)

                bar.update()

                k+=1
        tight_layout()
        #Save image
        if path_and_filename is not None:
            savefig(path_and_filename+'.png')
            savefig(path_and_filename+'.pdf')
        return
    
    def plot_raw_focal_plane(self):
        '''
        This function plots the raw timeline of the entire focal plane
        '''
        colors_plot = np.array(['c' for i in range(256)])
        #colors_plot = 'g'*np.ones(256, dtype=str) 
        colors_plot[128:] = 'b'
        thermo_tes_idx = np.array([3,35,67,99])
        #colors_plot[~thermo_tes_idx] = 'g'
        colors_plot[thermo_tes_idx] = 'k'
        colors_plot[thermo_tes_idx+128] = 'k'
        
        if isinstance(self.data_date, str): #len(self.a) == 1:
            plot_suptitle = '{} : FP timeline'.format(self.data_date)
            print('Plotting :', plot_suptitle)
            path_and_filename = './Focal_plane_{}'.format(self.data_date)
            self.plot_focal_plane(self.timeaxis, self.tod, colors_plot, plot_suptitle, path_and_filename)
        else:
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                plot_suptitle = '{} : FP timeline'.format(self.data_date[j])
                print('Plotting:', plot_suptitle)
                path_and_filename = './Focal_plane_{}'.format(self.data_date[j])
                self.plot_focal_plane(self.timeaxis, self.tod[j], colors_plot, plot_suptitle, path_and_filename)
        return
        
        
    def analysis(self, do_plot=True):
        '''
        Wrapper function that calls all the diagnostic methods and performs a global analysis in a (semi-)automatic way
        '''
        
        #plot the timelines over the focal plane
        self.plot_raw_focal_plane()
        
        #saturation diagnostic
        self.tes_saturation_state(do_plot)
        
        #power spectrum diagnostic
        self.tes_noise_estimate_and_spectrum(do_plot)
        
    #def analysis(self):

        # Perform method1
        #self.method1()

        # Perform method2
        #self.method2()

        # Perform method2
        #self.method2()        
############################################################################################################################