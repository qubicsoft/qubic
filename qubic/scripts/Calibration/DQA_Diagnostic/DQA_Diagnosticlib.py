# -*- coding: utf-8 -*-
# #############################################################################
# Authors: E. Manzan. M.Regnier, B.Costanza
# Date: 2023-03-08 
# Library dedicated at performing an overall analysis of the quality of the focal plane data

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
import os

# Scikit-Learn packages
import bottleneck as bn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from scipy.signal import savgol_filter 

from matplotlib import rc
rc('figure',figsize=(5,5))
rc('font',size=16)

####################################################### HOUSEKEEPING DIAGNOSTIC ########################################################################
class HK_Diagnostic:
    
    def __init__(self, a_list, data_date):
        '''
        This Class performs a HOUSEKEEPING diagnostic. It takes in:
        - a_list: a qubic.focalplane object. Either a single qubicfp or a list of qubicfp.
        - data_date: str or array of str. Contains the name of the dataset(s)
        WARNING: The list feature has to be implemented yet! Right now, pass a single qubicfp object!
        
        The methods perform the following type of diagnostic:
        - quicklook
        - ASICs synchro error display
        - 1K temperatures plot
        - 1K temperatures plot wo patological sensors
        - 1K and 300mK monitor: just the STAGE and FRIDGE sensors.
        The user can call each method individually or all of them through the analysis() method.
        '''
        
        self.a = a_list
        self.data_date = data_date       
        
        print()
        print('#######################################')
        print('Initialize HOUSEKEEPING diagnostic class for the dataset : \n {}'.format(self.data_date))
        
        #load in the tods from self.a
        if isinstance(self.data_date, str): 
            #there's only one qubicfc (dataset) to load. Create directory name to save focal plane plots using the dataset name
            self.path_HK = './HK_{}'.format(self.data_date)           
        else:
            #there's more than one qubicfc (dataset) to load
            self.num_datasets = len(self.a)
            print('There are {} datasets'.format(self.num_datasets))
            #create directory name to save focal plane plots: use only the DATE of the set of datasets in this case
            self.path_HK = './HK_{}'.format(self.data_date[0].split('_')[0])
            
        #Create directory to save focal plane plots    
        if not os.path.exists(self.path_HK):
            print('Creating directory : ', self.path_HK)
            os.mkdir(self.path_HK)

        print('#######################################')

    def plot_synch_error(self, a, data_date):
        '''
        This function plots the ASIC1 (red) and ASIC2 (blue) NETQUIC synchro error on a double-y axis plot. A png image is saved.
        If NETQUIC synchro error == 0 : OK. If NETQUIC synchro error == 1 : There's a synchro error.
        '''
        #save and plot ASIC 1
        figure(figsize=(10,10))
        title(data_date+' :\nNETQUIC synchro error', y=1.005)
        if a.asic(1) is not None:
            synch_err1 = a.asic(1).hk['CONF_ASIC1']['NETQUIC synchro error']
            tstamps1 = a.asic(1).hk['CONF_ASIC1']['ComputerDate']
            tstamps1 -= tstamps1[0] #convert time into hours
            tstamps1 /= (60*60)
            plot(tstamps1, synch_err1, color='r')
            if np.max(synch_err1) == 1:
                print('WARNING: There is a synchro error in ASIC1! Please check further!')
            else:
                print('No synchro error in ASIC1 detected!')
                
        else:
            textstr = 'No ASIC 1 data!' 
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            text(0., 0.4, textstr, fontsize=25, color='r', verticalalignment='top', bbox=props)
        ylabel('NETQUIC ASIC 1', color='r')
        xlabel('Time [hours]')
        twinx()
        #save and plot ASIC 2
        if a.asic(2) is not None:
            synch_err2 = a.asic(2).hk['CONF_ASIC2']['NETQUIC synchro error']
            tstamps2 = a.asic(2).hk['CONF_ASIC2']['ComputerDate']
            tstamps2 -= tstamps2[0] #convert time into hours
            tstamps2 /= (60*60)
            plot(tstamps2, synch_err2, color='b')
            if np.max(synch_err2) == 1:
                print('WARNING: There is a synchro error in ASIC2! Please check further!')
            else:
                print('No synchro error in ASIC2 detected!')                
        else:
            textstr = 'No ASIC 2 data!' 
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            text(0., 0.2, textstr, fontsize=25, color='r', verticalalignment='top', bbox=props)
        ylabel('NETQUIC ASIC 2', color='b')
        xlabel('Time [hours]')
        savefig(self.path_HK+'/{}_NETQUIC_synchro.png'.format(data_date))
        close()
        
        return
    
    def get_plot_synch_error(self):
        '''
        Wrapper to the -plot_synch_error- function. If there is more than one dataset, it calls the function for each dataset.
        '''
        print()
        print('-----> Synchro error diagnostic')
        if isinstance(self.data_date, str): #there's only one dataset
            #self.synch_status_asic1, self.synch_status_asic2 = 
            self.plot_synch_error(self.a, self.data_date) 
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.plot_synch_error(self.a[j], self.data_date[j])                
        return
    
        
    def plot_1Ktemperature_wo_problematic_sensors(self, a, data_date):
        '''
        This function returns a 1K temperature sensors plot without problematic sensors, like the 1K Stage Back sensor.  A png image is saved.
        '''
        #save sensor labels: 
        label = {}
        label['AVS47_1_ch1'] = '1K stage'
        label['AVS47_1_ch3'] = 'M1'
        label['AVS47_1_ch4'] = '1K fridge CH'
        label['AVS47_1_ch7'] = 'M2'
        label['AVS47_2_ch3'] = '1K stage back'
        label['AVS47_2_ch0'] = 'PT2 S2 CH'
        label['AVS47_2_ch2'] = 'Fridge plate MHS'
        label['AVS47_2_ch4'] = '4K shield Cu braids'
        '''
        if np.max(a.get_hk('AVS47_1_ch1')) < 10.:
            label['AVS47_1_ch1'] = '1K stage'
        if np.max(a.get_hk('AVS47_1_ch3')) < 10.:
            label['AVS47_1_ch3'] = 'M1'
        if np.max(a.get_hk('AVS47_1_ch4')) < 10.:    
            label['AVS47_1_ch4'] = '1K fridge CH'
        if np.max(a.get_hk('AVS47_1_ch7')) < 10.:
            label['AVS47_1_ch7'] = 'M2'
        if np.max(a.get_hk('AVS47_2_ch3')) < 10.:
            label['AVS47_2_ch3'] = '1K stage back'
        if np.max(a.get_hk('AVS47_2_ch0')) < 10.:
            label['AVS47_2_ch0'] = 'PT2 S2 CH'
        if np.max(a.get_hk('AVS47_2_ch2')) < 10.:
            label['AVS47_2_ch2'] = 'Fridge plate MHS'
        if np.max(a.get_hk('AVS47_2_ch4')) < 10.:
            label['AVS47_2_ch4'] = '4K shield Cu braids'
            '''
        #plot
        #a.plot_temperatures(None,label,'1K Temperatures wo patological stages',12)
        time_sensors = a.get_hk(data='RaspberryDate',hk='EXTERN_HK').copy()
        time_sensors -= time_sensors[0]
        time_sensors /= 60
        figure(figsize=(15,15))
        i=1
        suptitle(data_date+' : 1K Temperatures wo patological stages', y=1)
        for k in label.keys():
            subplot(3,3,i)
            #define mask corresponding to patological events (e.g. sensor behaving erratically).
            timest_out = (a.get_hk(k) > 5.) | (a.get_hk(k) < 0)
            plot(time_sensors[~timest_out], a.get_hk(k)[~timest_out], marker='D',markersize=0.2*12)
            ylabel(label[k]+' [K]')
            xlabel('Time [min]')
            i+=1
        tight_layout()
        #savefig(self.path_HK+'/{}_1K_Temperatures_wo_patological_stages.png'.format(data_date))
        savefig(self.path_HK+'/{}_1K_Temperatures_patological_stages_filtered.png'.format(data_date))
            
        close()
        return
    
    
    def get_plot_1Ktemperature_wo_problematic_sensors(self):
        '''
        Wrapper to the -1Ktemperature_wo_problematic_sensors- function. If there is more than one dataset, it calls the function for each dataset.
        '''
        print()
        print('-----> 1K temperatures diagnostic')
        if isinstance(self.data_date, str): #there's only one dataset
            self.plot_1Ktemperature_wo_problematic_sensors(self.a, self.data_date) 
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.plot_1Ktemperature_wo_problematic_sensors(self.a[j], self.data_date[j])                
        return

    
    def plot_1K_and_300mK_monitor(self, a, data_date):
        '''
        This function returns a 2-subplot image with the 1K monitor on the left and the 300mK monitor on the right.
        Each monitor shows a double-y axis with the temperature STAGE (blue) and FRIDGE (red).  A png image is saved.
        '''
        OneK_stage = a.get_hk('AVS47_1_ch1')
        OneK_fridge = a.get_hk('AVS47_1_ch4')
        TES_stage = a.get_hk('AVS47_1_CH2')
        ThreeHmK_fridge = a.get_hk('AVS47_1_CH6')
        time_hk = a.get_hk(data='RaspberryDate',hk='EXTERN_HK').copy()
        time_hk -= time_hk[0] #convert time into hours
        time_hk /= 60 #(60*60)
        
        #define mask corresponding to patological events (e.g. sensor behaving erratically).
        #define 1 mask for 1K, since 1K stage and fridge share the same time axis in the plot
        timest_out_oneK = (OneK_stage > 2) | (OneK_stage < 0.9) | (OneK_fridge > 2) | (OneK_fridge < 0.9)
        #define 1 mask for 300mK monitor, since TES stage and fridge share the same time axis in the plot
        timest_out_TESstage = (TES_stage > 0.5) | (TES_stage < 0.2) | (ThreeHmK_fridge > 0.5) | (ThreeHmK_fridge < 0.2)
        
        figure(figsize=(20,10))
        suptitle(data_date+' : 1K and 300 mK monitors', y=1)
        #plot 1K monitor
        subplot(1,2,1)
        title('1K monitor', y=0.95)
        plot(time_hk, OneK_stage, color='b', marker='D',markersize=0.2*12)
        ylabel('1K Stage [K]', color='b')
        xlabel('Time [min]')
        twinx()
        plot(time_hk, OneK_fridge, color='r', marker='D',markersize=0.2*12)
        ylabel('1K Fridge [K]', color='r')
        #plot 300mK monitor
        subplot(1,2,2)
        title('300 mK monitor', y=0.95)
        plot(time_hk, TES_stage, color='b', marker='D',markersize=0.2*12)
        ylabel('TES Stage [K]', color='b')
        xlabel('Time [min]')
        twinx()
        plot(time_hk, ThreeHmK_fridge, color='r', marker='D',markersize=0.2*12)
        ylabel('300 mK Fridge [K]', color='r')
        
        tight_layout()
        savefig(self.path_HK+'/{}_1K_300mK_monitors.png'.format(data_date))
        #savefig('./{}_1K_300mK_monitors.pdf'.format(self.data_date))
        
        if any(timest_out_oneK) or any(timest_out_TESstage):
            #redo the plot with problematic/instable temperature timestamps removed
            print('--------> Patological events in the temperature sensors have been found! Plotting again w/o those!')
            figure(figsize=(20,10))
            suptitle(data_date+' : TOD and temperature monitors wo patological events', y=1)

            #plot 1K monitor
            subplot(1,2,1)
            title('1K monitor', y=0.95)
            plot(time_hk[~timest_out_oneK], OneK_stage[~timest_out_oneK], color='b', marker='D',markersize=0.2*12)
            ylabel('1K Stage [K]', color='b')
            xlabel('Time [min]')
            twinx()
            plot(time_hk[~timest_out_oneK], OneK_fridge[~timest_out_oneK], color='r', marker='D',markersize=0.2*12)
            ylabel('1K Fridge [K]', color='r')
            #plot 300mK monitor
            subplot(1,2,2)
            title('300 mK monitor', y=0.95)
            plot(time_hk[~timest_out_TESstage], TES_stage[~timest_out_TESstage], color='b', marker='D',markersize=0.2*12)
            ylabel('TES Stage [K]', color='b')
            xlabel('Time [min]')
            twinx()
            plot(time_hk[~timest_out_TESstage], ThreeHmK_fridge[~timest_out_TESstage], color='r', marker='D',markersize=0.2*12)
            ylabel('300 mK Fridge [K]', color='r')

            tight_layout()
            savefig(self.path_HK+'/{}_1K_300mK_monitors_filtered.png'.format(data_date))
        else:
            print('--------> No patological events in the temperature sensors have been found!')
        
        close('all')
        return
    
    
    def get_plot_1K_and_300mK_monitor(self):
        '''
        Wrapper to the -plot_1K_and_300mK_monitor- function. If there is more than one dataset, it calls the function for each dataset.
        '''
        print()
        print('-----> 1K and 300mK monitor')
        if isinstance(self.data_date, str): #there's only one dataset
            self.plot_1K_and_300mK_monitor(self.a, self.data_date) 
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.plot_1K_and_300mK_monitor(self.a[j], self.data_date[j])                
        return
    
    
    def analysis(self):
        '''
        Wrapper function that calls all the HK_Diagnostic methods to get all the necessary plots for the hk diagnostic
        '''
        #do a quicklook
        print()
        print('-----> Quicklook')
        if isinstance(self.data_date, str): #there's only one dataset
            self.a.quicklook()
            close()
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.a[j].quicklook()
                close()


        #plot ASICs synchro error
        self.get_plot_synch_error()

        #plot all the 1K temperature sensors
        print()
        print('-----> plot 1K sensors')
        if isinstance(self.data_date, str): #there's only one dataset
            self.a.plot_1Ktemperatures()
            close()
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.a[j].plot_1Ktemperatures()
                close()

        #plot 1K temperature sensors w/o the 1K Stage Back sensor, which is problematic
        self.get_plot_1Ktemperature_wo_problematic_sensors()

        #plot 1K and 300mK monitors (stage and fridge)
        self.get_plot_1K_and_300mK_monitor()


# ###################################################### SCIENTIFIC DIAGNOSTIC ########################################################################

class FluxJumps:
    def __init__(self, data_date_full, use_verbose = True, doplot = True):
        '''
        This Class performs only flux jumps (FJ) detection. It takes in:
        - data_date: (str) the name of the dataset(s)
        - use_verbose : Bool. If True: do all the flux jump intermediate plots (zoom on jumps, comparison btw raw tod and smoothed tod, etc.).
        - doplot : Bool. If True: returns a png image with 3 subplots, showing: i. tod and haar filter; ii. the tod with the jumps in red (if jumps have been detected); iii. The haar filter with all the amplitude thresholds, the detected jumps in red, the start/end of the jump in green.
        
        The methods perform the following type of diagnostic:
        - determine if there is at least on FJ using the raw tod and its Haar filter
        - if there are many FJ, check if they are "true" FJ using the smoothed tod and its Haar filter
        - (regardless of the jump detection method) find how many FJ there are using DBSCAN on given masked array of jump timestamps
        - (regardless of the jump detection method) for each FJ (DBSCAN cluster), finds its initial and final timestamp (using a 95% reduction level)
        - (optional) displays intermediate plots, e.g. zoom on jumps, comparison btw raw tod and smoothed tod
        - (optional) displays a set of 3 plot showing the overal results and saves it as png.
        The user can call this class on single detectors or call it for the entire focal plane using the Diagnostic class below.
        '''
        
        #define variables common to all detectors
        self.data_date = data_date_full 
        self.verbose = use_verbose 
        self.doplot = doplot
        self.size_to_use = 150 # (float) It's the number of timestamps used as window in the moving median of the Haar filter. Also used in the DBSCAN.
        self.window = 401 # odd number, moving windows for the Savitzky-Golay filter.
        self.polyorder = 3 # polynomial ordeer for the Savitzky-Golay filter
        self.method = 'nearest' # approximation mode for the Savitzky-Golay filter
        self.thr = np.array([2e5, 1.5e5, 1e5, 5e4, 4e4, 3e4]) # array of amplitude thresholds. Usually: [2e5, 1e5, 5e4, 3e4], etc.
        
        #create directory to save flux jumps plots
        self.path_FJ = './FluxJumps_{}'.format(self.data_date)
        if not os.path.exists(self.path_FJ):
            print('Creating directory : ', self.path_FJ)
            os.mkdir(self.path_FJ)
        
        
    #Haar filter definition
    def haar(self, x):
        '''
        This function performs a Haar Filter of the tod: it returns a rescaled median of the TOD, where flux jumps appear as spikes.
        Input:
        - x : array, the tod
        '''
        out = np.zeros(x.size)
        xf = bn.move_median(x, self.size_to_use)[self.size_to_use:]   
        out[self.size_to_use+self.size_to_use//2:-self.size_to_use+self.size_to_use//2] = xf[:-self.size_to_use] - xf[self.size_to_use:]
        return out
    
    
    def find_jumps(self, tod_haar):
        '''
        This function performs an iterative jump detection by comparing the maximum of the Haar filter with a series of thresholds. 
        Input:
        - tod_haar : array, the Haar filter of the tod
        Output:
        - number : zero or one. If zero, there are no jumps. If one, there's at least one jump
        - jumps : Masked array, the timestamp of the jump(s)
        - thr_used : (float) The threshold corresponding to the jump detection
        '''
        print('##### JUMP DETECTION #####')
        #dummy variable
        self.number = 0
        self.jumps = 0
        self.thr_used = 0 
        #iterate over the amplitude thresholds
        for j,thr_val in enumerate(self.thr):
            print('Iteration with thr :', thr_val)
            if self.number == 0: #haven't detected any jump yet
                if max(tod_haar) < thr_val:
                    print('No jump')
                else: #there's a jump
                    self.number += 1
                    self.thr_used = thr_val
                    print('Found jump')
                    self.jumps = (tod_haar >= thr_val) #save the timestamp of the jump
            else: #jumps already detected
                pass
        return #self.number, self.jumps, self.thr_used
    
    
    def plot_HaarFilter_and_SVFilter(self, tes_num, tt, todarray, tod_sav, tod_haar, tod_haar_sav):
        '''
        This function plots the raw tod and it's Haar filter compared to a smoothed tod (Savitzky-Golay, sav or SG) and it's corresponding Haar filter 
        Input:
        - tes_num : (int) detector number
        - tt : array of timestamps
        - todarray : array, tod
        - tod_haar : array, the Haar filter of the tod
        - tod_sav : array, smoothed tod (SG)
        - tod_haar_sav : Haar filter of the smoothed tod
        '''

        figure(figsize=(10,10))
        suptitle(self.data_date+'\n TES {}'.format(tes_num), y=1)
        subplot(2,1,1)
        plot(tt, todarray , 'g', label='Raw tod')
        plot(tt, tod_sav, 'b', alpha=0.7, label='SG tod')
        xlabel('time')
        ylabel('TOD')
        legend(loc='best')

        subplot(2,1,2)
        plot(tt, tod_haar , 'g', label='Raw-Haar tod')
        plot(tt, tod_haar_sav, 'b', alpha=0.7, label='SG-Haar tod')
        for j,thr_val in enumerate(self.thr):
            plot(tt, thr_val*np.ones(len(tt)), color='k', linestyle='--', label = 'Amplitude threshold')
        xlabel('time')
        ylabel('Haar filter')
        legend(loc='best')
        tight_layout()
        savefig(self.path_FJ+'/HaarFilter_and_SVFilter_TES{}.png'.format(tes_num))
        close()
        return
    
    def get_number_of_jumps(self, todarray):
        '''
        This function takes in a masked array with the timestamps of all the jumps and perform a DBSCAN to detect how many jumps (cluster of timestamps) there are.
        Input:
        - todarray : array, the tod
        - jumps : a masked array of timestamps. If True, there's a jump at that timestamp
        Output:
        - nc : (int) the number of jumps (cluster) found
        - idx_jumps : array made by all the timestamps of the jumps
        - clust : the DBSCAN cluster
        '''
        idx = np.arange(len(todarray))
        idx_jumps = idx[self.jumps]
        print(idx_jumps)

        #find number of jumps with cluster method
        clust = DBSCAN(eps=self.size_to_use//5, min_samples=1).fit(np.reshape(idx_jumps, (len(idx_jumps),1)))
        nc = np.max(clust.labels_)+1
        print('Number of FJ : ', nc)
        return nc, idx_jumps, clust
    
    
    def find_jump_start_and_end(self, nc, idx_jumps, clust, tt, tod_haar, tes_num):
        '''
        This function finds and returns the beginning and ending timestamp of each flux jump. If verbose, plots a zoom-in of all the Haar filter spikes (jumps).
        '''
        xc = np.zeros(nc, dtype=int) 
        xcf = np.zeros(nc, dtype=int)
        for i in range(nc):
            idx_jumps_from_thr = idx_jumps[clust.labels_ == i]
            print('From thr, jump start at idx : ', idx_jumps_from_thr[0])
            print('From thr, jump ends at idx : ', idx_jumps_from_thr[-1])

            #consider the jump to be over when it's (filtered) amplitude is reduced by 90%
            idx_delta_end_jump = np.where( tod_haar[idx_jumps_from_thr[-1]:] < self.thr_used*0.05 )[0][0]
            print('From filter ampl, jump ends at idx : ', idx_jumps_from_thr[-1] + idx_delta_end_jump)
            print('Timestamps number to be added : ', idx_delta_end_jump)
            idx_delta_start_jump = idx_jumps_from_thr[0] - np.where( tod_haar[:idx_jumps_from_thr[0]] < self.thr_used*0.05 )[0][-1]
            xc[i] = idx_jumps_from_thr[0] - idx_delta_start_jump
            xcf[i] = idx_jumps_from_thr[-1] + idx_delta_end_jump

            delta = xcf - xc #time samples of a jump candidate
            print('Number of timestamps of each jump : ', delta)

        if self.verbose: 
            figure(figsize=(10,10))
            suptitle('TES = {}'.format(tes_num), y=1)
            for i in range(nc):
                subplot(nc, 1, i+1)
                title('Jump {}'.format(i+1), x=0.3, y=0.9)
                #plot(tt[xc[i]], todarray[xc[i]], 'g.')
                #plot(tt[xcf[i]], todarray[xcf[i]], 'g.')
                idx_single_jump = idx_jumps[clust.labels_ == i]
                time_idx = (tt[idx_single_jump[0]-500] < tt) & ( tt < tt[idx_single_jump[-1]+500])
                plot(tt[time_idx], tod_haar[time_idx], color='orange')
                plot(tt[xc[i]:xcf[i]], tod_haar[xc[i]:xcf[i]], color='c', label='Total jump')
                plot(tt[idx_single_jump], tod_haar[idx_single_jump], color='red', label='Jump based on ampl. thr.')
                plot(tt[time_idx], np.zeros(len(tt[time_idx])), color='k', linestyle='--')
                plot(tt[time_idx], (self.thr_used)*np.ones(len(tt[time_idx])), color='g', linestyle='--', label='Thr. for jump detection')
                plot(tt[time_idx], (self.thr_used*0.05)*np.ones(len(tt[time_idx])), color='g', linestyle=':', label='Thr. for jump start/end')
                xlabel('Time')
                ylabel('Haar filter', color='orange')
                legend(loc='center left', bbox_to_anchor=(0.6, 0.5))
            tight_layout()
            savefig(self.path_FJ+'/ZoomJumps_TES{}.png'.format(tes_num))
            close()

        print('Number of final Jumps: ', nc)
        print()  
        return xc, xcf
    
    
    def make_flux_diagnostic_plot(self, tt, todarray, tod_haar, xc, xcf, nc, idx_jumps, tes_num):
        '''
        This function makes a diagnostic plot of the jump detection: it returns a plot with 3 subplots, showing: i. tod and haar filter; ii. the tod with the jumps in red (if jumps have been detected); iii. The haar filter with all the amplitude thresholds, the detected jumps in red, the start/end of the jump in green.
        Input:
        '''
        fig, ax = plt.subplots(3, figsize = (10,10))
        suptitle(self.data_date+'\n TES {}'.format(tes_num), y=1)
        #plot TOD and Filter
        ax[0].plot(tt, todarray, color='b')
        ax[0].set_ylabel('tod', color='b')
        ax2 = ax[0].twinx()
        ax2.plot(tt, tod_haar, color='orange', alpha=0.5)
        ax2.set_ylabel('haar filter', color='orange')

        #plot TOD with jumps highlighted
        ax[1].plot(tt, todarray, color='b', label='Jumps nr = {}'.format(nc))
        if self.number > 0 : 
            for i in range(nc):
                ax[1].plot(tt[xc[i]:xcf[i]], todarray[xc[i]:xcf[i]], 'r')  
            ax[1].legend()
        ax[1].set_ylabel('tod', color='b')               

        #plot Filter with jumps highlighted         
        ax[2].plot(tt, tod_haar, color='orange')
        if self.number > 0 : 
            ax[2].plot(tt[idx_jumps], tod_haar[idx_jumps], 'r.')
            ax[2].plot(tt[xc], tod_haar[xc], 'g.')
            ax[2].plot(tt[xcf], tod_haar[xcf], 'g.')
        for j,thr_val in enumerate(self.thr):
            ax[2].plot(tt, thr_val*np.ones(len(tt)), color='k', linestyle='--', label = 'Amplitude threshold')
        ax[2].set_ylabel('haar filter', color='orange')
        ax[2].set_xlabel('time')
        tight_layout()
        savefig(self.path_FJ+'/FluxJump_TES{}.png'.format(tes_num))
        #savefig(self.path_FJ+'/FluxJump_TES{}_size{}.png'.format(tes_num))
        close('all')
        return
    
      
    def fluxjumps_analysis(self, tt, todarray, tes_num):
        '''
        This function evaluates if there are flux jumps and their initial and final timestamps. Optionally plots the tod and jump filter.
        Input:
        - tt : array of timestamps
        - todarray : array of tes timeline
        - tes_num : (int) detector number (not index!)
        - data_date : (string) dataset label to be used in the plot
        - size : (int) ? to be used in the Haar filter. Default is ?
        - doplot: Bool. If True, do plot and save it. Otherwise, skip plot
        - verbose : if True, do additional plots and prints
        Returns:
        - xc : beginning of the jump 
        - xcf : final of the jump 
        - dif : diference between the final of a jump and the beginning of the next jump 
        - nc : number of clusters/number of jumps
        '''

        #make haar filter of raw tod
        tod_haar = np.abs(self.haar(todarray))

        #JUMP DETECTION : 
        #check maximum of raw tod: if it's > thr : it's a jump. If it's < thr : no jump
        self.find_jumps(tod_haar) #sets the following variables: number; jumps; threshold for which we have jump detection
        xc = 0
        xcf = 0
        nc = 0
        idx_jumps = 0

        #if there's a jump (i.e. self.number > 0), continue. Else: go directly to the plot stage
        if self.number > 0 : 
            print('##### JUMP INVESTIGATION #####')
            #get number of jumps and the set of timestamps where there are jumps (beware you still don't know how many jumps are there)
            nc, idx_jumps, clust = self.get_number_of_jumps(todarray)

            if nc > 10: #if there are more than 5 jumps, redo jump detection on smoothed tod (SAVGOL)
                tod_to_use = todarray.copy()
                for j in range(3):
                    #apply Savitzky–Golay filter (SG or savgol) to the raw tod : it will smooth out the tod
                    tod_sav = savgol_filter(tod_to_use, window_length=self.window, polyorder=self.polyorder, mode=self.method) 
                    #make haar filter of SG tod
                    tod_haar_sav = np.abs(self.haar(tod_sav))

                    if self.verbose:
                        self.plot_HaarFilter_and_SVFilter(tes_num, tt, todarray, tod_sav, tod_haar, tod_haar_sav)

                    #update self.number, self.jumps, self.thr_used and the tod
                    self.find_jumps(tod_haar_sav)
                    tod_to_use = tod_sav.copy()

                if self.number == 0 : #there are no jumps with the smoothed tod. Exit.
                    if self.doplot == True:
                        self.make_flux_diagnostic_plot(tt, todarray, tod_haar, xc, xcf, nc, idx_jumps, tes_num)
                    return nc, xc, xcf
                else: #get the number of jumps and their time interval using SG tod
                    nc, idx_jumps, clust = self.get_number_of_jumps(tod_haar_sav)

            #find Start and End of each jump: consider the jump to be over when it's (filtered) amplitude is reduced by 95% (beware: now use raw tod!)
            xc, xcf = self.find_jump_start_and_end(nc, idx_jumps, clust, tt, tod_haar, tes_num)

        #plot tod, filter, jump interval
        if self.doplot:
            self.make_flux_diagnostic_plot(tt, todarray, tod_haar, xc, xcf, nc, idx_jumps, tes_num)

        return nc, xc, xcf #returns number_of_jumps, start_jumps, end_jumps        


class Diagnostic:

    def __init__(self, a_list, data_date, sat_thr=0.01, upper_satval = 4.19*1e6, use_verbose = False):
        '''
        This Class performs a SCIENTIFIC diagnostic. It takes in:
        - a_list: a qubic.focalplane object. Either a single qubicfp or a list of qubicfp.
        - data_date: str or array of str. Contains the name of the dataset(s)
        - sat_thr: float, fraction of time to be used as threshold for saturation. If the saturation time is > sat_thr : detector is saturated. Default is 0.01 (1%)
        - upper_satval: float, positive ADU value corresponding to saturation. Fixed at 2**22 - 2**7 = 4.19*1e6 due to the detector dynamic range
        - use_verbose : Bool. If True: do all the flux jump intermediate plots (zoom on jumps, comparison btw raw tod and smoothed tod, etc.). Recommended is: False
        
        The methods perform the following type of diagnostic:
        - basic hk plots (quicklook, temperature monitor, synch error)
        - focal plane display
        - timeline acquisition
        - saturation detection (TO BE IMPROVED)
        - flux jumps detection
        - power spectral density (TO BE IMPROVED: smooth PS, add lines of known frequencies, add curve fit)
        - coadded maps (TO BE IMPLEMENTED)
        The user can call each method individually or all of them through the analysis() method.
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
        
        #set other variables
        self.colors = 0 #dummy variable for the detector color, to be set later
        self.num_det = 256 #total number of detectors
        self.use_verbose = use_verbose #this is used for flux jump intermediate plots (zoom on jumps, comparison btw raw tod and smoothed tod, etc.)
        
        #select a reference detector for each asic. This is used for visual diagnostic.
        self.num_ref_tes_asic1 = 95
        self.num_ref_tes_asic2 = 183
        
        #load in the tods from self.a
        print('##########Loading in the TODs##########')
        if isinstance(self.data_date, str): 
            #there's only one qubicfc (dataset) to load
            self.timeaxis, tod_input = self.a.tod()
            #if for some reason we only have part of the focalplane, set the missing tod info to zero
            self.tod = np.zeros((self.num_det, tod_input.shape[-1]))
            self.tod[:tod_input.shape[0],:] = tod_input
            #make a 2D time array (N_det, N_timestamp)
            self.time_fp = np.tile(self.timeaxis, (self.num_det,1))
            
            #create directory name to save focal plane plots using the dataset name
            self.path_FP = './FocalPlane_{}'.format(self.data_date)
            
        else:
            #there's more than one qubicfc (dataset) to load
            tmp_time_array, _= self.a[0].tod()
            self.num_datasets = len(self.a)
            print('There are {} datasets'.format(self.num_datasets))
            self.tod = np.zeros((self.num_datasets, self.num_det, len(tmp_time_array))) 
            self.timeaxis = np.zeros((self.num_datasets, len(tmp_time_array))) 
            self.time_fp = np.zeros((self.num_datasets, self.num_det, len(tmp_time_array))) 
            for i in range(self.num_datasets):
                time_input, tod_input = self.a[i].tod()
                self.tod[i, :tod_input.shape[0], :] = tod_input
                self.timeaxis[i] = time_input
                #make a 2D time array (N_det, N_timestamp) #THIS NEEDS TO BE FIXED!!!!!!!!!!!!!!!!!!
                self.time_fp[i] = np.tile(self.timeaxis, (self.num_det,1))
                
            #create directory name to save focal plane plots: use only the DATE of the set of datasets in this case
            self.path_FP = './FocalPlane_{}'.format(self.data_date[0].split('_')[0])

        print('Timeline : ', self.timeaxis.shape)
        print('ADU : ', self.tod.shape)
        print('Timeline focalplane: ', self.time_fp.shape)
        #create directory to save focal plane plots    
        if not os.path.exists(self.path_FP):
            print('Creating directory : ', self.path_FP)
            os.mkdir(self.path_FP)

        print('#######################################')

# #################### SATURATION ##################################################

    def do_saturation_diagnostic(self, adu, data_date_full, do_plot=True):
        '''
        This function detects saturated TESs, prints out the saturation percentage, (optionally) plots the focal plane timeline with saturated detectors in red (thermo TES are in black, the others in green) and returns:
        - tes_to_use : array of Bool; True if TES is NOT saturated, False if TES is saturated or a thermometer
        - colors : array of strings; 'g' if the TES is NOT saturated, 'r' if TES is saturated, 'k' if TES is a thermometer
        - fraction_saturated_tes : (float) fraction of saturated detectors
        Input:
        - adu: (array) the detectors ADU timeline
        - data_date_full : string with the label of the dataset
        - do_plot: if True, plots the focal plane
        '''

        frac_sat_time = np.zeros(self.num_det) # to be filled with fraction of saturation time of each detector
        tes_to_use = np.ones(self.num_det, dtype=bool) # to be filled with a saturation Bool for each detector (False = saturated)

        colors = np.ones(self.num_det, dtype=str) # to be filled with a color based on the saturation state of the detector 

        for i in range(self.num_det):
            mask1 = adu[i] > self.upper_satval
            mask2 = adu[i] < self.lower_satval
            frac_sat_time[i] = (np.sum(mask1)+np.sum(mask2))/len(adu[i])

        saturated_tes = (frac_sat_time > self.sat_thr)
        fraction_saturated_tes = np.sum(saturated_tes) / self.num_det
        fraction_not_saturated_tes = np.sum(~saturated_tes) / self.num_det

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
            savefig_path_and_filename = self.path_FP+'/Focal_plane_saturation_{}'.format(data_date_full)
            self.plot_focal_plane(self.time_fp, adu, colors, plot_suptitle=suptitle_to_use, path_and_filename=savefig_path_and_filename, frac_sat_time=frac_sat_time)

        return tes_to_use, colors, fraction_saturated_tes, frac_sat_time
        

    def tes_saturation_state(self, do_plot=True):
        '''
        Wrapper to the -do_saturation_diagnostic- function, i.e. the saturation detection function. 
        If there is more than one dataset, it calls the -do_saturation_diagnostic- for each dataset.
        '''
        print()
        print('-----> Saturation detection')
        if isinstance(self.data_date, str): #there's only one dataset
            self.tes_to_use, self.colors, self.frac_saturation, self.frac_sat_time = self.do_saturation_diagnostic(self.tod, self.data_date, do_plot)
            pickle.dump([self.tes_to_use, self.frac_saturation, len(self.timeaxis), self.frac_sat_time], open(self.path_FP+'/{}_Saturated_TES_statistics.pkl'.format(self.data_date),'wb'))
        else: #there is more than one dataset. loop over all of them
            self.tes_to_use = np.ones((self.num_datasets, self.num_det), dtype=bool)
            self.colors = np.ones((self.num_datasets, self.num_det), dtype=str) 
            self.frac_saturation = np.zeros(self.num_datasets) 
            self.frac_sat_time = np.zeros((self.num_datasets, self.num_det)) 
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                ############################################ FIX THIS: ADD TIME self.time_fp[j] and change do_saturation_diagnostic!!!!!!!!!!!!!!
                self.tes_to_use[j,:], self.colors[j,:], self.frac_saturation[j], self.frac_sat_time[j] = self.do_saturation_diagnostic(self.tod[j], self.data_date[j], do_plot)
            pickle.dump([self.tes_to_use, self.frac_saturation, len(self.timeaxis), self.frac_sat_time], open(self.path_FP+'/{}_Saturated_TES_statistics.pkl'.format(self.data_date[0].split('_')[0]),'wb'))
                
        return

# #################### NOISE ESTIMATE ##################################################

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
            spectrum_f, freq_f = ft.power_spectrum(time[i], adu[i, :], rebin=True)
            spectrum_density.append(spectrum_f)
            mean_level = np.mean(spectrum_f[np.abs(freq_f) > (3*np.max(freq_f) / 4)])
            samplefreq = 1. / (time[1] - time[0])
            estimate[i] = (np.sqrt(mean_level * samplefreq / 2))
        
        if do_plot:
            #do plot
            f_to_use = (freq_f<=20.)
            self.plot_focal_plane(freq_f[f_to_use], np.array(spectrum_density)[:,f_to_use], colors_to_use, plot_suptitle='{}: power spectrum'.format(data_date_full),
                             path_and_filename=self.path_FP+'/FocalPlane_powerspectrum_{}'.format(data_date_full), the_xscale='log', the_yscale='log')
        
        return estimate, np.array(spectrum_density), freq_f
    
    def tes_noise_estimate_and_spectrum(self, do_plot=True):
        '''
        Wrapper to the -hf_noise_estimate- function, i.e. the power spectral density computation. 
        If there is more than one dataset, it calls the -hf_noise_estimate- for each dataset.
        '''
        
        #if self.colors doesn't exist, instantiate it and then do plot
        if not isinstance(self.colors, np.ndarray):
            #instantiate color array
            self.tes_saturation_state(do_plot=False)
        
        print('-----> Power spectrum computation ')
        if isinstance(self.data_date, str): #there's only one dataset
            self.noise_level, self.spectrum_density, self.frequency = self.hf_noise_estimate_and_spectrum(self.time_fp, self.tod, self.data_date, self.colors, do_plot)
        else: #there is more than one dataset. loop over all of them 
            noise_level = []
            spectrum_density = []
            freq = []
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                noise_level_tmp, spectrum_density_tmp, freq_tmp  = self.hf_noise_estimate_and_spectrum(self.time_fp[j], self.tod[j], self.data_date[j], self.colors[j], do_plot)
                noise_level.append(noise_level_tmp)
                spectrum_density.append(spectrum_density_tmp)
                freq.append(freq_tmp)
           
            self.noise_level = np.array(noise_level)
            self.spectrum_density = np.array(spectrum_density)
            self.frequency = np.array(freq)
        return

# #################### FLUX JUMPS ##################################################    

    def discard_tods(self, ncomp, eps):
        '''
        Mathias's function to detect flux jumps
        '''
        pca = PCA(n_components=ncomp)
        X_pca = pca.fit_transform(self.tod)
        Z = DBSCAN(eps=eps).fit(X_pca)
        return X_pca, Z

    def plot_scatter_PCA(self, X, Z, figsize=(10, 8)):
        '''
        Mathias's function to plot flux jumps
        '''
        unique_labels = set(Z.labels_)
        core_samples_mask = np.zeros_like(Z.labels_, dtype=bool)
        core_samples_mask[Z.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        plt.figure(figsize=figsize)
        for k, col in zip(unique_labels, colors):
            index = np.where(Z.labels_ == k)[0]
            plt.scatter(X[index, 0], X[index, 1], s=20, c=col, label=f'{k}')
        plt.legend()
        close()
        
    
    def do_fluxjump_diagnostic(self, time, adu, data_date_full, tes_to_use, colors_to_use, fraction_saturated_tes, do_plot=True):
        '''
        This function performs the flux jump detection on the focal plane of the given dataset.
        It selects only the not-saturated detectors and calls the FluxJumps class, which loops over all the selected detectors.
        (Optionally) it plots the focal plane timeline with : flux jump in yellow, saturated detectors in red, good detectors in green (and thermo TES are in black).
        Returns:
        - tes_to_use : array of Bool; True if TES is GOOD, False if TES has flux jumps or is saturated or a thermometer
        - colors : array of strings; 'g' if the TES is GOOD, 'y' if TES has flux jumps, 'r' if TES is saturated, 'k' if TES is a thermometer
        - frac_flux : (float) fraction of detectors with (at least one) flux jumps 
        Input:
        - time: (array) timestamps
        - adu: (array) the detectors ADU timeline 
        - data_date_full : string with the label of the dataset
        - colors_to_use : array of string, containing the color (state) of each detector. Saturated and thermos have to be discarded.
        - fraction_saturated_tes : (float) fraction of saturated detectors
        - do_plot: if True, plots the focal plane
        '''
        #initialize fluxjump class
        fluxjumps_class = FluxJumps(data_date_full, use_verbose = self.use_verbose, doplot = self.use_verbose)
       
        #apply flux jump detection on not-saturated detectors
        num_of_det = adu.shape[0]
        for i in range(num_of_det):
            Tesnum = i+1
            print('Doing TES :', Tesnum)
            if tes_to_use[i] == True:
                num_of_jumps, jumps_start_idx, jumps_end_idx = fluxjumps_class.fluxjumps_analysis(time, adu[i], Tesnum) #, size=size_to_use, doplot=do_plot, verbose=use_verbose)
                if num_of_jumps > 0 :
                    tes_to_use[i] = False
                    colors_to_use[i] = 'y'                    
            else:
                print('Skipping TES = ', Tesnum) 
        
        tes_with_fj = (colors_to_use=='y')
        frac_flux = np.sum(tes_with_fj) / num_of_det
        print(frac_flux*100)
        
        #plot focal plane
        if do_plot:
            suptitle_to_use = '{} : FP timeline. {:.2f}% TES are saturated; {:.2f}% TES have flux jumps'.format(data_date_full,
                                                                                                    fraction_saturated_tes*100,
                                                                                                    frac_flux*100)
            savefig_path_and_filename = self.path_FP+'/Focal_plane_saturation_and_jumps_{}'.format(data_date_full)
            self.plot_focal_plane(self.time_fp, adu, colors_to_use, plot_suptitle=suptitle_to_use, path_and_filename=savefig_path_and_filename)

        return tes_to_use, colors_to_use, frac_flux
        
        
    def tes_fluxjump_state(self, do_plot=True):
        '''
        Wrapper to the -do_fluxjump_diagnostic- function, i.e. the flux jump detection function. 
        If there is more than one dataset, it calls the -do_fluxjump_diagnostic- for each dataset.
        '''
        #if self.colors doesn't exist, instantiate it : do saturation analysis first.
        if not isinstance(self.colors, np.ndarray):
            #instantiate color array
            self.tes_saturation_state(do_plot=False)
        
        print()
        print('-----> Flux jump detection')
        if isinstance(self.data_date, str): #there's only one dataset
            self.tes_to_use, self.colors, self.frac_flux = self.do_fluxjump_diagnostic(self.time_fp, self.tod, self.data_date, self.tes_to_use ,self.colors, self.frac_saturation, do_plot) 
        else: #there is more than one dataset. loop over all of them
            self.frac_flux = np.zeros(self.num_datasets) 
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.tes_to_use[j,:], self.colors[j,:], self.frac_flux[j] = self.do_fluxjump_diagnostic(self.time_fp[j], self.tod[j], self.data_date[j], self.tes_to_use[j], self.colors[j], self.frac_saturation[j], do_plot)
                
        return


# #################### FOCAL PLANE PLOTS ##################################################

    def plot_focal_plane(self, x_data, tes_y_data, colors_plot, plot_suptitle=None, path_and_filename=None, frac_sat_time=None, the_xscale='linear', the_yscale='linear'):
        '''
        This function plots some data (timeline, spectrum) over the entire qubic focal plane.
        Input: 
        - x_data: an array of data running on the x axis (time, frequencies, etc.)
        - y_data: an array of data running on the y axis (ADU, power, etc.)
        - colors_plot: an array of 256 (or number of detectors) elements. Each element is a string with the color of that detector ('g' for good TESs, 'r' for saturation, etc.)
        - plot_suptitle (optional): string with the title of the plot
        - path_and_filename (optional): string with namefile of the plot. The function will save a png and a pdf version of the FP
        - frac_sat_time (optional): array with the percent saturation time for each detector. If passed in input, it will be written on the detector slot. Default is None.
        - the_xscale and the_yscale : 'linear' or 'log' (default is linear) ; to choose the scale of the x and y axes
        '''
        dictfilename = 'global_source_oneDet.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        d['synthbeam'] = 'CalQubic_Synthbeam_Calibrated_Multifreq_FI.fits'
        q = qubic.QubicInstrument(d)
        
        color_backgd_sat_state = cm.Wistia(np.linspace(0,1,100+1)) #jet
        norm = mpl.colors.Normalize(vmin=0, vmax=100)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Wistia)
        cmap.set_array([])

        fig = figure(figsize=(30, 30))
        bar=progress_bar(self.num_det, 'Display focal plane')
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
                ax = subplot(17,17, place_graph)
                #############################
                #### Do your plot here ######
                if x_data.ndim==1:
                    plot(x_data, tes_y_data[idx_tes, :], color=colors_plot[idx_tes])
                else:
                    plot(x_data[idx_tes], tes_y_data[idx_tes, :], color=colors_plot[idx_tes])                 
                #############################   
                annotate('{}'.format(num_tes), xy=(0, 0),  xycoords='axes fraction', fontsize=10, color='black',
                         fontstyle='italic', fontweight='bold', xytext=(0.05,0.85))#,backgroundcolor=bgcol)
                if any('powerspectrum' in path_and_filename):
                    vlines(0.4, np.min(tes_y_data[idx_tes, :]), np.max(tes_y_data[idx_tes, :]), color='k', linestyle='dashed')
                    vlines(1.73, np.min(tes_y_data[idx_tes, :]), np.max(tes_y_data[idx_tes, :]), color='k', linestyle='dashed')
                    vlines(0.003, np.min(tes_y_data[idx_tes, :]), np.max(tes_y_data[idx_tes, :]), color='k', linestyle='dotted')
                    vlines(0.00005, np.min(tes_y_data[idx_tes, :]), np.max(tes_y_data[idx_tes, :]), color='k', linestyle='dashdot')
                    #xlim([0,10])
                    
                if any('saturation' in path_and_filename) and colors_plot[idx_tes]=='r':
                    ax.set_facecolor(color_backgd_sat_state[int(round(frac_sat_time[idx_tes]*100))])
                    ax.set_alpha(0.4)
                    annotate('{:.2f}%'.format(frac_sat_time[idx_tes]*100), xy=(0, 0),  xycoords='axes fraction', fontsize=14, color='black',
                             fontstyle='italic', fontweight='bold', xytext=(0.2,0.3))#, backgroundcolor=) #bgcol
                
                xscale(the_xscale)
                yscale(the_yscale)
                bar.update()

                k+=1
                
        tight_layout()
        if any('saturation' in path_and_filename):
            cb_ax = fig.add_axes([0.01, 0.092, 0.3, 0.02])
            c = colorbar(cmap, orientation="horizontal", cax=cb_ax)
            c.set_label('% of saturation time', fontsize=27)#, loc='left')
            c.ax.tick_params(labelsize=26)
        
        #Save image
        if path_and_filename is not None:
            savefig(path_and_filename+'.png')
            #savefig(path_and_filename+'.pdf')
        close('all')
        return
    
    def plot_raw_focal_plane(self):
        '''
        This function plots the raw timeline of the entire focal plane
        '''
        colors_plot = np.array(['c' for i in range(self.num_det)])
        colors_plot[128:] = 'b'
        thermo_tes_idx = np.array([3,35,67,99])
        #colors_plot[~thermo_tes_idx] = 'g'
        colors_plot[thermo_tes_idx] = 'k'
        colors_plot[thermo_tes_idx+128] = 'k'
        
        if isinstance(self.data_date, str): #len(self.a) == 1:
            plot_suptitle = '{} : FP timeline'.format(self.data_date)
            print('Plotting :', plot_suptitle)
            path_and_filename = self.path_FP+'/Focal_plane_{}'.format(self.data_date)
            self.plot_focal_plane(self.timeaxis, self.tod, colors_plot, plot_suptitle, path_and_filename)
        else:
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                plot_suptitle = '{} : FP timeline'.format(self.data_date[j])
                print('Plotting:', plot_suptitle)
                path_and_filename = self.path_FP+'/Focal_plane_{}'.format(self.data_date[j])
                self.plot_focal_plane(self.timeaxis[j], self.tod[j], colors_plot, plot_suptitle, path_and_filename)
        return
    

# #################### VISUAL DIAGNOSTIC PLOTS ##################################################
    
    def plot_tod_and_temp_monitor(self, a, adu, data_date):
        '''
        This function returns a 4-subplot image with the TOD from 2 reference TESs overplot to a temperature monitor, either the 300mK or 1K stage monitor on a double-y axis with the TOD in blue and the temperature in red.
        The reference TES from asic 1 (TES 95) is shown on the left; the reference TES from asic 2 (TES 183) is shown on the right.
        A png image is saved.
        '''
        
        #get HK temp data. Interpolate them on the TOD axis.
        OneK_stage = a.get_hk('AVS47_1_ch1')
        TES_stage = a.get_hk('AVS47_1_CH2')
        time_hk = a.get_hk(data='RaspberryDate',hk='EXTERN_HK').copy()
                
        TES_stage_interp = np.interp(self.timeaxis, time_hk, TES_stage)
        OneK_stage_interp = np.interp(self.timeaxis, time_hk, OneK_stage)
        
        time_sci = self.timeaxis.copy()
        time_sci -= time_sci[0] #convert time into minutes
        time_sci /= 60 #(60*60)
        
        #define tod mask corresponding to patological events (e.g. sensor behaving erratically)
        timest_out_oneK = (OneK_stage_interp > 2) | (OneK_stage_interp < 0.9)
        timest_out_TESstage = (TES_stage_interp > 0.5) | (TES_stage_interp < 0.2)
        
        #plot
        figure(figsize=(20,10))
        suptitle(data_date+' : TOD and temperature monitors', y=1)
        
        #plot TES asic 1 and 300mK monitor
        subplot(2,2,1)
        title('TES = {}'.format(self.num_ref_tes_asic1), y=0.92, backgroundcolor= 'silver')
        plot(time_sci, adu[self.num_ref_tes_asic1-1,:], color='b', marker='D', markersize=0.2*12)
        ylabel('TOD [ADU]', color='b')
        twinx() 
        plot(time_sci, TES_stage_interp, color='r', marker='D', markersize=0.2*10)
        ylabel('TES Stage [K]', color='r')
        
        #plot TES asic 1 and 1K monitor
        subplot(2,2,3)
        xlabel('Time [min.]')
        plot(time_sci, adu[self.num_ref_tes_asic1-1,:], color='b', marker='D',markersize=0.2*12)
        ylabel('TOD [ADU]', color='b')
        twinx()
        plot(time_sci, OneK_stage_interp, color='orange', marker='D',markersize=0.2*10)
        ylabel('1K Stage [K]', color='orange')

        
        #plot TES asic 2 and 300mK monitor
        subplot(2,2,2)
        title('TES = {}'.format(self.num_ref_tes_asic2), y=0.92, backgroundcolor= 'silver')
        plot(time_sci, adu[self.num_ref_tes_asic2-1,:], color='b', marker='D',markersize=0.2*12)
        ylabel('TOD [ADU]', color='b')
        twinx() 
        plot(time_sci, TES_stage_interp, color='r', marker='D',markersize=0.2*10)
        ylabel('TES Stage [K]', color='r')
        
        #plot TES asic 2 and 1K monitor
        subplot(2,2,4) 
        plot(time_sci, adu[self.num_ref_tes_asic2-1,:], color='b', marker='D',markersize=0.2*12)
        ylabel('TOD [ADU]', color='b')
        xlabel('Time [min.]')
        twinx() 
        plot(time_sci, OneK_stage_interp, color='orange', marker='D',markersize=0.2*10)
        ylabel('1K Stage [K]', color='orange')
        
        tight_layout()
        savefig(self.path_FP +'/{}_TOD_and_temp_1K_300mK_monitors.png'.format(data_date))
        #savefig('./{}_1K_300mK_monitors.pdf'.format(self.data_date))
        
        if any(timest_out_oneK) or any(timest_out_TESstage):
            #redo the plot with problematic/instable temperature timestamps removed
            print('--------> Patological events in the temperature sensors have been found! Plotting again w/o those!')
            figure(figsize=(20,10))
            suptitle(data_date+' : TOD and temperature monitors wo patological events', y=1)

            #plot TES asic 1 and 300mK monitor
            subplot(2,2,1)
            title('TES = {}'.format(self.num_ref_tes_asic1), y=0.92, backgroundcolor= 'silver')
            plot(time_sci[~timest_out_TESstage], adu[self.num_ref_tes_asic1-1,~timest_out_TESstage], color='b', marker='D', markersize=0.2*12)
            ylabel('TOD [ADU]', color='b')
            twinx() 
            plot(time_sci[~timest_out_TESstage], TES_stage_interp[~timest_out_TESstage], color='r', marker='D', markersize=0.2*10)
            ylabel('TES Stage [K]', color='r')

            #plot TES asic 1 and 1K monitor
            subplot(2,2,3)
            xlabel('Time [min.]')
            plot(time_sci[~timest_out_oneK], adu[self.num_ref_tes_asic1-1,~timest_out_oneK], color='b', marker='D',markersize=0.2*12)
            ylabel('TOD [ADU]', color='b')
            twinx()
            plot(time_sci[~timest_out_oneK], OneK_stage_interp[~timest_out_oneK], color='orange', marker='D',markersize=0.2*10)
            ylabel('1K Stage [K]', color='orange')


            #plot TES asic 2 and 300mK monitor
            subplot(2,2,2)
            title('TES = {}'.format(self.num_ref_tes_asic2), y=0.92, backgroundcolor= 'silver')
            plot(time_sci[~timest_out_TESstage], adu[self.num_ref_tes_asic2-1,~timest_out_TESstage], color='b', marker='D',markersize=0.2*12)
            ylabel('TOD [ADU]', color='b')
            twinx() 
            plot(time_sci[~timest_out_TESstage], TES_stage_interp[~timest_out_TESstage], color='r', marker='D',markersize=0.2*10)
            ylabel('TES Stage [K]', color='r')

            #plot TES asic 2 and 1K monitor
            subplot(2,2,4) 
            plot(time_sci[~timest_out_oneK], adu[self.num_ref_tes_asic2-1,~timest_out_oneK], color='b', marker='D',markersize=0.2*12)
            ylabel('TOD [ADU]', color='b')
            xlabel('Time [min.]')
            twinx() 
            plot(time_sci[~timest_out_oneK], OneK_stage_interp[~timest_out_oneK], color='orange', marker='D',markersize=0.2*10)
            ylabel('1K Stage [K]', color='orange')

            tight_layout()
            savefig(self.path_FP +'/{}_filtered_TOD_and_temp_1K_300mK_monitors.png'.format(data_date))
        else:
            print('--------> No patological events in the temperature sensors have been found!')
        
        close('all')
        return

    
    def get_plot_tod_and_temp_monitor(self):
        '''
        Wrapper to the -plot_tod_and_temp_monitor- function. If there is more than one dataset, it calls the function for each dataset.
        '''
        print()
        print('-----> Visual diagnostic: TOD and temperature monitor')
        if isinstance(self.data_date, str): #there's only one dataset
            self.plot_tod_and_temp_monitor(self.a, self.tod, self.data_date) 
        else: #there is more than one dataset. loop over all of them
            for j in tqdm(range(self.num_datasets)):
                print('Doing dataset nr. {}'.format(j+1))
                self.plot_tod_and_temp_monitor(self.a[j], self.tod[j], self.data_date[j])                
        return
# #################### WRAPPER FOR GLOBAL ANALYSIS ##################################################

    def analysis(self, do_plot=True):
        '''
        Wrapper function that calls all the diagnostic methods and performs a global analysis in a (semi-)automatic way
        '''
        
        #plot the timelines over the focal plane
        self.plot_raw_focal_plane()
        
        #plot reference TODs together with temperature monitor for visual diagnostic
        self.get_plot_tod_and_temp_monitor()
        
        #saturation diagnostic
        self.tes_saturation_state(do_plot)
        
        #fluxjumps diagnostic
        self.tes_fluxjump_state(do_plot)
        
        #power spectrum diagnostic
        self.tes_noise_estimate_and_spectrum(do_plot)
        
    #def analysis(self):

        # Perform method1
        #self.method1()

        # Perform method2
        #self.method2()

        # Perform method2
        #self.method2()        

