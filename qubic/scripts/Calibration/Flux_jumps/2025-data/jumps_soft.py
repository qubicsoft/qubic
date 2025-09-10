# Author: Belén Costanza, Lucas Merlo, Claudia Scóccola

import numpy as np
import sys,os
import glob
import scipy.ndimage.filters as f
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt, savgol_filter 
import bottleneck as bn
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class saturation:

    def __init__(self, sat_value = 4.19e6, TES_number = 256):

        """Class for the saturation function
        
        Params

        sat_value = limit of the TOD to be considered saturated
        TES_number = number of TES in the focal plane

        Output 

        ok = array of True and False, True if it's saturated, False if's not
        bad_idx = idx of the saturated TES in the focal plane
        frac = fraction of the TOD saturated in the TES
        number = number of TES saturated

        """

        self.sat_value = sat_value
        self.TES_number = TES_number

    def detect_saturation(self, todarray):

        ok = np.ones(self.TES_number, dtype=bool)
        maxis = np.max(abs(todarray), axis=1)
        upper_satval = self.sat_value
        lower_satval = -self.sat_value

        frac = np.zeros((self.TES_number))
        size = todarray.shape[1]

        for i in range(len(todarray)):

            mask1 = todarray[i] > upper_satval
            mask2 = todarray[i] < lower_satval
            fraction = (np.sum(mask1)+np.sum(mask2))/size
            frac[i] = fraction

            if frac[i] ==0:
                ok[i] = True #good, no saturated
            elif frac[i] > 0.:
                ok[i] = False #bad, saturated
            else:
                ok[i] = True

        bad_idx = np.array(np.where(ok==False))
        bad_idx = np.reshape(bad_idx, bad_idx.shape[1])
        number = len(bad_idx)

        return ok, bad_idx, frac, number


class fluxjumps:

    def __init__(self, thr, window_size): 

        """ Class for detection of discontinuities in the data 

        Params: 

        thr = threshold or list of thresholds of the haar filter, if a flux it's higher than the treshold, the time sample would be considered as a fux jump candidate
        window_size = size of the bottleneck moving median 

        Return: 

        nc = number of flux jumps
        xc = time sample of the inition of the flux jump
        xcf = time sample of the end of the flux jump

        """

        self.thr = np.array(thr) 
        self.window_size = window_size

    def haar_function(self, todarray):

        tod_haar = np.zeros(todarray.size)
        xf = bn.move_median(todarray, self.window_size)[self.window_size:]
        tod_haar[self.window_size+self.window_size//2:-self.window_size+self.window_size//2] = xf[:-self.window_size] - xf[self.window_size:]

        return tod_haar

    def find_candidates(self, tod_haar, thr):

        number = 0
        jumps = 0
        thr_used = 0

        #iterate over the amplitude thresholds

        for j,thr_val in enumerate(thr):
            print(j, thr_val)

            if number == 0: #haven't detected any jump yet

                if max(abs(tod_haar)) < thr_val:
                    print('No jump')

                else: #there's a jump
                    number += 1
                    thr_used = thr_val
                    print('Found jump')
                    jumps = (abs(tod_haar) >= thr_val) #True in the index where there is flux jumps
            else:
                pass

        return jumps, thr_used

    def clusters(self, todarray, jumps):

        idx = np.arange(len(todarray))
        idx_jumps = idx[jumps]

        if idx_jumps.size > 1:
            clust = DBSCAN(eps=self.window_size//5, min_samples=1).fit(np.reshape(idx_jumps, (len(idx_jumps),1)))
            nc = np.max(clust.labels_)+1
        else:
            nc = 0.
            idx_jumps = 0.
            clust = 0.

        return nc, idx_jumps, clust

    def initial_start_end(self, nc, idx_jumps, tod_haar, thr_used, clust):

        xc = np.zeros(nc, dtype=int)
        xcf = np.zeros(nc, dtype=int)

        for i in range(nc):

            idx_jumps_from_thr = idx_jumps[clust.labels_ == i]
            idx_delta_end_jump = np.where( abs(tod_haar[idx_jumps_from_thr[-1]:]) < thr_used*0.05 )[0][0]
            idx_delta_start_jump = idx_jumps_from_thr[0] - np.where( abs(tod_haar[:idx_jumps_from_thr[0]]) < thr_used*0.05 )[0][-1]
            #idx_delta_start_jump = np.where( tod_haar[:idx_jumps_from_thr[0]] < thr_used*0.05 )[0][-1]
            xc[i] = idx_jumps_from_thr[0] - idx_delta_start_jump
            xcf[i] = idx_jumps_from_thr[-1] + idx_delta_end_jump
            #delta = xcf - xc

        return xc, xcf #delta 

    def unique(self, xc, xcf):

        xc_unique = np.unique(xc)
        xcf_unique = np.unique(xcf)
        nc_unique = len(xc_unique)

        return nc_unique, xc_unique, xcf_unique

    def change_values(self, xc, xcf, max_gap=10):

        xc2 = []
        xcf2 = []

        i = 0
        while i < len(xc):
            xini = xc[i]
            xfin = xcf[i]
            j = i + 1

            # Agrupar mientras estén dentro del margen
            while j < len(xc) and xc[j] - xfin <= max_gap:
                xfin = xcf[j]
                j += 1

            xc2.append(xini)
            xcf2.append(xfin)
            i = j

        return xc2, xcf2


    def jumps_detection(self, tt, todarray, consec = True, nc_cond=False):

        tod_haar = self.haar_function(todarray) #1. make the haar filter of the raw TOD

        jumps, thr_used= self.find_candidates(tod_haar, self.thr) #2. if the haar filter is higher than a threshold then is a jump (iterate through an array of possible thresholds)

        nc, idx_jumps, clust = self.clusters(todarray, jumps) #3. Cluster the jumps and find the number of jumps detected in every TES

        if nc_cond == True:
            if nc > 11:
                print(nc)
                thr = np.array([3e5]) #higher value for the treshold
                #tod_sav = savgol_filter(todarray,window_length=401,polyorder=3, mode='nearest')
                tod_haar = self.haar_function(todarray)
                jumps, thr_used = self.find_candidates(tod_haar, thr)
                nc, idx_jumps, clust = self.clusters(todarray, jumps)

        if nc==0:
            xc=0
            xcf=0
            return nc, xc, xcf, thr_used

        xc, xcf = self.initial_start_end(nc, idx_jumps, tod_haar, thr_used, clust) #5. find the beginning and the end of a jump, also the size of the jump
        nc_unique, xc_unique, xcf_unique = self.unique(xc, xcf)

        if consec == True:
            xc_unique, xcf_unique = self.change_values(xc_unique, xcf_unique)
            nc_unique = len(xc_unique)

        return nc_unique, xc_unique, xcf_unique, thr_used

class correction:

    def __init__(self, region_off = 5, region_amp = 10, change_mode = "const"):

        """ Class that calculates the amplitude of the flux jump found and applies the correction to the TOD

        Params:

        region_off = 5
        region_amp = 10

        Return:         

        """

        self.region_off = region_off
        self.region_amp = region_amp
        self.change_mode = change_mode 

    def calculate_amplitude(self, tt, todarray, xc, xcf, nc):

        offset = np.zeros(nc)
        idx = np.arange(len(todarray))
        for i in range(len(xc)):
            offset[i] = np.median(todarray[xcf[i]:xcf[i]+self.region_off])-np.median(todarray[xc[i]-self.region_off:xc[i]])

        return offset

    def move_offset(self, todarray, offset, xc, xcf, nc):

        tod_new = todarray.copy()
        idx = np.arange(len(todarray))  

        if nc == 1:
            initial = xcf[0]
            final = idx[-1]+1
            tod_new[initial:final] = todarray[initial:final] - offset[0]
        else:
            for i in range(len(xcf)-1,-1,-1):
                initial = xcf[i]
                final = idx[-1]+2
                tod_new[initial:final]=tod_new[initial:final]-offset[i]

        return tod_new

    def changesignal_init(self, tod_new, xc, xcf):

        y_cor = tod_new.copy()
        std = np.std(tod_new[:xc[0]])
        #mean = np.mean(tod_new[:xc[0]])

        for i in range(len(xc)):
            mean = np.mean(tod_new[xc[i]-self.region_amp:xc[i]])
            ynew=np.random.normal(mean, std, len(tod_new[xc[i]:xcf[i]]))
            y_cor[xc[i]:xcf[i]] = ynew

        return y_cor

    def changesignal_noise(self, tod_new, xc, xcf):

        y_cor = tod_new.copy()

        for i in range(len(xc)):
            std = np.std(tod_new[xc[i]-self.region_amp:xc[i]])# * (len(y[:xini]))/(len(y[:xini])-1)
            mean = np.mean(tod_new[xc[i]-self.region_amp:xc[i]])
            ynew=np.random.normal(mean, std, len(tod_new[xc[i]:xcf[i]]))
            y_cor[xc[i]:xcf[i]] = ynew

        return y_cor

    def constrained_realization(self, tod_new, xini, xfin):

        xini = int(xini)
        xfin = int(xfin)

        y = tod_new.copy()
        L = xfin - xini

        region_pre = min(L//2, xini) # no ir mas alla del principio
        region_post = min(L - region_pre, len(y)-xfin) # no ir mas alla del final

        total_need = L 
        still_need = total_need - (region_pre + region_post)

        if still_need > 0.:
            #sacamos de donde haya mayor cantidad, para que las regiones sumen L 
            extra_pre = min(still_need, region_pre)
            extra_post = still_need - extra_pre
            region_pre += extra_pre 
            region_post += extra_post

        pre = y[xini - region_pre:xini]
        post = y[xfin:xfin + region_post]
        
        local_data = np.concatenate([pre, post])

        N = len(local_data)
        #power spectrum to the data before and after the jump

        fft_data = np.fft.rfft(local_data - np.mean(local_data))
        power_spectrum = np.abs(fft_data) ** 2

        #random phase realization
        random_phases = np.exp(1j * 2 * np.pi * np.random.rand(len(fft_data)))
        new_fft = np.sqrt(power_spectrum) * random_phases
        sim_signal = np.fft.irfft(new_fft, n=N)

        sim_trunc = sim_signal[:L]

        start_val = y[xini - 1]
        end_val = y[xfin]
        sim_trunc -= np.mean(sim_trunc)  # centrado
        sim_trunc = sim_trunc / np.std(sim_trunc) * np.std(pre)  # escalar

        #linear transition to the edges
        window = np.linspace(0, 1, L)
        sim_trunc = sim_trunc + (1 - window) * (start_val - sim_trunc[0]) + window * (end_val - sim_trunc[-1])
        y[xini:xfin] = sim_trunc

        return y


    def correct_TOD(self, todarray, offset, xc, xcf, nc):

        tod_new = self.move_offset(todarray, offset, xc, xcf, nc)
        if self.change_mode == "init":
            tod_corr = self.changesignal_init(tod_new, xc, xcf)
        elif self.change_mode == "noise":
            tod_corr = self.changesignal_noise(tod_new, xc, xcf)
        elif self.change_mode == "const":
            tod_corr = tod_new 
            for i in range(nc):
                tod_corr = self.constrained_realization(tod_corr, xc[i], xcf[i])

        return tod_corr


class DT:

    def __init__(self, thr_count = 600, thr_amp = 2e5, tol=1e2, depth = True, depth_number = 0):

        """ Class for the decision tree to calculate levels between flux jumps and improves the correction

        Params:

        thr_count = threshold in the number of repetitions for a level  
        thr_amp = threshold in the amplitude between levels
        tol =  tolerance of the signal to be asigned to a spececifi DT level 
        depth = if depth equal to True, then the max_depth hyperaramenter is going to be the number of flux jumps found, otherwise will have the same number for TES. Default equal to True 
        depth_number = if depth = False, then depth_number for the depth of the DT should be assigned

        Return: 

        """

        self.thr_count = thr_count
        self.thr_amp = thr_amp
        self.tol = tol 
        self.depth = depth
        self.depth_number = depth_number

    def define_model(self, tt, todarray, num):

        depth = max(3, num)
        x = tt.reshape(-1,1)
        regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
        regressor.fit(x, todarray)
        ypred = regressor.predict(x)

        return ypred

    def uniqueindex(self, ypred):

        predunique, index = np.unique(ypred, return_index=True)
        index = np.sort(index)
        predunique = ypred[index]

        count = np.zeros((len(predunique)), dtype=int)
        for i in range(len(predunique)):
            count[i] = len(np.where(ypred==predunique[i])[0])

        return predunique, index, count

    def count_filter(self, predunique, index, count):

        filpred = predunique[count > self.thr_count]
        filindex = index[count > self.thr_count]
        filcount = []
        for i in range(len(count)):
            if count[i] > self.thr_count:
                filcount.append(count[i])

        return filpred, filindex, filcount

    def amplitude_filter(self, filpred, filindex, filcount):

        ampnew = [] ##save amplitudes higher than self.thr_amp
        valini = [] ##save the initial level 
        valfin = [] ##save the final level, the amplitude between those levels is ampnew
        indexini = [] ##index of the initial level
        indexfin = [] ##index of the final level
        #countini = [] ##
        #countfin = [] ##
        for i in range(len(filpred)- 1):
            amp = filpred[i+1] - filpred[i]
            if abs(amp) > self.thr_amp:
                ampnew.append(amp)
                valini.append(filpred[i])
                valfin.append(filpred[i+1])
                indexini.append(filindex[i])
                indexfin.append(filindex[i+1])
                #countini.append(filcount[i])
                #countfin.append(filcount[i+1])

        return ampnew, valini, valfin, indexini, indexfin#, countini, countfin

    def calculate_start_end(self, todarray, valini, valfin, indexfin):

        start = np.zeros(len(valini), dtype=int)
        end = np.zeros(len(valini), dtype=int)
        for i in range(len(valini)):
            index1 = np.where((todarray < valini[i]+self.tol) & (todarray > valini[i]-self.tol))[0]
            index2 = np.where((todarray < valfin[i]+self.tol) & (todarray > valfin[i]-self.tol))[0]

            if len(index1) == 0 or len(index2) == 0:
                tol2 = 50*self.tol
                index1 = np.where((todarray < valini[i]+tol2) & (todarray > valini[i]-tol2))[0]
                index2 = np.where((todarray < valfin[i]+tol2) & (todarray > valfin[i]-tol2))[0]

            end[i] = index2[np.where(index2 > indexfin[i])[0]][0]
            start[i] = np.max(index1[index1 < end[i]])

            if len(valini) > 1:
                if end[i-1] == start[i]:
                    start[i] += 1


            #if i==0:
            #    end[i]=index2[np.where(index2 > indexfin[0])[0]][0]
            #    start[i] = np.max(index1[index1 < end[i]]) 
            #else:
            #    end[i] = index2[index2>end[i-1]][0]
            #
            #    start[i] = np.max(index1[index1 < end[i]])

        return start, end

    def calculate_start_end2(self, todarray, valini, valfin, indexini, indexfin):

        start = np.zeros(len(valini), dtype=int)
        end = np.zeros(len(valini), dtype=int)
        for i in range(len(valini)):
            index1 = np.where((todarray < valini[i]+self.tol) & (todarray > valini[i]-self.tol))[0]
            index2 = np.where((todarray < valfin[i]+self.tol) & (todarray > valfin[i]-self.tol))[0]

            if len(index1) == 0 or len(index2) == 0:
                tol2 = 50*self.tol
                index1 = np.where((todarray < valini[i]+tol2) & (todarray > valini[i]-tol2))[0]
                index2 = np.where((todarray < valfin[i]+tol2) & (todarray > valfin[i]-tol2))[0]

            candidates_end = index2[index2 > indexfin[i]]
            if len(candidates_end) > 0:
                end[i] = candidates_end[0]
            else:
                end[i] = indexfin[i] + 1

            candidates_start = index1[(index1 < end[i]) & (index1 > indexini[i])]
            if len(candidates_start) > 0:
                start[i] = np.max(candidates_start)
            else:
                start[i] = indexini[i] + 1
            

            if len(valini) > 1:
                if end[i-1] == start[i]:
                    start[i] += 1

        return start, end


    def change_values(self, xc, xcf, max_gap=10):

        order = np.argsort(xc)
        xc = np.array(xc)[order]
        xcf = np.array(xcf)[order]

        xc2 = []
        xcf2 = []

        i = 0
        while i < len(xc):
            xini = xc[i]
            xfin = xcf[i]
            j = i + 1

            # Agrupar mientras estén dentro del margen
            while j < len(xc) and xc[j] - xfin <= max_gap:
                xfin = xcf[j]
                j += 1

            xc2.append(xini)
            xcf2.append(xfin)
            i = j

        return xc2, xcf2

    def calculate_levels(self, tt, todarray, nc, consec=True):

        if self.depth == False:
            num = self.depth_number
        else:
            num = nc

        ypred = self.define_model(tt, todarray, num)
        ypred_unique, index, count = self.uniqueindex(ypred)
        ypred_unique, index, count = self.count_filter(ypred_unique, index, count)
        amplitude, valini, valfin, indexini, indexfin = self.amplitude_filter(ypred_unique, index, count)
        xc, xcf = self.calculate_start_end2(todarray, valini, valfin, indexini, indexfin)

        if consec == True:
            xc_unique, xcf_unique = self.change_values(xc, xcf)

        return xc_unique, xcf_unique#, amplitude
















