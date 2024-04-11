import numpy as np
from qubicpack.qubicfp import qubicfp
import sys,os
import numpy as np
import glob

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
import scipy.ndimage.filters as f

from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt, savgol_filter 

import bottleneck as bn
from sklearn.cluster import DBSCAN

#----------------------------------------------------------------------------------
#Identify Saturated TES
def saturation(todarray): 
    
    #returns  ok = array of True and False, True if it's saturated, False if's not
    #         bad_idx = idx of the saturated TES in the focalplane 
    #         frac_sat_pertes = fraction of the TOD saturated in the TES
    #         number = number of TES saturated 
    
    ok = np.ones(256,dtype=bool)
    maxis = np.max(abs(todarray), axis=1)
    upper_satval = 4.19e6
    lower_satval = -4.19e6
    
    frac_sat_pertes = np.zeros((256))
    size = todarray.shape[1]

    for i in range(len(todarray)): 
        mask1 = todarray[i] > upper_satval
        mask2 = todarray[i] < lower_satval
        frac = (np.sum(mask1)+np.sum(mask2))/size
        frac_sat_pertes[i] = frac
    
        if frac_sat_pertes[i] ==0:
            ok[i] = True #good, no saturated
        elif frac_sat_pertes[i] > 0.:
            ok[i] = False #bad, saturated
        else:
            ok[i] = True
    
    bad_idx = np.array(np.where(ok==False))
    bad_idx = np.reshape(bad_idx, bad_idx.shape[1])        
    number = len(bad_idx)    
        
    return ok, bad_idx, frac_sat_pertes, number

#-------------------------------------------------------------------------------------
#Necessary functions for flux detections
def haar(x, size=100):
    out = np.zeros(x.size)
    xf = bn.move_median(x, size)[size:]   
    out[size+size//2:-size+size//2] = xf[:-size] - xf[size:]
    return out

def find_jumps(tod_haar, thr):    #Elenia's version, function that iterate through many thresholds 
    number = 0
    jumps = 0
    thr_used = 0 
        #iterate over the amplitude thresholds
    for j,thr_val in enumerate(thr):
        if number == 0: #haven't detected any jump yet
            if max(abs(tod_haar)) < thr_val:
                print('No jump')
            else: #there's a jump
                number += 1
                thr_used = thr_val
                print('Found jump')
                jumps = (abs(tod_haar) >= thr_val) #save the timestamp of the jump
                threshold_TES = thr_val
        else:
            pass
    return jumps, thr_used

def clusters(todarray,jumps):
    size=130
    idx = np.arange(len(todarray))
    idx_jumps = idx[jumps]
    if idx_jumps.size > 1:
        clust = DBSCAN(eps=size//5, min_samples=1).fit(np.reshape(idx_jumps, (len(idx_jumps),1)))
        nc = np.max(clust.labels_)+1
    else: 
        nc = 0.
        idx_jumps = 0.
        clust = 0.
    return nc, idx_jumps, clust

def star_end(nc, idx_jumps, tod_haar, thr_used, clust):
    xc = np.zeros(nc, dtype=int) 
    xcf = np.zeros(nc, dtype=int)
    
    for i in range(nc):
        idx_jumps_from_thr = idx_jumps[clust.labels_ == i]
        idx_delta_end_jump = np.where( abs(tod_haar[idx_jumps_from_thr[-1]:]) < thr_used*0.05 )[0][0]   
        idx_delta_start_jump = idx_jumps_from_thr[0] - np.where( abs(tod_haar[:idx_jumps_from_thr[0]]) < thr_used*0.05 )[0][-1]
        #idx_delta_start_jump = np.where( tod_haar[:idx_jumps_from_thr[0]] < thr_used*0.05 )[0][-1]
        xc[i] = idx_jumps_from_thr[0] - idx_delta_start_jump
        xcf[i] = idx_jumps_from_thr[-1] + idx_delta_end_jump
        
    delta = xcf - xc
    return xc, xcf, delta 

def offset_funct(tt, todarray, xc, xcf, number, region=5, order=1):
    #tod_new = todarray.copy()
    offset_lin = np.zeros(number)
    idx = np.arange(len(todarray))
    for i in range(len(xc)): 
        offset_lin[i] = np.median(todarray[xcf[i]:xcf[i]+region])-np.median(todarray[xc[i]-region:xc[i]])
    
    pol = np.zeros(len(xc))
    offset_pol = np.zeros(len(xc))
    for i in range(len(xc)):        
        tp = tt[xc[i]-region:xcf[i]+region]
        adup = todarray[xc[i]-region:xcf[i]+region]
        z = np.polyfit(tp, adup, order)
        p = np.poly1d(z)
        pol = p(tp)
        offset_pol[i] = pol[-1]-pol[0]
    
    return offset_lin, offset_pol


def jumps_detection(tt, todarray, offset_cond=False):

    #return nc: number of jumps in the TOD
    #       xc: beginning jumps
    #       xcf: final jumps
    #       delta: size of the jumps

    size = 130
    #thr = np.array([2e5, 1.5e5, 1e5, 5e4, 4e4, 3e4])
    thr = np.array([2e5])
    
    tod_haar = haar(todarray,size) #1. make the haar filter of the raw TOD
    
    jumps, thr_used= find_jumps(tod_haar, thr) #2. if the haar filter is higher than a threshold then is a jump 

   
    nc, idx_jumps, clust = clusters(todarray, jumps) #3. Cluster the jumps and find the number of jumps detected in every TES
    
    if nc==0.:
        xc=0
        xcf=0
        delta=0
        return nc, xc, xcf, delta
    
    if nc > 11:                                         #4. If the number of jumps is higher than 11 put a bigger threshold in order to avoid the confusion with noise
        thr = np.array([4e5])
    #    tod_sav = savgol_filter(todarray,window_length=401,polyorder=3, mode='nearest')
    #    tod_haar_sav = haar(todarray, size)
        jumps_sav, thr_used = find_jumps(tod_haar, thr)
        nc, idx_jumps, clust = clusters(todarray, jumps_sav)
        if nc==0:
            xc=0
            xcf=0
            delta=0
            return nc, xc, xcf, delta
    
    xc, xcf, delta = star_end(nc, idx_jumps, tod_haar, thr_used, clust) #5. find the beginning and the end of a jump, also the size of the jump


    if offset_cond == True:
        offset_lin,_ = offset_funct(tt, todarray, xc, xcf, nc)
        xc_list = []
        xcf_list = []
        delta_list = []
        for j in range(len(offset_lin)):
            if abs(offset_lin[j]) > 3.5e5:
                xc_list.append(xc[j])
                xcf_list.append(xcf[j])
                delta_list.append(delta[j])
        nc_list = len(xc_list)
        return nc_list, xc_list, xcf_list, delta_list
    else: 
        return nc, xc, xcf, delta

# +
def three(offset, xc, xcf, nc):
    if nc == 3: 
        offset = offset[:-1]
        xc = xc[:-1]
        xcf = xcf[:-1]
        nc -= 1
    return offset, xc, xcf, nc

def offset_delete(offset, xc, xcf):

#condition in the amplitude expected by a real flux jump  
    offset_corr = []
    xc_corr = []
    xcf_corr = []
    for i in range(len(offset)):
        amplitude_per_tes = offset[i]
        if abs(amplitude_per_tes) > 2.5e5:
            offset_corr.append(offset[i])
            xc_corr.append(xc[i])
            xcf_corr.append(xcf[i])
    nc_corr = len(xc_corr)
    return offset_corr, xc_corr, xcf_corr, nc_corr

def offset_array(offset_corr):
    off = []
    for i in range(len(offset_corr)):
        if len(offset_corr[i]) > 0 and len(offset_corr[i]) < 9:
            off.append(np.reshape(offset_corr[i], (len(offset_corr[i]),1)))
    off = np.concatenate(off)
    return off


# -

#-------------------------------------------------------
####discrimination functions
def redefine_jumps(tt, nc, xc, xcf, delta):
    delta_thr = np.rint(len(tt)/4915.2)
    del_idx = np.reshape(np.array(np.where(delta<delta_thr)),np.array(np.where(delta<delta_thr)).shape[1])
    xc = np.delete(xc, del_idx) #if the amount of time samples is less than 90 probably is not a jump 
    xcf = np.delete(xcf, del_idx)
    nc -= len(del_idx)
    return nc, xc, xcf   

def derivation(tt, todarray, xc, xcf, region=10):
    ini, fin = xc, xcf
    tod_portion, time_portion = todarray[ini-region:fin+region], tt[ini-region:fin+region]
    smooth_tod = savgol_filter(tod_portion, window_length=401, polyorder=4, mode='nearest')
    deriv_tod_smooth = np.diff(smooth_tod)
    deriv_tod_raw = np.diff(tod_portion)
    return time_portion, tod_portion, smooth_tod, deriv_tod_smooth    


# +
#-----------------------------------------------------------
###correction

def correction(todarray, offset, xc, xcf, number, region=100):  
    tod_new = todarray.copy()
    idx = np.arange(len(todarray))  
    if number == 1: 
        initial = xcf[0]
        final = idx[-1]+1
        tod_new[initial:final] = todarray[initial:final] - offset[0]
    else:
        for i in range(len(xcf)-1,-1,-1):
            initial = xcf[i]
            final = idx[-1]+2
            tod_new[initial:final]=tod_new[initial:final]-offset[i]
    return tod_new

def changesignal(y, xini, xend):
    y_cor = y.copy()
    for i in range(len(xini)):        
        std = np.std(y[xini[i]-10:xini[i]])# * (len(y[:xini]))/(len(y[:xini])-1)
        mean = np.mean(y[xini[i]-10:xini[i]])
        ynew=np.random.normal(mean, std, len(y[xini[i]:xend[i]]))
        y_cor[xini[i]:xend[i]] = ynew
    
    return y_cor
# -


