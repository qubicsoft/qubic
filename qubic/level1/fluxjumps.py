'''
$Id: fluxjumps.py
$auth: Belen Costanza <belen@fcaglp.unlp.edu.ar>
$auth: Steve Torchinsky <satorchi@apc.in2p3.fr>
$created: Mon 23 Jan 2023 15:21:44 CET
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

check data for flux jumps, assign flag, or make correction and assign the "corrected" flag

code adapted from Jupyter notebook "Automatic_flux_jump_versions" by Belen Costanza
'''
from scipy.signal import argrelextrema, find_peaks, find_peaks_cwt, savgol_filter 
import bottleneck as bn
from sklearn.cluster import DBSCAN


def haar_filter(inputarray, filterwidth=100):
    '''
    The Haar filter is used to find flux jumps
    '''
    outputarray = np.zeros(len(inputarray))
    xf = bn.move_median(inputarray, filterwidth)[filterwidth:]   
    outputarray[filterwidth+filterwidth//2:-filterwidth+filterwidth//2] = xf[:-filterwidth] - xf[filterwidth:]
    return outputarray


def identify_fluxjumps(todarray, filterwidth=100, threshold=0.2e6):
    '''
    Function that detects the jumps by Belen Costanza (version 2)
    
    return jumpstart = beginning of the jump 
           jumpend = final of the jump 
           jumpfree = number of samples between the end of a jump and the beginning of the next jump 
           njumps = number of clusters (ie. number of jumps)
    '''

    retval = {} # dictionary for return values
    
    
    tod_sav = savgol_filter(todarray,window_length=401,polyorder=3, mode='nearest')
    
    #threshold using that the difference is going to be a value between 0.25e6, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6
    
    tod_haar_sav = haar_filter(tod_sav, filterwidth)
    tod_haar = haar_filter(todarray,filterwidth)
    
    #if you don't find any jump then don't make the savgol filter and try with the unfiltered TOD
    if max(abs(tod_haar_sav)) < threshold:
        #if the tood haar also is less than 0.2, probably is not a jump 
        if max(abs(tod_haar)) < 0.2e6:
            number = 0
            print('no jump')
            return number
        else:
            jumps = np.abs(tod_haar) > threshold
            use_savgol = False #don't use the savgol filter
    else:
        use_savgol = True
        print('use savgol')
        jumps = np.abs(tod_haar_sav) > threshold
    
    idx = np.arange(len(todarray))
    # According to the requested threshold, it seems to me that you are losing values, that's why I subtract 50 from the index
    # (Steve:  I don't understand this.)
    idx_jumps = idx[jumps] - 50
    time_jumps = tt[jumps]
        
    
    clust = DBSCAN(eps=filterwidth//5, min_samples=1).fit(np.reshape(idx_jumps, (len(idx_jumps),1)))
    njumps = np.max(clust.labels_)+1
    jumpstart = np.zeros(njumps, dtype=int) 
    jumpend = np.zeros(njumps, dtype=int)
    for i in range(njumps):
        jumpstart[i] = np.min(idx_jumps[clust.labels_ == i])
        jumpend[i]= np.max(idx_jumps[clust.labels_ == i])
        
    delta = jumpend - jumpstart
    print(delta)
    for i in range(len(delta)):
        if delta[i] < 80: 
            number = 0 
            print('no jump')
            return number 
        
    jumpfree = np.zeros(njumps-1)
    for j in range(njumps):
        #print(j)
        if j < njumps-1:
            jumpfree[j] = jumpstart[j+1]-jumpend[j]


    
    retval['jumpstart'] = jumpstart
    retval['jumpend'] = jumpend
    retval['jumpfree'] = jumpfree
    retval['njumps'] = njumps
    retval['use_savgol'] = use_savegol
    retval['tod savgol'] = tod_haar_sav
    retval['tod'] = tod_haar
    
    return retval



