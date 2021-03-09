import numpy as np
import matplotlib.pyplot as plt
import healpy as hp




class Plots :
	
    def __init__(self) :
    
        pass
      
    def diff_2_maps(map1, map2, seenpix, ifreq, rot, reso, title1, title2) :

        """
        
        Definition which plot all the (I, Q, U) maps.
        
        --------
        inputs :
            - map1 & map2 : two maps that we want to compare
            - seenpix : array type which show us which pixel is seen
            - ifreq : frequency of the map

        --------
        output :
            - (1x3) maps for each Stokes parameter
            
        """

        Stokes = ['I', 'Q', 'U']

        nsig = 3  
        plt.figure()
        for istk in range(3) :
            
            sig = np.std(map1[ifreq, istk, seenpix])

            hp.gnomview(map1[ifreq, istk, :], rot = rot, reso = reso,
                              title = title1 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+1), min = -nsig * sig, max = nsig * sig)
            hp.gnomview(map2[ifreq, istk, :], rot = rot, reso = reso, 
                              title = title2 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+2), min = -nsig * sig, max = nsig * sig)
            residue = map1[ifreq, istk, :] - map2[ifreq, istk, :]
            hp.gnomview(residue, rot = rot, reso = reso, 
                              title = 'Residue \n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+3), min = -nsig * sig, max = nsig * sig)

        plt.show()


    def diff_2_hist(map1, map2, seenpix, ifreq, label1, label2, bins = 100) :
        
        """
        
        Definition which plot all the histograms (I, Q, U) of a map.
        
        --------
        inputs :
            - map1 & map2 : two maps that we want to compare
            - seenpix : array type which show us which pixel is seen
            - ifreq : frequency of the map

        --------
        output :
            - (1x3) histograms for each Stokes parameter
            
        """
        
        Stokes = ['I', 'Q', 'U']
        color = ['red', 'black']
        plt.rc('figure', figsize=(16, 10))
        for istk in range(3) :
            
            plt.subplot(2, 3, istk+1)
            _, _, _ = plt.hist(map1[ifreq, istk, seenpix], histtype = 'step', color = color[0], label = label1, bins = bins)
            _, _, _ = plt.hist(map2[ifreq, istk, seenpix], histtype = 'step', color = color[1], label = label2, bins = bins)
            plt.title('{} Stokes parameter'.format(Stokes[istk]))

            plt.legend(loc = 'best', fontsize = 'small')

        plt.show()
    
    
    
    def all_components(map_list, ifreq, rot, reso, title_list, nb_component = 3) :

        Stokes = ['I', 'Q', 'U']
        
        plt.figure()
        for istk in range(3) :
            for i in range(nb_component) :
                hp.gnomview(map_list[i][ifreq, istk, :], rot = rot, reso = reso, 
                           sub = (3, nb_component, nb_component*istk+(i+1)), title = Stokes[istk] + ' - ' + title_list[i])
        
        
        
        
        
        
        
        
        
        
        
