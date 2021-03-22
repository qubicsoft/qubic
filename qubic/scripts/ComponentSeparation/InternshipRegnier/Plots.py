import numpy as np
import matplotlib.pyplot as plt
import healpy as hp




class Plots :
	
    def __init__(self) :
    
        pass
      
    def diff_2_maps(map1, map2, seenpix, ifreq, rot, reso, title1, title2, Stokesparameter = 'IQU') :

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

        if Stokesparameter == 'IQU' :
            plt.rc('figure', figsize=(16, 10))
            for istk in range(3) :
            
                sig = np.std(map1[ifreq, istk, seenpix])

                hp.gnomview(map1[ifreq, istk, :], rot = rot, reso = reso,
                              title = title1 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+1), min = -nsig * sig, max = nsig * sig)
                hp.gnomview(map2[ifreq, istk, :], rot = rot, reso = reso,
                              title = title2 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+2), min = -nsig * sig, max = nsig * sig)
                residue = map1[ifreq, istk, :] - map2[ifreq, istk, :]
                
                hp.gnomview(residue, rot = rot, reso = reso,
                              title = 'Residue \n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+3), min = -nsig * sig, max = nsig * sig)

        elif Stokesparameter == 'QU' :
            plt.rc('figure', figsize=(16, 10))
            for istk in range(2) :
            
                sig = np.std(map1[ifreq, istk+1, seenpix])

                hp.gnomview(map1[ifreq, istk+1, :], rot = rot, reso = reso,
                              title = title1 + '\n {} Stokes parameter'.format(Stokes[istk+1]), sub = (2, 3, 3*istk+1), min = -nsig * sig, max = nsig * sig)
                hp.gnomview(map2[ifreq, istk, :], rot = rot, reso = reso,
                              title = title2 + '\n {} Stokes parameter'.format(Stokes[istk+1]), sub = (2, 3, 3*istk+2), min = -nsig * sig, max = nsig * sig)
                residue = map1[ifreq, istk+1, :] - map2[ifreq, istk, :]
                hp.gnomview(residue, rot = rot, reso = reso,
                              title = 'Residue \n {} Stokes parameter'.format(Stokes[istk+1]), sub = (2, 3, 3*istk+3), min = -nsig * sig, max = nsig * sig)

        elif Stokesparameter == 'I' :
            plt.rc('figure', figsize=(16, 10))
            for istk in range(1) :
            
                sig = np.std(map1[ifreq, istk, seenpix])

                hp.gnomview(map1[ifreq, istk, :], rot = rot, reso = reso,
                              title = title1 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (1, 3, 3*istk+1), min = -nsig * sig, max = nsig * sig)
                hp.gnomview(map2[ifreq, istk, :], rot = rot, reso = reso,
                              title = title2 + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (1, 3, 3*istk+2), min = -nsig * sig, max = nsig * sig)
                residue = map1[ifreq, istk, :] - map2[ifreq, istk, :]
                hp.gnomview(residue, rot = rot, reso = reso,
                              title = 'Residue \n {} Stokes parameter'.format(Stokes[istk]), sub = (1, 3, 3*istk+3), min = -nsig * sig, max = nsig * sig)

        else :

             raise TypeError('Incorrect Stokes parameter')

        plt.show()


    def diff_2_hist(map1, map2, seenpix, ifreq, label1, label2, ranges = None, bins = 100, Stokesparameter = 'IQU') :
        
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
        color = ['red', 'blue']

        if Stokesparameter == 'IQU' :

            plt.rc('figure', figsize=(28, 15))
            for istk in range(3) :
            
                plt.subplot(2, 3, istk+1)
                _, _, _ = plt.hist(map1[ifreq, istk, seenpix], histtype = 'step', color = color[1], label = label1, bins = bins, range = ranges[istk])
                _, _, _ = plt.hist(map2[ifreq, istk, seenpix], histtype = 'step', color = color[0], label = label2, bins = bins, range = ranges[istk])
                plt.title('{} Stokes parameter'.format(Stokes[istk]))

                plt.legend(loc = 'best', fontsize = 'small')

        elif Stokesparameter == 'QU' :

            plt.rc('figure', figsize=(25, 12))
            for istk in range(2) :
            
                plt.subplot(2, 3, istk+1)
                _, _, _ = plt.hist(map1[ifreq, istk+1, seenpix], histtype = 'step', color = color[1], label = label1, bins = bins, range = ranges[istk+1])
                _, _, _ = plt.hist(map2[ifreq, istk, seenpix], histtype = 'step', color = color[0], label = label2, bins = bins, range = ranges[istk+1])
                plt.title('{} Stokes parameter'.format(Stokes[istk+1]))

                plt.legend(loc = 'best', fontsize = 'small')

        else :

            plt.rc('figure', figsize=(20, 10))
            for istk in range(1) :
            
                plt.subplot(2, 3, istk+1)
                _, _, _ = plt.hist(map1[ifreq, istk, seenpix], histtype = 'step', color = color[1], label = label1, bins = bins, range = ranges[istk])
                _, _, _ = plt.hist(map2[ifreq, istk, seenpix], histtype = 'step', color = color[0], label = label2, bins = bins, range = ranges[istk])
                plt.title('{} Stokes parameter'.format(Stokes[istk]))

                plt.legend(loc = 'best', fontsize = 'small')

        plt.show()
    
    
    
    def all_components(map_list, ifreq, rot, reso, title_list, nb_component = 3) :

        Stokes = ['I', 'Q', 'U']
        
        plt.rc('figure', figsize=(16, 10))
        for istk in range(3) :
            for i in range(nb_component) :
                hp.gnomview(map_list[i][ifreq, istk, :], rot = rot, reso = reso, 
                           sub = (3, nb_component, nb_component*istk+(i+1)), title = Stokes[istk] + ' - ' + title_list[i])
        
        
        
        
        
        
        
        
        
        
        
