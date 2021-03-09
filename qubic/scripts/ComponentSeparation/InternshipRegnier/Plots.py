import numpy as np
import matplotlib.pyplot as plt
import healpy as hp




class Plots :
	
    def __init__(self) :
    
        pass

    def all_components(map1, ifreq, rot, reso, title) :
        
        """
        
        Definition which plot all the components (I, Q, U) of a map. We can plot for only one frequency ifreq at resolution reso.
        
        --------
        inputs :
            - maps1 : Map that we want to plot -> Be careful of the order of components (ifreq, istk, Npix)
            - ifreq : Index of the frequency
            - title : Title of each map

        --------
        output :
            - (1x3) maps for each Stokes parameter

        """

        Stokes = ['I', 'Q', 'U']
        
        plt.figure()
        for istk in range(3) :
            
            hp.gnomview(map1[ifreq, istk, :], rot = rot, reso = reso, title = title + '\n {} Stokes parameter'.format(Stokes[istk]), sub = (1, 3, istk+1))
        plt.show()

    def diff_2_maps(map1, map2, coverage, ifreq, rot, reso, title1, title2) :

        """

        """

        Stokes = ['I', 'Q', 'U']
        seenpix = (coverage > (0.1*np.max(coverage)))
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
                              title = 'Residue \n {} Stokes parameter'.format(Stokes[istk]), sub = (3, 3, 3*istk+3), min = -nsig * sig/10, max = nsig * sig/10)

        plt.show()


    def diff_2_hist(map1, map2, coverage, ifreq, label1, label2, bins = 100) :

        Stokes = ['I', 'Q', 'U']
        seenpix = (coverage > (0.1*np.max(coverage)))
        color = ['red', 'black']
        a_map1 = []
        b_map1 = []
        a_map2 = []
        b_map2 = []
        plt.figure()
        for istk in range(3) :
            
            plt.subplot(2, 3, istk+1)
            a1, b1, _ = plt.hist(map1[ifreq, istk, seenpix], histtype = 'step', color = color[0], label = label1, bins = bins)
            a2, b2, _ = plt.hist(map2[ifreq, istk, seenpix], histtype = 'step', color = color[1], label = label2, bins = bins)
            plt.title('{} Stokes parameter'.format(Stokes[istk]))

            a_map1.append(a1)
            b_map1.append(b1)
            a_map2.append(a2)
            b_map2.append(b2)

            plt.legend(loc = 'upper left', fontsize = 'small')

        plt.show()

        return a_map1, b_map1, a_map2, b_map2

    def plot_residue(a1, b1, a2, b2, lim = 500) :
        Stokes = ['I', 'Q', 'U']
        plt.figure()
        for istk in range(3) :
            plt.subplot(2, 3, istk+1)
            plt.plot(b1[istk][:-1], a1[istk] - a2[istk], '-r')
            plt.axhline(0, ls = '--', color = 'black')
            plt.title('{} Stokes parameter'.format(Stokes[istk]))
            plt.ylim(-lim, lim)
        plt.show()

        

        

        

        

        

        

