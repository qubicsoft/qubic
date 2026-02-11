import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import qubic
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


#### This file is a toolbox to produce plots of components maps and some converged parameters.

def plot_one_map(map, center, reso, sub=(1, 1, 1), path=None, **kwargs):

    plt.figure()
    hp.gnomview(map, rot=center, reso=reso, sub=sub, notext=True, **kwargs)
    
    if path is not None:
        plt.savefig(path)
    plt.tight_layout()
    plt.close()

def plot_all_maps(input, output, center, reso, path=None, **kwargs):

    plt.figure()
    hp.gnomview(input, rot=center, reso=reso, sub=(1, 3, 1), notext=True, **kwargs)
    hp.gnomview(output, rot=center, reso=reso, sub=(1, 3, 2), notext=True, **kwargs)
    r = input - output
    hp.gnomview(r, rot=center, reso=reso, sub=(1, 3, 3), notext=True, **kwargs)
    
    if path is not None:
        plt.savefig(path)

    plt.tight_layout()
    plt.close()

def plot_panel(map1, map2, center, reso, pixok, path=None, nsig=3, **kwargs):
    
    input = map1.copy()
    output = map2.copy()
    n = input.shape[0]
    
    #input[:, ~pixok] = hp.UNSEEN
    #output[:, ~pixok] = hp.UNSEEN
    plt.figure(figsize=(8, 10))
    k=1
   
    for j in range(n):
        if j == 0:
            nsigg = nsig
            title0 = 'Input'
            title1 = 'Output'
            title2 = 'Residual'
        else:
            nsigg = nsig/2
            title0 = ''
            title1 = ''
            title2 = ''

        sig = np.std(input[j, pixok])

        hp.gnomview(input[j], rot=center, reso=reso, sub=(n, 3, k), notext=True, min=-nsigg*sig, max=nsig*sig, title=title0, **kwargs)
        k+=1
        hp.gnomview(output[j], rot=center, reso=reso, sub=(n, 3, k), notext=True, min=-nsigg*sig, max=nsig*sig, title=title1, **kwargs)
        k+=1
        r = input[j] - output[j]
        #r[~pixok] = hp.UNSEEN
        sig = np.std(r[pixok])

        hp.gnomview(r, rot=center, reso=reso, sub=(n, 3, k), notext=True, title=title2, min=-nsig*sig, max=nsig*sig, **kwargs)
        k+=1
    
    if path is not None:
        plt.savefig(path)

    plt.tight_layout()
    plt.close()

def plot_convergence_rms(theta, istk, label=None, ylabel=None, path=None, log=True):

    """
    
    theta is (N, comp, stk)
    
    """

    N, ncomp, nstk = theta.shape
    ite = np.arange(1, N+1, 1)
    plt.figure(figsize=(12, 8))

    for i in range(ncomp):
        if label is not None:
            lab = label[i]
        plt.plot(ite, theta[:, i, istk], label=lab)

    if log:
        plt.yscale('log')

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.grid(True)
    plt.legend(frameon=False, fontsize=12)

    if path is not None:
        plt.savefig(path)

    plt.close()

def plot_convergence(theta, truth, log=False, ylabel=None, path=None):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)

    num_theta, num_samples = theta.shape
    for i in range(num_theta):
        plt.plot(theta[i])
    plt.axhline(truth, ls='--', color='black')
        
    plt.grid(True)
    if log:
        plt.yscale('log')
        
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Iteration', fontsize=12)



    plt.subplot(2, 1, 2)
    for i in range(num_theta):
        plt.plot(abs(theta[i] - truth[i]))

    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=12)
    plt.ylabel(r'$\Delta$', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.grid(True)


    if path is not None:
        plt.savefig(path)


    

    plt.close()


def plot_convergence_allbeta(theta, truth, log=False, ylabel=None, path=None):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)

    num_theta, num_samples = theta.shape
    allc = []
    for i in range(num_theta):
        np.random.seed(i)
        c = (np.random.random(), np.random.random(), np.random.random())
        allc += [c]
    for i in range(num_theta):
        
        plt.plot(theta[i], color=allc[i])
        plt.axhline(truth[i], ls='--', color=allc[i])

        
    plt.grid(True)
    if log:
        plt.yscale('log')
        
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Iteration', fontsize=12)



    plt.subplot(2, 1, 2)
    for i in range(num_theta):
        plt.plot(abs(theta[i] - truth[i]))

    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=12)
    plt.ylabel(r'$\Delta$', fontsize=12)
    plt.xlabel('Iteration', fontsize=12)
    plt.grid(True)


    if path is not None:
        plt.savefig(path)

    plt.close()
