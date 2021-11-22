import healpy as hp
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.optimize import curve_fit
import pickle

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from qubic import camb_interface as qc

# ### To run that script: $ python FastSim_CreateNoiseFiles.py config alpha signoise

def get_maps_from_louise(directory, nfsub, config):
    residuals_patch = np.load(directory + f'residuals_{nfsub}bands_{config}.npy')
    seenmap = np.load(directory + f'seenmap_{nfsub}bands_{config}.npy')
    coverage_patch = np.load(directory + f'coverage_{nfsub}bands_{config}.npy')

    nside = 256
    residuals = np.zeros((nfsub, 12 * nside ** 2, 3))
    residuals[:, seenmap, :] = residuals_patch

    coverage = np.zeros(12 * nside ** 2)
    coverage[seenmap] = coverage_patch

    return residuals, coverage, seenmap


def plot_maps(residuals, seenmap, center, nfsub, config, dirsave=None):
    stn = ['I', 'Q', 'U']
    myrmsI = np.std(residuals[0, seenmap, 0])
    nn = 3
    plt.figure(figsize=(20, 40))
    for i in range(nfsub):
        for s in range(3):

            hp.gnomview(residuals[i, :, s],
                        min=-nn * myrmsI,
                        max=nn * myrmsI,
                        sub=(nfsub, 3, i * 3 + s + 1),
                        title=f'{stn[s]} - Band {i+1}/{nfsub}',
                        rot=center,
                        reso=15,
                        notext=True)
    if dirsave is None:
        plt.show()
    else:
        plt.savefig(dirsave + f'residuals_{nfsub}bands_' + config + '.pdf', format='pdf')
        plt.close()

    return


def get_RMS_profile(residuals, config, nfsub, dirsave=None):
    xx, yyI, yyQ, yyU = qss.get_angular_profile(residuals[0, :, :],
                                                nbins=30,
                                                separate=True,
                                                center=center)
    pix_size = hp.nside2resol(256, arcmin=True)
    meanvalI = np.mean(yyI[xx < 10]) * pix_size
    meanvalQU = np.mean((yyQ[xx < 10] + yyQ[xx < 10]) / 2) * pix_size

    plt.figure()
    plt.plot(xx, yyI * pix_size, 'o', label='I')
    plt.plot(xx, yyQ * pix_size, 'o', label='Q')
    plt.plot(xx, yyU * pix_size, 'o', label='U')

    plt.axhline(y=meanvalI,
                label=r'I RMS = {0:5.1f} $\mu K.arcmin$'.format(meanvalI),
                color='r', ls=':')
    plt.axhline(y=meanvalQU,
                label=r'QU RMS = {0:5.1f} $\mu K.arcmin$'.format(meanvalQU),
                color='m', ls=':')

    plt.xlabel('Degrees from center of the field')
    plt.ylabel(r'Noise RMS $[\mu K.arcmin]$')
    plt.title('QUBIC End-To-End - ' + config + ' - Nptg = {}'.format(nptg))
    plt.legend(fontsize=11)
    plt.xlim(0, 20)
    plt.ylim(0, meanvalQU * 2)

    if dirsave is None:
        plt.show()
    else:
        plt.savefig(dirsave + f'RMSprofile_{nfsub}bands_' + config + '.pdf', format='pdf')
        plt.close()
    return


def noise_profile_fitting(residuals, coverage, nfsub, config, nbins, dirsave=None):
    plt.figure(figsize=(16, 10))
    myfitcovs = []
    for isub in range(nfsub):
        sqn = np.int(np.sqrt(nfsub))
        if (sqn ** 2) != nfsub:
            sqn += 1
        plt.subplot(sqn, sqn, isub + 1)
        xx, yyfs, fitcov = qss.get_noise_invcov_profile(residuals[isub, :, :],
                                                        coverage,
                                                        QUsep=True,
                                                        nbins=nbins,
                                                        label='End-To-End sub={}/{}'.format(isub + 1, nfsub),
                                                        fit=True,
                                                        norm=False,
                                                        allstokes=True,
                                                        doplot=True)
        plt.legend(fontsize=9)
        myfitcovs.append(fitcov)
    if dirsave is None:
        plt.show()
    else:
        plt.savefig(dirsave + f'NoiseProfileFitting_{nfsub}bands_' + config + '.pdf', format='pdf')
        plt.close()
    return myfitcovs


def get_nunu_covariance(residuals, coverage, nfsub, config, dirsave=None):
    cI, cQ, cU, fitcov, noise_norm = qss.get_cov_nunu(residuals, coverage, QUsep=True)
    corr_mats = [cI, cQ / 2, cU / 2]
    plt.figure(figsize=(16, 6))
    stn = ['I', 'Q/2', 'U/2']
    bla = np.max([np.abs(np.min(np.array(corr_mats))), np.max(np.array(corr_mats))])
    mini = -bla
    maxi = bla
    for s in range(3):
        plt.subplot(1, 3, 1 + s)
        plt.imshow(corr_mats[s], vmin=mini, vmax=maxi, cmap='bwr')
        plt.colorbar(orientation='horizontal')
        plt.title('End-To-End Cov {} nsub={}'.format(stn[s], nfsub))

    if dirsave is None:
        plt.show()
    else:
        plt.savefig(dirsave + f'nunuCovariance_{nfsub}bands_' + config + '.pdf', format='pdf')
        plt.close()
    return cI, cQ, cU


def ctheta_measurement(residuals, coverage, myfitcovs, nfsub, config, alpha, dirsave=None):
    plt.figure(figsize=(16, 6))
    fct = lambda x, a, b, c: a * np.sin(x / b) * np.exp(-x / c)
    thth = np.linspace(0, 180, 1000)
    allcth = []
    allclth = []
    allresults = []
    pixgood = coverage > 0.1
    for i in range(nfsub):
        corrected_qubicnoise = qss.correct_maps_rms(residuals[i, :, :], coverage, myfitcovs[i])
        th, thecth, _ = qss.ctheta_parts(corrected_qubicnoise[:, 0], pixgood, 0, 20, 20, nsplit=5, degrade_init=128,
                                      verbose=False)
        okfit = np.isfinite(thecth)
        results = curve_fit(fct, th[okfit][1:], (thecth[okfit][1:] / thecth[0]), maxfev=100000, ftol=1e-7, p0=[0, 1, 1])
        allcth.append(thecth)
        allresults.append(results)

        plt.subplot(1, 2, 1)
        p = plt.plot(th, allcth[i] / allcth[i][0], 'o', label='End-To-End Sub {}'.format(i + 1))
        plt.plot(thth, fct(thth, *allresults[i][0]), color=p[0].get_color())
        plt.axhline(y=0, color='k', ls=':')
        plt.xlim(0, 20)
        plt.legend(fontsize=9)
        plt.xlabel(r'$\theta$ [deg]')
        plt.ylabel(r'$C(\theta$)')

        # ### Convert to Cl and display
        ctheta = fct(thth, *allresults[i][0])
        ctheta[0] = 1
        lll, clth = qc.ctheta_2_cell(thth, ctheta, lmax=1024)
        clth = (clth - 1) * alpha + 1
        allclth.append(clth)

        plt.subplot(1, 2, 2)
        plt.plot(lll, clth, label='End-To-End Sub {}'.format(i + 1), color=p[0].get_color())
        plt.axhline(y=1, color='k', ls=':')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell$')

    if dirsave is None:
        plt.show()
    else:
        plt.savefig(dirsave + f'Ctheta_{nfsub}bands_' + config + '.pdf', format='pdf')
        plt.close()
    return allresults, allcth, allclth, lll, clth


global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
# Repository with full pipeline simulations
datadir = os.environ['DATA_SPECTROIM'] + 'Data_for_FastSimulator/'

# Repository where plots will be saved
dirsave = os.environ['DATA_SPECTROIM'] + 'Data_for_FastSimulator/plots/'

all_nf = [1, 2, 3, 4, 5, 6, 7, 8]
center = np.array([0, 0])
nptg = 10000
config = sys.argv[1] # TD150 or FI150 or FI220
nbins = 50

for nfsub in all_nf:
    print(f'\n STARTING nfsub = {nfsub}')
    residuals, coverage, seenmap = get_maps_from_louise(datadir, nfsub, config)
    # Look at the maps
    plot_maps(residuals, seenmap, center, nfsub, config, dirsave=dirsave)

    # RMS angular profile
    get_RMS_profile(residuals, config, nfsub, dirsave=dirsave)

    # Noise Profile Fitting
    myfitcovs = noise_profile_fitting(residuals, coverage, nfsub, config, nbins, dirsave=dirsave)

    # nunu Covariance
    cI, cQ, cU = get_nunu_covariance(residuals, coverage, nfsub, config, dirsave=dirsave)

    # C(theta) Measurement
    # See notebook called "2pt-Correlation Function" for an empirical explanation of alpha
    alpha = np.float(sys.argv[2])
    allresults, allcth, allclth, lll, clth = ctheta_measurement(residuals, coverage, myfitcovs,
                                                                nfsub, config, alpha, dirsave=dirsave)

    ### The option below will save the average over sub-bands of the Clth
    ### However significant residuals exist on the end-to-end simulations as of today, and
    ### they would be reproduced here, while they are likely to be caused by some issue
    # clth_tosave = np.mean(np.array(allclth), axis=0)
    # ### As a result we will instead save the nsub=1 correlation function
    if nfsub == 1:
        clth_tosave = clth

    # Plot the saved one
    plt.plot(lll, clth_tosave, lw=3, color='k', label='saved')
    plt.legend(fontsize=9)
    plt.title(f'Cl theta {nfsub} bands - ' + config)
    plt.savefig(dirsave + f'ClthetaSaved_{nfsub}bands_' + config + '.pdf', format='pdf')
    plt.close()
    plt.show()

    # ============== Save pickle files for the Fast Simulator ======================
    # ############## Comment this is you don't want to overwrite files ! #########################

    data = {'nfsub': nfsub,
            'CovI': cI,
            'CovQ': cQ,
            'CovU': cU,
            'alpha': np.float(sys.argv[2]),
            'signoise': np.float(sys.argv[3]),
            'effective_variance_invcov': myfitcovs,
            'clnoise': clth_tosave}
    name = 'DataFastSimulator_' + config + '_nfsub_{}.pkl'.format(nfsub)
    pickle.dump(data, open(global_dir + 'doc/FastSimulator/Data/' + name, "wb"))

datacov = {'coverage': coverage}
name = 'DataFastSimulator_' + config + '_coverage.pkl'
pickle.dump(datacov, open(global_dir + 'doc/FastSimulator/Data/' + name, "wb"))
