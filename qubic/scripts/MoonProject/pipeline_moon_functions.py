import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft2, ifft2
import sys
import healpy as hp
import time
from scipy.signal import butter, filtfilt, bessel

import fitting as fit
import pickle
from datetime import datetime

import iminuit
from iminuit.cost import LeastSquares

#########################

### General imports
from joblib import Parallel, delayed
from multiprocessing import Manager, Lock
from scipy.signal import medfilt

### Astropy configuration
from astropy.visualization import quantity_support
quantity_support()
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_moon

#### QUBIC IMPORT
from qubicpack.qubicfp import qubicfp
import qubic.lib.Calibration.Qfiber as ft

from qubic.lib import Qdictionary
from qubic.lib.Instrument import Qacquisition

import pipeline_moon_plotting as pmp

#########################

conv_reso_fwhm = 2.35482

#########################
# import matplotlib.style as style
# style.use("/Users/huchet/Documents/phd_code/matplotlib_styles/ah_basic_style.mplstyle")
# plt.rc('text', usetex=False)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{bm}")
# plt.style.use('default')

#########################

def healpix_map(azt, elt, tod, flags=None, flaglimit=0, nside=128, countcut=0, unseen_val=hp.UNSEEN):
    if flags is None:
        flags = np.zeros(len(azt))
    
    ok = flags <= flaglimit 
    return healpix_map_(azt[ok], elt[ok], tod[ok], nside=nside, countcut=countcut, unseen_val=unseen_val)


def healpix_map_(azt, elt, tod, nside=128, countcut=0, unseen_val=hp.UNSEEN):
    ips = hp.ang2pix(nside, azt, elt, lonlat=True)
    mymap = np.zeros(12*nside**2)
    mapcount = np.zeros(12*nside**2)
    for i in range(len(azt)):
        mymap[ips[i]] += tod[i]
        mapcount[ips[i]] += 1
    unseen = mapcount <= countcut
    mymap[unseen] = unseen_val
    mapcount[unseen] = unseen_val
    mymap[~unseen] = mymap[~unseen] / mapcount[~unseen]
    return mymap, mapcount


#######################

def gaussian(x, mu, reso):
		sig = reso / conv_reso_fwhm
		res = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-((x - mu) / sig)**2 / 2)
		return res / np.sum(res) # area under the curve = 1

def gauss2D(Nx, Ny, x0, y0, reso, amp=None, normal=True):
    # don't forget to convert all values (x0, y0, reso) in pixel space
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    if len(reso) == 1:
        reso = np.array([reso, reso])
    sig = reso / conv_reso_fwhm
    res = np.exp(-(x - x0)**2/(2*sig[0]**2)) * np.exp(-(y - y0)**2/(2*sig[1]**2))
    if normal:
        return(res/np.sum(res))
    else:
        return amp*res


def get_new_azel(azt, elt, azmoon, elmoon):
    newazt = (azt - azmoon) * np.cos(np.radians(elt))
    newelt = -(elt - elmoon) # so the Moon is higher than trees in maps (?)
    # newelt = (elt - elmoon)
    return newazt, newelt


def make_coadded_maps_TES(tt, tod, azt, elt, scantype, azmoon, elmoon, nside=256, doplot=True, check_back_forth=False):

    # Inversion in signal
    mytod = -tod.copy()

    # Filter the TOD
    mytod = my_filt(mytod)

    if doplot:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(tt, -tod, label="TOD")
        ax.plot(tt, mytod, label="filtered TOD")
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flux [ADU]")
        plt.tight_layout()
        plt.savefig("tod_filtering.pdf")
        plt.show()
        
    # Map-making

    newazt, newelt = get_new_azel(azt, elt, azmoon, elmoon)

    # newelt = -newelt
    # Calculate center of maps from pointing w.r.t. Moon
    center=[np.mean(newazt), np.mean(newelt)]

    # To compare the map created with only forth scans with the map created with only back scans
    if check_back_forth:
        mapsb_forth, mapcount_forth = healpix_map(newazt[scantype > 0], newelt[scantype > 0], mytod[scantype > 0], nside=nside)
        mapsb_back, mapcount_back = healpix_map(newazt[scantype < 0], newelt[scantype < 0], mytod[scantype < 0], nside=nside)
        plt.figure()
        hp.gnomview(mapsb_forth, reso=10, sub=(1, 3, 1), min=-5e3, max=1.2e4, 
                title="forth scans", rot=center)
        hp.gnomview(mapsb_back, reso=10, sub=(1, 3, 2), min=-5e3, max=1.2e4, 
                title="back scans", rot=center)
        hp.gnomview(mapsb_forth - mapsb_back, reso=10, sub=(1, 3, 3), min=-5e3, max=1.2e4, 
                title="forth - back scans", rot=center)
        plt.show()
        # return mapsb_forth, mapcount_forth, mapsb_back, mapcount_back
    
    mapsb, mapcount = healpix_map(newazt[scantype != 0], newelt[scantype != 0], mytod[scantype != 0], nside=nside)

    # plt.figure()
    # hp.gnomview(testmap, reso=10, sub=(1, 2, 1), min=-5e3, max=1.2e4, 
    #             title="gaussian map", rot=center)
    # hp.gnomview(mapsb, reso=10, sub=(1, 2, 2), min=-5e3, max=1.2e4, 
    #             title="final map", rot=center)
    # plt.show()

    return mapsb, mapcount


# https://stackoverflow.com/questions/14695367/most-efficient-way-to-filter-a-long-time-series-python
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data, *args, **kwargs):
    b, a = butter_bandpass(*args, **kwargs)
    return filtfilt(b, a, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def butter_pseudo_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_lowpass(highcut, fs, order=order)
    data_altered = filtfilt(b, a, data)
    b, a = butter_highpass(lowcut, fs, order=order)
    return filtfilt(b, a, data_altered) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def butter_lowpass(highcut, fs, order=2):
    nyq = 0.5*fs
    high = highcut/nyq
    b, a = butter(order, high, btype='lowpass')
    return b,a

def butter_highpass(lowcut, fs, order=2):
    nyq = 0.5*fs
    low = lowcut/nyq
    b, a = butter(order, low, btype='highpass')
    return b,a

def butter_highpass_filter(data, *args, **kwargs):
    b, a = butter_highpass(*args, **kwargs)
    return filtfilt(b, a, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def bessel_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = bessel(order, [low, high], btype='band')
    return b,a

def bessel_bandpass_filter(data, *args, **kwargs):
    b, a = bessel_bandpass(*args, **kwargs)
    # b, a = butter_bandpass(*args, **kwargs)
    return filtfilt(b, a, data) # no phase but filter applied twice (forwards and backwards), because I use filtfilt instead of lfilter

def my_filt(mytod): # utiliser cette fonction ?
    # Cuts are expressed in Hz, a back and forth scan takes 107.5 seconds
    lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
    # highcut = 2/107.5*100/2 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
    # filt_tod = butter_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=150, order=2) # Hz
    # filt_tod = butter_pseudo_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=150, order=2) # Hz
    filt_tod = butter_highpass_filter(mytod, lowcut=lowcut, fs=150, order=2) # Hz
    # filt_tod = bessel_bandpass_filter(mytod, lowcut=lowcut, highcut=highcut, fs=150, order=3) # Hz
    return filt_tod
        

class gauss2dfit:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy
    def __call__(self, x, pars):
        amp, xc, yc, sig = pars
        mygauss = amp * np.exp(-0.5*((self.xx-xc)**2+(self.yy-yc)**2)/sig**2)
        return np.ravel(mygauss)

class filtgauss2dfit:
    def __init__(self, xx, yy):
        self.xx = xx
        self.yy = yy
        # self.i = 0
    def __call__(self, x, pars):
        amp, xc, yc, sig = pars
        mygauss = amp * np.exp(-0.5*((self.xx - xc)**2+(self.yy - yc)**2)/sig**2)
        mygauss_flat = np.ravel(mygauss)
        # lowcut = 4/107.5 # 4/107.5, i.e. half a forth (or back) scan
        # highcut = 2/107.5*100*5 # 2/107.5*100/4, i.e. approx. 4 % of a forth (or back) scan --> passer à 2% parce que 4% est trop proche de la taille de la Lune (2/107.5*100/6 makes the Moon round but it's fine-tuned for it...)
        # myfiltgauss = butter_bandpass_filter(mygauss_flat, lowcut=lowcut, highcut=highcut, fs=150, order=2) # Hz
        # myfiltgauss = butter_highpass_filter(mygauss_flat, lowcut=lowcut, fs=150, order=2) # Hz

        # Not working yet, has to be TOD and not just a flat map! Anyway, this will be done with the
        # synthesized beam, not a gaussian
        myfiltgauss = my_filt(mygauss_flat)

        # filtgauss2d = np.reshape(myfiltgauss, np.shape(self.xx))
        # if self.i == 0:
        #     plt.figure()
        #     plt.imshow(filtgauss2d)
        #     plt.show()
        #     self.i += 1
        #     sys.exit()
        return myfiltgauss
    

def get_dict(params):
    """QUBIC dictionary.

    Method to modify the qubic dictionary.

    Parameters
    ----------
    key : str, optional
        Can be "in" or "out".
        It is used to build respectively the instances to generate the TODs or to reconstruct the sky maps,
        by default "in".

    Returns
    -------
    dict_qubic: dict
        Modified QUBIC dictionary.

    """
    ### Arguments used when simulating the maps

    args = {
                "npointings": params["QUBIC"]["npointings"],
                "nf_recon": params["QUBIC"]["nrec"],
                "nf_sub": params["QUBIC"]["nsub_in"],
                "nside": params["SKY"]["nside"],
                "MultiBand": True,
                "period": 1,
                "RA_center": params["SKY"]["RA_center"],
                "DEC_center": params["SKY"]["DEC_center"],
                "filter_nu": 150 * 1e9,
                "noiseless": False,
                "beam_shape": 'gaussian',
                #"comm": comm,
                "dtheta": params["QUBIC"]["dtheta"],
                "nprocs_sampling": 1,
                #"nprocs_instrument": comm.size,
                "photon_noise": True,
                "nhwp_angles": 3,
                #'effective_duration':3,
                "effective_duration150": 3,
                "effective_duration220": 3,
                "filter_relative_bandwidth": 0.25,
                "type_instrument": "two",
                "TemperatureAtmosphere150": None,
                "TemperatureAtmosphere220": None,
                "EmissivityAtmosphere150": None,
                "EmissivityAtmosphere220": None,
                "detector_nep": float(params["QUBIC"]["NOISE"]["detector_nep"]),
                "synthbeam_kmax": params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"] #,
                #"synthbeam_fraction": params["QUBIC"]["SYNTHBEAM"]["synthbeam_fraction"],
            }

    dictfilename = "pipeline_demo.dict"
    qubic_dict = Qdictionary.qubicDict()
    qubic_dict.read_from_file(dictfilename)

    for i in args.keys():
        qubic_dict[str(i)] = args[i]
    return qubic_dict

def get_synthbeam(nside=340, xs=401, reso=10):
    # Not finished but the aim is to fit a synthesized beam on the Moon maps
    #### Here we read QUBIC + Planck maps
    # data_both = pickle.load(open(dirFast + 'MC_planck_None.pkl', 'rb'))
    file_both = "MC_CMB_w_Planck_FMM.pkl"
    dirFast_2 = "/Users/huchet/qubic/qubic/scripts/MapMaking/src/FMM/test_Fastsimulator/maps/"
    data_both = pickle.load(open(dirFast_2 + file_both, 'rb'))
    params = data_both["parameters"]
    dict_qubic = get_dict(params)
    dict_qubic["nside"] = nside
    # # qubic = Qacquisition.QubicUltraWideBand(
    # #             dict_out, Nsub=1, Nrec=1
    # #         )
    # qubic_inst = Qinstrument.QubicInstrument(dict_qubic, FRBW=0.25)
    # scene = Qscene.QubicScene(dict_qubic)
    # sb = qubic_inst.get_synthbeam(scene, idet=0)
    # print(np.shape(sb))
    # print(np.sum(sb))
    # synthbeam =  hp.gnomview(sb, reso=reso, return_projected_map=True, xsize=xs, no_plot=True).data
    # print(np.sum(synthbeam))
    from qubic.lib.Qsamplings import get_pointing
    idet = 0
    # sampling = get_pointing(dict_qubic)
    acq = Qacquisition.QubicMultiAcquisitions(dict_qubic, dict_qubic['nf_sub'], 2)
    sb = acq.subacqs[0].instrument[idet].get_synthbeam(acq.subacqs[0].scene)[0]
    hp.gnomview(np.log10(sb/np.max(sb)), rot=[0,90], reso=20, min=-3, title="Synthesized Beam - log scale")
    sys.exit()
    return synthbeam


def get_synthbeam_fit(nsub=16, nside=340, xs=401, reso=10):
    # Not finished but the aim is to fit a synthesized beam on the Moon maps
    # It means it should be fake TOD with synthesized beam
    dictfilename = "pipeline_demo.dict"
    qubic_dict = Qdictionary.qubicDict()
    qubic_dict.read_from_file(dictfilename)

    qubic_dict["nside"] = nside
    qubic_dict["nf_sub"] = nsub
    qubic_dict["MultiBand"] = True
    qubic_dict["filter_nu"] = 150 * 1e9
    qubic_dict["synthbeam_kmax"] = 1
    qubic_dict["noiseless"] = True
    qubic_dict["npointings"] = 2

    # print(qubic_dict["synthbeam_fraction"])

    from qubic.lib.Qsamplings import get_pointing
    idet = 67
    # sampling = get_pointing(dict_qubic)
    acq = Qacquisition.QubicMultiAcquisitions(qubic_dict, nsub=qubic_dict['nf_sub'], nrec=2)
    sb = acq.subacqs[0].instrument[idet].get_synthbeam(acq.subacqs[0].scene)[0]
    for ifreq in range(1, nsub//2):
        sb += acq.subacqs[ifreq].instrument[idet].get_synthbeam(acq.subacqs[0].scene)[0]
    # hp.gnomview(np.log10(sb/np.max(sb)), rot=[0,90], reso=10, min=-3, title="Synthesized Beam - log scale")
    sb_map = hp.gnomview(np.log10(sb/np.max(sb)), rot=[0, 95], reso=4, min=-3, xsize=401, title="Synthesized Beam - log scale", return_projected_map=True, no_plot=True).data
    # hp.gnomview(mm, reso=4, rot=center_map, min=-5e3, max=1.2e4, return_projected_map=True, xsize=xs, no_plot=True).data
    # plt.figure()
    fig, (ax, cax) = plt.subplots(1, 2, width_ratios=(1, 0.05))
    img = ax.imshow(sb_map, vmin=-3, origin="lower")
    fig.colorbar(img, cax=cax)
    # fig.subplots_adjust(hspace=.0)
    plt.show()
    sys.exit()
    return synthbeam


# def img_to_TOD(img, center, pixsize, newazt, newelt):
def img_to_TOD(img, amp_azt, amp_elt, newazt, newelt):
    # To create fake TOD quickly from an input image (not finished)
    Npix = np.shape(img) # vérifier ordre lignes colonnes (lignes = azimuth ou elevation ?)
    azel_arr = []
    # for i, coord in enumerate(["az", "el"]):
    #     imsize = Npix[i] * pixsize[i]
    #     min_coord = center[i] - imsize / 2
    #     max_coord = center[i] + imsize / 2
    #     print(Npix[i], flush=True)
    #     print(pixsize[i])
    #     azel_arr.append(np.linspace(min_coord, max_coord - pixsize[i], Npix[i]) + pixsize[i]/2)
    amplitude = np.array([amp_azt, amp_elt]) # Here amplitude contains the intervals in azimuth and elevation for the pixels' centers
    for i, coord in enumerate(["az", "el"]):
        azel_arr.append(np.linspace(amplitude[i, 0], amplitude[i, 1], Npix[i]))
    print("new azt in ({}, {}), new elt in ({}, {})".format(np.min(azel_arr[0]), np.max(azel_arr[0]), np.min(azel_arr[1]), np.max(azel_arr[1])))
    print(np.min(newazt), np.max(newazt), np.min(newelt), np.max(newelt))
    grid_interp = RegularGridInterpolator( (azel_arr[0], azel_arr[1]), img, method='nearest' ) 
    img_tod = grid_interp((newazt, newelt))
    return img_tod

def map_to_TOD(hp_map, nside):
    # To create fake TOD quickly from an input HEALPix map (not finished)
    theta, phi = hp.pix2ang(nside, np.arange(len(hp_map)), lonlat=True) # longitute and latitude in degrees, need to convert to azimuth elevation
    return None

def fitgauss_img(mapxy, x, y, xs, guess=None, doplot=False, distok=3, mytit='', nsig=1, mini=None, maxi=None, ms=10, renorm=False, mynum=33, axs=None, verbose=False, reso=None):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    
    ### Displays the image as an array
    mm, ss = ft.meancut(mapxy, 3)
    if mini is None:
        mini = mm-nsig*ss
    if maxi is None:
        maxi = np.max(mapxy)

    ### Guess where the maximum is and the other parameters with a matched filter
    if guess is None:
        Nx = len(mapxy)
        Ny = len(mapxy[0])
        lobe_pos = (Nx//2, Ny//2)
        Kx, Ky, K = get_K(Nx, Ny)
        ft_phase = get_ft_phase(lobe_pos, Nx, Ny)
        cos_win = cos_window(Nx, Ny, lx=20, ly=20)
        deltaK = 1
        Kbin = get_Kbin(deltaK, K)
        nKbin = len(Kbin) - 1  # nb of bins
        Kcent = (Kbin[:-1] + Kbin[1:])/2
        size_pix = reso/60 # degree
        reso_instr = 0.92
        ft_shape = fft2(gauss2D(Nx, Ny, x0=lobe_pos[0], y0=lobe_pos[1], reso=[reso_instr/size_pix], normal=True))
        
        filtmapsn = get_filtmapsn(mapxy * cos_win, nKbin, K, Kbin, Kcent, ft_shape, ft_phase)
        maxii = filtmapsn == np.nanmax(filtmapsn)
        maxx = np.mean(xx[maxii])
        maxy = np.mean(yy[maxii])
        guess = np.array([1e4, maxx, maxy, reso_instr])
        if verbose:
            print(guess)
    else:
        maxx = guess[1]
        maxy = guess[2]
        
    ### Do the fit putting the UNSEEN to a very low weight
    errpix = xx*0+ss
    errpix[mapxy==0] *= 1e5
    g2d = gauss2dfit(xx, yy)
    # g2d = filtgauss2dfit(xx, yy)
    data = fit.Data(np.ravel(xx), np.ravel(mapxy), np.ravel(errpix), g2d)
    m, ch2, ndf = data.fit_minuit(guess, limits=[[0, 1e3, 1e8], [1, maxx - distok, maxx + distok], [2, maxy - distok, maxy + distok], [3, 0.6/conv_reso_fwhm, 1.2/conv_reso_fwhm]], renorm=renorm)

    ### Image of the fitted Gaussian
    fitted = np.reshape(g2d(x, m.values), (xs, xs))

    if doplot:
        origin = "upper" #"lower" swaps the y-axis and the guess doesn't match 
        if axs is None:
            fig, axs = plt.subplots(1, 4, width_ratios=(1, 1, 1, 0.05), figsize=(16, 5))
            axs[1].imshow(fitted, origin=origin, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=mini, vmax=maxi)
            im = axs[2].imshow(mapxy - fitted, origin=origin, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], vmin=mini, vmax=maxi)
            axs[0].set_ylabel('Degrees')
            for i in range(3):
                axs[i].set_xlabel('Degrees')
            axs[2].set_title('Residuals')
    
    if doplot:
        axs = pmp.plot_fit_img(mapxy, axs, x, y, xguess=guess[1], yguess=guess[2], xfit=m.values[1], yfit=m.values[2], vmin=mini, vmax=maxi, ms=ms, origin=origin)
        return m, fitted, axs
    return m, fitted
    
    

def fit_one_tes(mymap, xs, reso, rot=np.array([0., 0., 0.]), doplot=False, verbose=False, guess=None, distok=3, mytit='', return_images=False, ms=10, renorm=False, xycreid_corr=None, axs=None):
    ### get the gnomview back into a np.array in order to fit it
    mm = mymap.copy()
    badpix = mm == hp.UNSEEN
    mm[badpix] = 0          ### Set bad pixels to zero before returning the np.array()
    mapxy = hp.gnomview(mm, reso=reso, rot=rot, return_projected_map=True, xsize=xs, no_plot=True).data

    ### np.array coordinates
    # Doesn't work with the fit plot but is ok with final gnomview plot of the Moon map corrected
    # But in order to stack the maps I now have to use (azt, -elt) position fitted here (why??)
    x = -(np.arange(xs) - (xs - 1)/2)*reso/60
    y = x.copy()
    x += rot[0]
    y -= rot[1]

    # Works on fit plot but then azt and elt are with the wrong sign on the final gnomview plot. Weird!!
    # x = (np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x -= rot[0]
    # y += rot[1]

    # Other tests
    # x = (np.arange(xs) - (xs - 1)/2)*reso/60
    # y = x.copy()
    # x -= rot[0]
    # y += rot[1]


    # print(np.min(y), np.max(y))
    # sys.exit()

    if xycreid_corr is not None:
        try:
            guess = np.array([1e4, xycreid_corr[0], xycreid_corr[1], 0.92])
            if verbose:
                print(guess)
        except:
            guess = None
            if verbose:
                print("TES has no position on sky")
                print(guess)
        
        
    if doplot:
        m, fitted, fig_axs = fitgauss_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, axs=axs, verbose=verbose, reso=reso)
        if verbose:
            print(m.values)
    else:
        m, fitted = fitgauss_img(mapxy, x, y, xs, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm, verbose=verbose, reso=reso)
    # try:
    #     m, fitted = fitgauss_img(mapxy, x, y, guess=guess, doplot=doplot, distok=distok, mytit=mytit, ms=ms, renorm=renorm)
    # except:
    #     m = None
    #     fitted = None
    
    if return_images:
        return m, mapxy, fitted, [np.min(x), np.max(x), np.min(y), np.max(y)], fig_axs
    return m
    

def get_close(deltax, deltay, tolerance):
    return np.sqrt(deltax**2 + deltay**2) <= tolerance

def assign_TES(x, y, xc, yc, tolerance, doplot=True):
    # xc and yc have been corrected for the shift and rotation
    # We want to check if some TES have the wrong number assigned to them

    # We first get the TES that are correctly numbered
    OK_1 = get_close(x - xc, y - yc, tolerance)

    if doplot:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.plot(x, y, "ro", alpha=0.2)
        ax.plot(xc, yc, "ko", label="Creidhe rotated ({})".format(len(xc)))
        ax.plot(x[OK_1], y[OK_1], "go", label="Well assigned ({})".format(np.sum(OK_1)))
        plt.legend()
        plt.show()
        
    # We then iterate


# From fitting.py of JC
class Data:
    def __init__(self, x, y, cov, model, pnames=None):
        self.x = x
        self.y = y
        self.model = model
        self.cov = cov
        if np.prod(np.shape(x)) == np.prod(np.shape(cov)):
            self.diag = True
            self.errors = cov
        else:
            self.diag = False
            self.errors = 1./np.sqrt(np.diag(cov))
            self.invcov = np.linalg.inv(cov)
        self.fit = None
        self.fitinfo = None
        self.pnames = pnames
        
    def __call__(self):
        return 0

    def plot(self, nn=1000, color=None, mylabel=None, nostat=False):
        p=plt.errorbar(self.x, self.y, yerr=self.errors, fmt='o', color=color, alpha=1)
        if self.fit is not None:
            xx = np.linspace(np.min(self.x), np.max(self.x), nn)
            plt.plot(xx, self.model(xx, self.fit), color=p[0].get_color(), alpha=1, label=mylabel)
        if mylabel is None:
            if nostat == False:
                plt.legend(title="\n".join(self.fit_info))
        else:
            plt.legend()


    
    def fit_minuit(self, guess, fixpars = None, limits=None, scan=None, renorm=False, simplex=False, minimizer=LeastSquares):
        ok = np.isfinite(self.x) & (self.errors != 0)

        ### Prepare Minimizer
        if self.diag == True:
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)
        else:
            print('Non diagoal covariance not yet implemented: using only diagonal')
            myminimizer = minimizer(self.x[ok], self.y[ok], self.errors[ok], self.model)

        ### Instanciate the minuit object
        if simplex == False:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames)
        else:
            m = iminuit.Minuit(myminimizer, guess, name=self.pnames).simplex()
        
        ### Limits
        if limits is not None:
            mylimits = []
            for k in range(len(guess)):
                mylimits.append((None, None))
            for k in range(len(limits)):
                mylimits[limits[k][0]] = (limits[k][1], limits[k][2])
            m.limits = mylimits

        ### Fixed parameters
        if fixpars is not None:
            for k in range(len(guess)):
                m.fixed["x{}".format(k)]=False
            for k in range(len(fixpars)):
                m.fixed["x{}".format(fixpars[k])]=True

        ### If requested, perform a scan on the parameters
        if scan is not None:
            m.scan(ncall=scan)

        ### Call the minimization
        m.migrad()  

        ### accurately computes uncertainties
        m.hesse()   

        ch2 = m.fval
        ndf = len(self.x[ok]) - m.nfit
        self.fit = m.values

        self.fit_info = [
            f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {ch2:.1f} / {ndf}",
        ]
        for i in range(len(guess)):
            vi = m.values[i]
            ei = m.errors[i]
            self.fit_info.append(f"{m.parameters[i]} = ${vi:.3f} \\pm {ei:.3f}$")

        if renorm:
            m.errors *= 1./np.sqrt(ch2/ndf)

        return m, ch2, ndf
    

def get_K(Nx, Ny):
    '''
    Parameters
    ----------
    hd : map header

    Returns
    -------
    Kx : 2D numpy array (Nx,Ny)
        K values for x dimension for the map.
    Ky : 2D numpy array (Nx,Ny)
        K values for y dimension for the map.
    K : 2D numpy array (Nx,Ny)
        K values for the map.
    '''
    Kx, Ky = np.meshgrid(fftfreq(Nx,d=1/Nx),fftfreq(Nx,d=1/Ny),indexing='ij')
    K=np.sqrt(Kx**2+Ky**2)
    return Kx ,Ky, K

def get_Kbin(deltaK, K):
    Kmax = np.ceil(np.max(K))
    k = np.arange(3+deltaK/2,Kmax+deltaK-1,deltaK)
    Kbin = np.concatenate(([0,1.5],k[:-2],[Kmax]))  # same def as JB bins (the middle of the bins are JB kp) except for the last bin
    return Kbin

def get_ft_phase(lobe_pos, Nx, Ny):  # problème si pas de round ! pourquoi ? parce que image décalée d'un nombre non entier de pixels/modes ?
    '''
    Parameters
    ----------
    lobe_pos : int tuple (2)
        x position and y position of the lobe.
    hd : map header

    Returns
    -------
    ft_phase : double
        corrective ft_phase of beam.
    '''
    px = lobe_pos[0]
    py = lobe_pos[1]
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    ft_phase = np.exp(-2*np.pi*1j*x*round(-px)/Nx)*np.exp(-2*np.pi*1j*y*round(-py)/Ny)
    return ft_phase


def get_filtmapsn(mapj, nKbin, K, Kbin, Kcent, ft_beam_map, ft_phase):
    ftmapj = fft2(mapj)
    result = np.zeros((nKbin))
    modu2 = ftmapj*np.conj(ftmapj)
    for i in range(nKbin):
        iKbin = np.logical_and(K>=Kbin[i], K<Kbin[i + 1])
        if len(K[iKbin])>0:
            result[i] = np.abs(np.mean(modu2[iKbin]))
        else:
            print("Kbin [{}, {}] is empty.".format(Kbin[i], Kbin[i + 1]))
    gp = np.interp(K, Kcent, result)   # Pk bins interpolated
    
    ftfilt = np.conj(ft_beam_map)/gp
    normfilt = np.sum(np.abs(ft_beam_map)**2/gp)
    filtmapsn = np.real(ifft2(ftfilt*ftmapj*ft_phase)/np.sqrt(normfilt))  # M convol T / sigma
    return filtmapsn

def cos_window(Nx, Ny, lx=None,ly=None):
    """
    Jean-Baptiste appelle le code avec :
    lx = Nx*0.05/2
    ly = Ny*0.05/2  # donc 2,5% de l'image de chaque côté
    """
    if lx==None:
        lx=Nx*0.05/2
    if ly==None:
        ly=Ny*0.05/2
    result = np.ones((Nx,Ny))
    X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny),indexing='ij')
    whx = X <= lx
    result[whx]=1/2.*(1-np.cos(np.pi/lx*X[whx]))
    whx = X >= Nx-1-lx
    result[whx]=1/2.*(1-np.cos(np.pi/lx*(Nx-1-X[whx])))
    why = Y <= ly
    result[why]=result[why]*1/2.*(1-np.cos(np.pi/ly*Y[why]))  # pour faire les coins aussi
    why = Y >= Ny-1-ly
    result[why]=result[why]*1/2.*(1-np.cos(np.pi/ly*(Ny-1-Y[why])))  # pour faire les coins aussi
    return result


def read_data(datadir, remove_t0=True):
    """
    Reads QUBIC raw data: time and TOD, as well azimuth, elevation and 
    their corresponding time

    Parameters
    ----------
    datadir : string
        Full path of the directory where the raw data is stored
        ex/ '/Volumes/QubicData/Calibration/2022-07-14/
    remove_t0 : bool
        subtracts the time of the first sample to the time vector, by default True.

    Returns
    -------
    tt : time for TOD
    tod : the TODs for all detectors
    thk : time for housekeeping data
    az : azimuth of the mount
    el : elevation of the mount
    """
    
    a = qubicfp()
    a.read_qubicstudio_dataset(datadir)
    tt, alltod = a.tod()
    az = a.azimuth()
    el = a.elevation()
    thk = a.timeaxis(datatype='hk')
    tinit = tt[0]
    if remove_t0:
        ### We remove tt[0]
        tinit = tt[0]
        tt -= tinit
        thk -= tinit
    del(a)
    return tt, alltod, thk, az, el, tinit

def get_azel_moon(ObsSite, tt, tinit, doplot=True):
    MySite = EarthLocation(lat=ObsSite['lat'], lon=ObsSite['lon'], height=ObsSite['height'])
    # utcoffset = ObsSite['UTC_Offset']

    dt0 = datetime.utcfromtimestamp(int((tt + tinit)[0]))
    print(dt0)

    nbtime = 100
    tt_hours_loc = tt/3600
    delta_time = np.linspace(np.min(tt_hours_loc), np.max(tt_hours_loc), nbtime)*u.hour

    alltimes = Time(dt0) + delta_time

    ### Local coordinates
    frame_Site = AltAz(obstime=alltimes, location=MySite)

    ### Source
    moon_Site = get_moon(alltimes)
    moonaltazs_Site = moon_Site.transform_to(frame_Site)  

    myazmoon = moonaltazs_Site.az.value
    myelmoon = moonaltazs_Site.alt.value

    azmoon = np.interp(tt_hours_loc, delta_time/u.hour, myazmoon)
    elmoon = np.interp(tt_hours_loc, delta_time/u.hour, myelmoon)
    if doplot:
        plt.figure()
        plt.plot(myazmoon, myelmoon, 'ro')
        plt.plot(azmoon, elmoon)
        plt.show()
    return azmoon, elmoon


def make_coadded_maps(datadir, ObsSite, allTESNum, start_tt=10000, data=None, speedmin=0.05, 
                      doplot=True, nside=256, az_qubic=0, parallel=False, check_back_forth=False):
    ### First read the data from disk if needed
    if data is None:
        print('Reading data from disk: '+datadir)
        tt, alltod, thk, az, el, tinit = read_data(datadir, remove_t0=False)
        az += az_qubic
        tt_save = np.copy(tt)
        alltod_save = np.copy(alltod)
        thk_save = np.copy(thk)
        az_save = np.copy(az)
        el_save = np.copy(el)
        data = [tt_save, alltod_save, thk_save, az_save, el_save, tinit]
        print("tinit = {}".format(tinit))
    else:
        print('Using data already stored in memory - not read from disk')
        tt_save, alltod_save, thk_save, az_save, el_save, tinit = data
        tt = np.copy(tt_save)
        alltod = np.copy(alltod_save)
        thk = np.copy(thk_save)
        az = np.copy(az_save)
        el = np.copy(el_save)
        print(np.shape(tt))
        print(np.shape(alltod))
        print("tinit = {}".format(tinit))
    
    # Remove the first start_tt points (out of 1998848)
    tinit = tt[start_tt]
    print("tinit = {}".format(tinit))
    alltod = alltod[:, start_tt:]
    tt = tt[start_tt:]
    # I need to put tt[0] to zero, but be careful of real time
    # Also, I would have to adjust the mount time?
    tt -= tinit + 0.21 # delta_t seen in plotting back and forth images
    thk -= tinit
    print(np.shape(tt))
    print(np.shape(alltod))
    print("tinit = {}".format(tinit))

    ### Azimuth and Elevation of the Moon at the same timestamps from the observing site
    azmoon, elmoon = get_azel_moon(ObsSite, tt, tinit, doplot=False)
    
    ### Identify scan types and numbers
    scantype_hk, azt, elt, scantype, vmean = identify_scans(thk, az, el, 
                                                                tt=tt, doplot=False, 
                                                                plotrange=[0, 2000], 
                                                                thr_speedmin=speedmin)

    ### Loop over TES to do the maps
    print('\nLooping coaddition mapmaking over selected TES')
    print('nside = ',nside)
    start_time = time.perf_counter()
    if parallel is False:
        print('Using sequential loop')
        allmaps = np.zeros((len(allTESNum), 12*nside**2))
        for i in range(len(allTESNum)):
            TESNum = allTESNum[i]
            print('TES# {}'.format(TESNum), end=" ")
            tod = alltod[TESNum-1,:]
            
            allmaps[i,:], mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, azmoon, elmoon,
                                                             nside=nside, 
                                                             doplot=doplot, check_back_forth=check_back_forth)
            print('OK', flush=True)
    else:
        print('using a parallel loop : no output will be given while processing... be patient...')
        ### Note that this code has been generated using ChatGPT
        def process_TES(i, TESNum, allmaps, alltod, tt, azt, elt, scantype, azmoon, elmoon, nside, doplot):
            # Create a lock for each process to ensure safe access to shared memory
            lock = Lock()
            
            tod = alltod[TESNum - 1, :]

            map_result, mapscounts = make_coadded_maps_TES(tt, tod, azt, elt, scantype, azmoon, elmoon,
                                                           nside=nside, doplot=doplot)        
            # Use lock to ensure safe access to shared memory inside the inner function
            with lock:
                # Directly assign the result to the correct index in allmaps
                # allmaps is a list of numpy arrays, so we can use allmaps[i] directly
                allmaps[i] = map_result
        
        def parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, scantype, azmoon, elmoon, nside, doplot):
            # Use Manager to create a shared list that will be modified by parallel processes
            with Manager() as manager:
                # Create a list of NumPy arrays initialized to zeros
                allmaps = manager.list([np.zeros(12 * nside ** 2) for _ in range(len(allTESNum))])
        
                # Run the parallel processing with the correct arguments
                Parallel(n_jobs=-1)(delayed(process_TES)(i, allTESNum[i], allmaps, alltod, tt, azt, elt, scantype, azmoon, elmoon, nside, doplot)
                                    for i in range(len(allTESNum)))
        
                # Convert the manager list back to a NumPy array (this ensures allmaps is a numpy array of arrays)
                allmaps_np = np.array([np.array(allmaps[i]) for i in range(len(allTESNum))])
        
            return allmaps_np

        allmaps = parallel_coadded_maps(allTESNum, alltod, tt, azt, elt, 
                                        scantype, azmoon, elmoon, nside, doplot)
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds => average of {(elapsed_time/len(allTESNum)):.4f} per TES")    
        

    # Get central Az and El from pointing
    newazt, newelt = get_new_azel(azt, elt, azmoon, elmoon)
    center = [np.mean(newazt), np.mean(newelt)]
    return allmaps, data, center

# from QdataHandling
def identify_scans(thk, az, el, tt=None, median_size=101, thr_speedmin=0.1, doplot=False, plotrange=[0,1000]):
    """
    This function identifies and assign numbers the various regions of a back-and-forth scanning using the housepkeeping time, az, el
        - a numbering for each back & forth scan
        - a region to remove at the end of each scan (bad data due to FLL reset, slowingg down of the moiunt, possibly HWP rotation
        - is the scan back or forth ?
    It optionnaly iinterpolate this information to the TOD sampling iif provided.
    Parameters
    ----------
    input
    thk : np.array()
            time samples (seconds) for az and el at the housekeeeping sampling rate
    az : np.array()
            azimuth in degrees at the housekeeping sampling rate
    el : np.array()
            elevation in degrees at the housekeeping sampling rate
    tt : Optional : np.array()
            None buy default, if not None:
            time samples (seconds) at the TOD sampling rate
            Then. the output will also containe az,el and scantype interpolated at TOD sampling rate
    thr_speedmin : Optional : float
            Threshold for angular velocity to be considered as slow
    doplot : [Optional] : Boolean
            If True displays some useeful plot
    output :
    scantype_hk: np.array(int)
            type of scan for each sample at housekeeping sampling rate:
            * 0 = speed to slow - Bad data
            * n = scanning towards positive azimuth
            * -n = scanning towards negative azimuth
            where n is the scan number (starting at 1)
    azt : [optional] np.array()
            azimuth (degrees) at the TOD sampling rate
    elt : [optional] np.array()
            elevation (degrees) at the TOD sampling rate
    scantype : [optiona] np.array()
            same as scantype_hk, but interpolated at TOD sampling rate
    """

    def get_az_vel(time, azimuth, order=2): # get the angular azimuth velocity
        az_vel = np.zeros(len(time))
        az_vel[:order] = (azimuth[1:order + 1] - azimuth[:order])/(time[1:order + 1] - time[:order])
        az_vel[-order:] = (azimuth[-order:] - azimuth[-order - 1:-1])/(time[-order:] - time[-order - 1:-1])
        dt_ = time[2*order:] - time[:-2*order]
        az_vel[order:-order] = (az[2*order:] - az[:-2*order])/dt_
        return az_vel
    medaz_dt_ = get_az_vel(thk, az, order=50) # high order necessary to remove glitches
    medaz_dt = medfilt(medaz_dt_, median_size)
    if doplot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(thk, medaz_dt_)
        plt.plot(thk, medaz_dt)
        plt.xlim(plotrange[0],plotrange[1])

        plt.subplot(1, 2, 2)
        plt.plot(thk, az)
        plt.xlim(plotrange[0],plotrange[1])
        plt.show()
    ### Identify regions of change
    # Low velocity -> Bad
    c0 = np.abs(medaz_dt) < thr_speedmin
    # Positive velicity => Good
    cpos = (~c0) * (medaz_dt >= 0)
    # Negative velocity => Good
    cneg = (~c0) * (medaz_dt < 0)

    ### Scan identification at HK sampling
    scantype_hk = np.zeros(len(thk), dtype='int')-10
    scantype_hk[c0] = 0
    scantype_hk[cpos] = 1
    scantype_hk[cneg] = -1
    # check that we have them all
    count_them = np.sum(scantype_hk==0) + np.sum(scantype_hk==-1) + np.sum(scantype_hk==1)
    if count_them != len(scantype_hk):
        ValueError('Identify_scans: Bad Scan counting at HK sampling level - Error')

    ### Now give a number to each back and forth scan
    num = 0
    previous = 0
    for i in range(len(scantype_hk)):
        if scantype_hk[i] <= 0:
            previous = 0
        elif previous == 0:
            # we have a change
            num += 1
            previous = 1
        scantype_hk[i] *= num

    dead_time = np.sum(c0) / len(thk)

    if doplot:
        ### Some plotting (a lot), moved to other file not to take too much space here
        pmp.plots_identify_scans(thk, plotrange, az, medaz_dt, c0, cpos, cneg, dead_time, el, scantype_hk)

        

    vmean = 0.5 * (np.abs(np.mean(medaz_dt[cpos])) +  np.abs(np.mean(medaz_dt[cneg])))
    if tt is not None:
        ### We propagate these at TOD sampling rate  (this is an "step interpolation": we do not want intermediatee values")
        scantype = interp1d(thk, scantype_hk, kind='previous', fill_value='extrapolate')(tt)
        scantype = scantype.astype(int)
        count_them = np.sum(scantype==0) + np.sum(scantype<=-1) + np.sum(scantype>=1)
        if count_them != len(scantype):
            ValueError('Bad Scan counting at data sampling level - Error')
        ### Interpolate azimuth and elevation to TOD sampling
        azt = np.interp(tt, thk, az)
        elt = np.interp(tt, thk, el)
        ### Return evereything
        return scantype_hk, azt, elt, scantype, vmean
    else:
        ### Return scantype at HK sampling only
        return scantype_hk
    
### DBSCAN
from sklearn.cluster import DBSCAN
def run_DBSCAN(params, eps=0.5, min_samples=10):
    # clustering = DBSCAN(eps=1.3, min_samples=10).fit(params)
    # clustering = DBSCAN(eps=0.5, min_samples=10).fit(params)
    # clustering = DBSCAN(eps=0.25, min_samples=10).fit(params)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(params)
    labels = clustering.labels_
    return labels


# transformer en fonction pour réutiliser avec les fits sur les valeurs corrigées de Créidhe
def get_DBscan_res(x_fit, y_fit, x_theo, y_theo, x_err, y_err, FWHM, errFWHM, visibly_ok_arr, doplot, eps=0.5, min_samples=10):

    delta_az = x_fit - x_theo
    err_delta_az = x_err
    delta_el = y_fit - y_theo
    err_delta_el = y_err

    params_dbscan = np.array([delta_az, delta_el, err_delta_az, err_delta_el, FWHM, errFWHM]).T
    rng_nan = np.random.default_rng(seed=12345)
    params_dbscan[np.isnan(params_dbscan)] = rng_nan.uniform(low=1, high=2, size=(len(params_dbscan[np.isnan(params_dbscan)]),)) * 1e8

    labels = run_DBSCAN(params_dbscan, eps=eps, min_samples=min_samples)
    DB_ok = labels == 0
    if doplot:
        plt.figure()
        plt.subplot().set_aspect(1)
        plt.plot(delta_az[visibly_ok_arr], delta_el[visibly_ok_arr], 'ko', label='all visibly ok ({})'.format(len(delta_az[visibly_ok_arr])))
        plt.plot(delta_az[DB_ok], delta_el[DB_ok], 'ro', label='DBSCAN selected ({})'.format(len(delta_az[DB_ok])))
        plt.xlabel('$\Delta_{az}^{Moon} - Offset_{Creidhe}$')
        plt.ylabel('$\Delta_{el}^{Moon} - Offset_{Creidhe}$')
        plt.legend()
        plt.show()

    return DB_ok


### Function to rotate a set of points around a given center
def rotate_translate_scale_2d(xin, theta, center, scale):
    rotmat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return scale * np.dot(rotmat, (xin-center).T).T

def rot_trans_scale_pts(x, pars):
    pts = np.reshape(x, (len(x)//2, 2))
    return np.ravel(rotate_translate_scale_2d(pts, np.radians(pars[0]), np.array([pars[1],pars[2]]), pars[3]))