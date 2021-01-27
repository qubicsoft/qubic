#Importing system packages
import os
import sys

#Importing science packages
import numpy as np
import healpy as hp
from matplotlib.pyplot import *
import scipy.ndimage.filters as f

#Importing qubic packages
from qubic import *
from qubicpack.pixel_translation import make_id_focalplane, plot_id_focalplane, tes2pix, tes2index
import qubic.sb_fitting as sbfit

def create_hall_pointing(d, az, el, angspeed_psi, maxpsi,
                 date_obs = None, latitude = None, longitude = None, 
                 fix_azimuth = None, random_hwp = True):
    
    '''This method will reproduce the pointing that is used in the hall to take the data. 
    Will start from bottom left and will go up at fixed elevation.
    '''

    nsamples = len(az)*len(el)
    pp = QubicSampling(nsamples,date_obs = d['date_obs'], period = 0.1, 
        latitude = latitude, longitude = longitude)
    
    #Comented because we do not back and forth in simulations.. 
    #mult_el = []
    #for eachEl in el:
    #    mult_el.append(np.tile(eachEl, 2*len(az)))
    # Azimuth go and back and same elevation. 
    #az_back = az[::-1]
    #az = list(az)
    #az.extend(az_back)
    #mult_az = np.tile(az, len(el))
    #print(i,np.asarray(mult_el).ravel().shape)
    #pp.elevation = np.asarray(mult_el).ravel()
    #pp.azimuth = np.asarray(mult_az).ravel()
    
    mult_el = []
    for eachEl in el:
        mult_el.extend(np.tile(eachEl, len(az)))
    mult_az = []
    mult_az.append(np.tile(az, len(el)))
    pp.elevation = np.asarray(mult_el)
    pp.azimuth = np.asarray(mult_az[0])
    
    ### scan psi as well,
    pitch = pp.time * angspeed_psi
    pitch = pitch % (4 * maxpsi)
    mask = pitch > (2 * maxpsi)
    pitch[mask] = -pitch[mask] + 4 * maxpsi
    pitch -= maxpsi
    
    pp.pitch = pitch
    
    if random_hwp:
        pp.angle_hwp = np.random.random_integers(0, 7, nsamples) * 11.25
        
    if d['fix_azimuth']['apply']:
        pp.fix_az=True
        if d['fix_azimuth']['fix_hwp']:
            pp.angle_hwp=pp.pitch*0+ 11.25
        if d['fix_azimuth']['fix_pitch']:
            pp.pitch= 0
    else:
        pp.fix_az=False

    return pp

def npp(ipix):
    
    #neighboring pixel of the peak
    # map in nest
    return np.array(hp.get_all_neighbours(256, ipix,nest=True))

def selectcenter(hpmap, center, delta = 3, nside = 256, nest = True,
                threshold = 3, displaycenters = False, plot = False):
    
    #return the pixel of the central peak
    npix = 12 * nside ** 2
    centerarr = [center,
                center - delta * np.array([1,0]),
                #center - 2 * delta * np.array([1,0]),
                center + delta * np.array([1,0]),
                #center + 2 * delta * np.array([1,0]),
                center - delta * np.array([0,1]),
                #center - 2 * delta * np.array([0,1]),
                center + delta * np.array([0,1])]
                #center + 2 * delta * np.array([0,1])]

    fullvec = hp.pix2vec(nside, range(0,npix), nest = nest)
    relmaxpx = np.zeros((len(centerarr),))
    px = np.zeros((len(centerarr),), dtype = int)

    for j,icenter in enumerate(centerarr):
        ivec = hp.ang2vec(np.deg2rad(icenter[0]), np.deg2rad(icenter[1]))
        imaskpx = np.rad2deg(np.arccos(np.dot(ivec,fullvec))) < threshold
        imaskidx = np.where(imaskpx == True)[0]
        relmaxpx[j] = np.max(hpmap[imaskpx])
        px[j] = imaskidx[np.argmax(hpmap[imaskpx])]

    indxmax = np.argmax(relmaxpx)
    pixmax, newcenter = px[indxmax], centerarr[indxmax]
    if plot:
        hp.gnomview(hpmap, reso = 12, rot = np.array([90,0]) - newcenter, nest = nest)
        if displaycenters:
            for each in centerarr:
                hp.projscatter(np.deg2rad(each), marker = '+', color = 'r')
        hp.projscatter(hp.pix2ang(256, pixmax, nest = nest), marker = '+', color = 'r')

    return pixmax, newcenter

def thph_qsoft(qinst, scene, soft_pix, 
               PiRot = True, ref_detector = False, index_ref = None):
    
    """
    Returns th,ph for each peaks for a given TES and instrument. 
    There is a pi rotation in z-axis to match with demodulated data

    Parameter:
    ============
    qinst: QubicInstrument method
    scene: QubicScene method
    soft_pix: index of TES in qubicsoft convention
    PiRot: boolean. Makes a rotation in z-axis to match software simulation with demod data. Default: True 

    Return:
    ============
    th_tes_all: theta coordinates of all peaks for given soft_pix TES
    ph_tes_all: phi coordinates of all peaks for given soft_pix TES
    index: 
    """

    sim_th, sim_ph, _ = qinst._peak_angles(scene, qinst.filter.nu, 
                                            qinst.detector[soft_pix].center, qinst.synthbeam, 
                                            getattr(qinst, 'horn', None), 
                                            getattr(qinst, 'primary_beam', None))
    
    # PI rotation
    if PiRot:
        newthph = np.zeros((sim_th.shape[1],2))
        th_tes_all = np.zeros((1, sim_th.shape[1]) ) 
        ph_tes_all = np.zeros((1, sim_th.shape[1]) )

        for i in range(sim_th.shape[1]):
            newuv = np.dot(sbfit.rotmatZ(np.pi), sbfit.thph2uv(sim_th[0, i], sim_ph[0, i]) )
            newthph = sbfit.uv2thph(newuv)
            th_tes_all[0, i] = newthph[0]
            ph_tes_all[0, i] = newthph[1]
    else: 
        th_tes_all = sim_th
        ph_tes_all = sim_ph
    return th_tes_all, ph_tes_all

def flat2hp_map(maps, az, el, nside = 256):
    
    auxth, auxph = np.pi/2 - np.deg2rad(el), np.deg2rad(az)
    auxpix = hp.ang2pix(nside, auxth, auxph)
    auxmap = np.zeros((12 * nside ** 2,))
    auxmap[auxpix] = maps
    
    return auxmap


class SbHealpyModel:

    """
    This class aims to fit the location and amplitude of the peaks in healpix projections for TODs 
    obtained in qubic simulations or from the data. Uses a pointing strategy adapted from
    qubicsoft using scan azimuth and elevation coordinates.     
    """
    
    def __init__(self):

def fit_hpmap(PIXNum, q, s, dirfiles, verbose = False, #centerini, 
              nside = 256, nest = True, filterbeam = 3, PiRot = True, simulation = False, maps = None,
             threshold = 3, threshold0 = 4, plotcenter = False, plot = False,
             plotnine = False, plotneig = False, refilter = False ):
    
    """
    This method fits the peak location in Healpix projection for a given TES. 

    Parameters: 
    =================
    PIXNum: 
        Pixel in JCh (data-files convention used by JCh) convention (1-256). 
    q: 
        QubicInstrument class
    s:
        QubicScene class
    simulation: 
        If False, uses data files to fit. If False, it fits simulated maps
    dirfiles: 
        Path pointing to Parent directory of a scan (e.g. LOCAL_PATH_TO_DATA + '150GHz-2019-04-06/')
    centerini: [deprecated]
        Initial position of the central peak  

    Returns:
    =================
    Healpy map in NEST projection, thphpeaksnew, absmaxpx

    """
    # Convert from JCh convention to Qubicinstrument one.
    global TESNum

    TESNum, asic = (PIXNum, 1) if (PIXNum < 128) else (PIXNum - 128, 2)
    PIXq = tes2pix(TESNum, asic) - 1
    
    if verbose:
        print("You are running fitting in healpix maps.")
        print("========================================")
        print("TES number {} asic number {}".format(TESNum, asic))
        print("Index number: qpack {} qsoft {} ".format(\
                                tes2index(TESNum, asic), q.detector[PIXq].index[0] ))
        print("qubicsoft number: {}".format(PIXq))


    # Get healpix projection of the TOD for a given PIXNum
    if not simulation:
        # Compute theoreticall th,ph from qsoft
        th_tes_all, ph_tes_all = thph_qsoft(q, s, PIXq, PiRot = PiRot)
        
        # Using only central peak
        th_tes, ph_tes = th_tes_all[0,0], ph_tes_all[0,0]

        # theta, phi to vector of central peak for TES (p0 for fitting function)
        vec_tes = np.array([np.sin(th_tes) * np.cos(ph_tes),
                        np.sin(th_tes) * np.sin(ph_tes),
                        np.cos(th_tes)])
        
        fullvec = hp.pix2vec(nside, range(0, 12 * nside ** 2), nest = nest)

        # Carry synth beam from polar cap to the equatorial one
        centerini = [hp.vec2ang(np.dot(sbfit.rotmatY(np.pi/2), vec_tes))[0][0],
                    hp.vec2ang(np.dot(sbfit.rotmatY(np.pi/2), vec_tes))[1][0]]

        npix = 12 * nside ** 2

        hpmap = sbfit.get_hpmap(PIXNum, dirfiles)

    elif simulation:
        # Compute theoreticall th,ph from qsoft
        th_tes_all, ph_tes_all = thph_qsoft(q, s, PIXq, PiRot = PiRot)
                
        # Using only central peak
        th_tes, ph_tes = th_tes_all[0,0], ph_tes_all[0,0]

        # theta, phi to vector of central peak for TES (p0 for fitting function)
        vec_tes = np.array([np.sin(th_tes) * np.cos(ph_tes),
                        np.sin(th_tes) * np.sin(ph_tes),
                        np.cos(th_tes)])
        
        fullvec = hp.pix2vec(nside, range(0, 12 * nside ** 2), nest = nest)

        # Carry synth beam from polar cap to the equatorial one
        centerini = [hp.vec2ang(np.dot(sbfit.rotmatY(np.deg2rad(90 - 50.712)), vec_tes))[0][0],
                    hp.vec2ang(np.dot(sbfit.rotmatY(np.deg2rad(90 - 50.712)), vec_tes))[1][0]]

        npix = 12 * nside ** 2

        #hpmap = q.get_synthbeam(s)[PIXq, newpix]
        hpmap = maps

    if nest:
        hpnest = hp.reorder(hpmap, r2n = nest)
    else:
        hpnest = hpmap
    hpnest_filt = f.gaussian_filter(hpnest, filterbeam)

    centerini = [np.rad2deg(centerini[0]), np.rad2deg(centerini[1])]
    px, center = selectcenter(hpnest_filt, centerini, plot = plot, nside = nside)
    thetaphi = hp.pix2ang(nside, px, nest = nest)
    
    if plotcenter:
        vli = [px,]
        xlo = px-100
        xhi = px+100
        
        fig, ax = subplots(nrows = 1, ncols = 2, figsize = (12,4))
        ax[0].set_xlim(xlo,xhi)
        ax[0].plot(hpnest, 'bo--', label = 'raw')
        ax[0].plot(hpnest_filt, 'bo--', alpha = 0.4, label = 'filtered')
        ax[0].legend()
        for i in vli:
            ax[0].axvline(i, c = 'k', alpha = 0.4, ls = '--')
            neig = npp(i)
        hp.gnomview(hpnest, reso = 12, rot = center, nest = True)
        hp.projscatter(np.deg2rad(centerini[0]), np.deg2rad(centerini[1]), marker = '+', color = 'r')

    pxvec = hp.pix2vec(nside, px, nest = nest)
    fullvec = hp.pix2vec(nside, range(0,npix), nest = nest)
    fullpx = np.linspace(0, npix, npix, dtype = int)

    aberr = np.deg2rad(np.array([0,05.]))
    delta = np.deg2rad(13.)
    #Old peaksordering
    #thphpeaks = [thetaphi,
    #             thetaphi-delta*np.array([1,0]),                 
    #            thetaphi+delta*np.array([1,0]), 
    #            thetaphi-delta*np.array([0,1]+aberr),
    #            thetaphi+delta*np.array([0,1]+aberr),
    #            thetaphi-delta*0.5*np.array([1,1]),
    #            thetaphi+delta*0.5*np.array([1,1]),
    #            thetaphi-delta*0.5*np.array([-1,1]),
    #            thetaphi+delta*0.5*np.array([-1,1]),
    #             ]
    
    #peaks ordering according JCh and instrument module
    thphpeaks = [thetaphi + delta * np.array([1,0]),
                 thetaphi - delta * 0.5 * np.array([-1,1]),
                 thetaphi - delta * np.array([0,1] + aberr), 
                 thetaphi + delta * 0.5 * np.array([1,1]),
                 thetaphi,
                 thetaphi - delta * 0.5 * np.array([1,1]),
                 thetaphi + delta * np.array([0,1] + aberr),
                 thetaphi + delta * 0.5 * np.array([-1,1]),
                 thetaphi - delta * np.array([1,0])
                 ]

    fullvec = hp.pix2vec(nside, range(0,npix), nest = nest)
    realmaxpx = np.zeros((9,), dtype = int)
    absmaxpx = np.zeros((9,), dtype = int)

    if plotnine: fig, ax = subplots(nrows = 9, ncols = 1, figsize = (8,8),)
    thphpeaksnew = np.zeros((9,2))
    
    for j, ithphpx in enumerate(thphpeaks):
        c = 'b'
        if j == 4: 
            threshold = threshold0
        else:
            threshold = threshold
            
        ivec = hp.ang2vec(ithphpx[0], ithphpx[1], )
        ifullpx = np.linspace(0, npix, npix, dtype = int)
        maskipx = np.rad2deg(np.arccos(np.dot(ivec,fullvec))) < threshold
        
        if refilter:
            mean, std = np.mean(hpnest_filt[maskipx]),np.std(hpnest_filt[maskipx])
            maskipx2 = hpnest_filt[maskipx] < mean+3*std
            maskipx3 = hpnest_filt[maskipx] > mean-3*std
            maskipx[maskipx] = maskipx2 * maskipx3
        maskidx = np.where(maskipx == True)[0]
        #useless max (just to plot in 1d not healpix)
        realmaxpx[j] = np.where(hpnest_filt[maskipx] == np.max(hpnest_filt[maskipx]))[0][0]
        #usefull max (healpix)
        absmaxpx[j] = maskidx[realmaxpx[j]]
        thphpeaksnew[j] = hp.pix2ang(nside,absmaxpx[j],nest = nest)

        if plotnine:
            if j == 3: c = 'r'
            ax[j].axvline(realmaxpx[j], c = 'k', alpha = 0.4, ls = '--')
            ax[j].plot(hpnest_filt[maskipx], 'o--', color = c, alpha = 0.4, label = 'filtered')
            ax[j].legend()        

    return hpnest, thphpeaksnew, absmaxpx


def _argsort_reverse(a, axis=-1):
    i = list(np.ogrid[[slice(x) for x in a.shape]])
    i[axis] = a.argsort(axis)[:, ::-1]
    return i

def _peak_angles_ref(scene, nu, position, synthbeam, horn, primary_beam,
                ref_detector = False, index_ref = None):
    
    """
    Compute the angles and intensity of the synthetic beam peaks which
    accounts for a specified energy fraction.
    
    ref_detector == True means you compute the indexes array in the old way (ordering the peaks 
    according vals values and can be used for ordering other detectors
    """

    theta, phi = qubic.QubicInstrument._peak_angles_kmax(
        synthbeam.kmax, horn.spacing, horn.angle, nu, position)
    val = np.array(primary_beam(theta, phi), dtype=float, copy=False)
    val[~np.isfinite(val)] = 0
    
    if ref_detector:
        index = _argsort_reverse(val)
    elif not ref_detector:
        #if index_ref is None:
        #    raise ValueError("You have to give an indexes array from reference detector")
        index = index_ref
    
    theta = theta[tuple(index)]
    phi = phi[tuple(index)]
    val = val[tuple(index)]
    cumval = np.cumsum(val, axis=-1)
    imaxs = np.argmax(cumval >= synthbeam.fraction * cumval[:, -1, None],
                      axis=-1) + 1
    imax = max(imaxs)

    # slice initial arrays to discard the non-significant peaks
    theta = theta[:, :imax]
    phi = phi[:, :imax]
    val = val[:, :imax]

    # remove additional per-detector non-significant peaks
    # and remove potential NaN in theta, phi
    for idet, imax_ in enumerate(imaxs):
        val[idet, imax_:] = 0
        theta[idet, imax_:] = np.pi / 2  # XXX 0 fails in polarization.f90.src (en2ephi and en2etheta_ephi)
        phi[idet, imax_:] = 0
    solid_angle = synthbeam.peak150.solid_angle * (150e9 / nu) ** 2
    val *= solid_angle / scene.solid_angle * len(horn)

    return theta, phi, val, index


def mask_unseen(hpmap, az, el, doconvert = False, nest = False):

    '''
    Mask unseen directions in the scan. Use as input the house keeping coordinates.
    Parameters: 
        hpmap. Healpix map (npix)
        az and el. Azimuth and Elevation coordinates read from azimuth.fits and elevation.fits
        doconvert. Consider the data as real azimuth and elevation 
                    (not implemented in that way in demodulation yet).
    Return: 
        hpmap[masked]
    '''
    if doconvert:
        hkcoords = np.meshgrid(az, el)
        radec = qubic.hor2equ(hkcoords[0].ravel(), hkcoords[1].ravel(), 0)
        phi = radec[0]
        theta = radec[1]
        #Rotation from Az,El housekiping to Az, El = 0,0
        newcoords = np.dot(sbfit.rotmatY(qubic.hor2equ(0,+50,0)[1]),  
                           hp.ang2vec(np.pi/2-np.deg2rad(theta), np.deg2rad(phi)).T).T
    else:
        hkcoords = np.meshgrid(az, el-50)
        phi = hkcoords[0].ravel()
        theta = hkcoords[1].ravel()
        newcoords = hp.ang2vec(np.pi/2-np.deg2rad(theta), np.deg2rad(phi))

    nside = hp.get_nside(hpmap)
    coordspix = hp.vec2pix(nside, newcoords[...,0], newcoords[...,1], newcoords[...,2], nest = nest)
    mask = np.zeros((12 * nside **2 ), dtype = bool)
    mask[coordspix] = 1
    hpmap[~mask] = hp.UNSEEN    
    #hp.mollview(hpmap, nest = True)
    show()
    
    return hpmap