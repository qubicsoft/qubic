#Importing system packages
import os
import sys
from warnings import warn

#Importing science packages
import numpy as np
import healpy as hp
from matplotlib.pyplot import *
import scipy.ndimage.filters as f
from astropy.time import Time, TimeDelta

#Importing qubic packages
from qubic import *
from qubicpack.pixel_translation import make_id_focalplane, plot_id_focalplane, tes2pix, tes2index
import qubic.sb_fitting as sbfit
from pyoperators import (
    Cartesian2SphericalOperator,
    Rotation3dOperator,
    Spherical2CartesianOperator)
from pysimulators import ( 
    CartesianEquatorial2HorizontalOperator,
    CartesianHorizontal2EquatorialOperator)

def generate_region(az, el):
    """
    Generates the squared region scanned in the hall from the azimuth&elevation coordinates
    saved in housekeeping data.
    
    Return a grid with coordinates
    """
    mult_el = []
    for eachEl in el:
        mult_el.extend(np.tile(eachEl, len(az)))
    mult_el = mult_el[::-1]
    mult_az = []
    mult_az.append(np.tile(az, len(el)))
    
    return mult_az, mult_el

def create_hall_pointing(d, az, el, hor_center, angspeed_psi = 0, maxpsi = 0, period = 0, fillfield = False,
                 date_obs = None, latitude = None, longitude = None, doplot = False,
                 fix_azimuth = None, random_hwp = True, verbose = False):
    
    '''
    Model of the pointing used in the hall. No back and forth. 
    
    Input coordinates are az, el. The function authomatically will convert (az, el) into (phi, theta) 
    defined as qubic.sampling.create_random_pointing to match with qubicsoft. 
    
    The coverage map center the region in hor_center coordinates. Take it into account for 
    plotting and projecting maps
    
    Parameters:
        d: QUBIC dictionary
        az, el: azimuth and elevation data from housekeeping data or fits file. 1-d array
        period: QubicSampling parameter. If equal to zero, it matches with transformation from az,el
        to ra, dec using qubic.hor2equ(az, el time = 0). Otherwise is not equal. Default: zero. 
        hor_center: center of the FOV
    Return: 
        QUBIC's pointing object
    '''

    if fillfield:
        az = np.arange(az[0], az[-1], hp.nside2resol(d['nside'], arcmin = True) / 60)
        el = np.arange(el[0], el[-1], hp.nside2resol(d['nside'], arcmin = True) / 60)
    
    nsamples = len(az)*len(el)
    
    mult_az, mult_el = generate_region(az,el)
    theta = np.array(mult_el) #- np.mean(el)
    phi = np.array(mult_az[0]) #- np.mean(az)
    

    # By defalut it computes HorizontalSampling in with SphericalSamplig
    pp = qubic.QubicSampling(nsamples, #azimuth = mult_az[0], elevation = mult_el[0],
                             date_obs = d['date_obs'], period = period, 
                            latitude = latitude, longitude = longitude)
    
    time = pp.date_obs + TimeDelta(pp.time, format='sec')
    print("time", np.shape(time))
    c2s = Cartesian2SphericalOperator('azimuth,elevation', degrees=True)
    h2e = CartesianHorizontal2EquatorialOperator(
        'NE', time, pp.latitude, pp.longitude)
    s2c = Spherical2CartesianOperator('elevation,azimuth', degrees=True)
    
    rotation = c2s(h2e(s2c))
    coords = rotation(np.asarray([theta.T, phi.T]).T)

    pp.elevation = mult_el
    pp.azimuth = mult_az[0]
    pp.equatorial[:,0] = coords[:,0]
    pp.equatorial[:,1] = coords[:,1]

    if doplot:
        fig, ax = subplots(nrows = 1, ncols = 2, figsize = (14,6))
        pixsH = hp.ang2pix(d['nside'], np.radians(90 - theta), np.radians(phi))
        mapaH = np.ones((12*d['nside']**2))
        mapaH[pixsH] = 100
        axes(ax[0])
        hp.gnomview(mapaH, title = "Horizontal coordinates", reso = 12,
            rot = [np.mean(phi),np.mean(theta)], hold = True)
        hp.graticule(verbose = False)
        pixsEq = hp.ang2pix(d['nside'], np.radians(90 - pp.equatorial[:,1]), np.radians(pp.equatorial[:,0]))
        mapaEq = np.ones((12*d['nside']**2))
        mapaEq[pixsEq] = 100
        axes(ax[1])
        hp.mollview(mapaEq, title = "Equatorial coordinates", hold = True)
        hp.graticule(verbose = False)
        
    azcen_fov, elcen_fov = hor_center[0], hor_center[1]
    if period < 1e-4:
        newcenter = qubic.hor2equ(azcen_fov, elcen_fov, 0)
    else:
        newcenter = qubic.hor2equ(azcen_fov, elcen_fov, pp.time[int(len(pp.time)/2)])

    warn("Update RA, DEC in dictionary")
    d['RA_center'], d['DEC_center'] = newcenter[0], newcenter[1]
    # center = ra, dec
    #center = (d['RA_center'], d['DEC_center'])

    if verbose: print("Time: len(time) = {} \n t0 {} \n time/2 {} \n tf {}".format(time, 
                                                                                   time[0],
                                                                                   time[int(len(time)/2)],
                                                                                  time[-1])  )
    
    ### scan psi as well,
    pitch = pp.time * angspeed_psi
    pitch = pitch % (4 * maxpsi)
    mask = pitch > (2 * maxpsi)
    pitch[mask] = -pitch[mask] + 4 * maxpsi
    pitch -= maxpsi
    
    pp.pitch = pitch
    
    if random_hwp:
        pp.angle_hwp = np.random.randint(0, 7, nsamples) * 11.25
        
    if d['fix_azimuth']['apply']:
        pp.fix_az=True
        if d['fix_azimuth']['fix_hwp']:
            pp.angle_hwp=pp.pitch*0+ 11.25
        if d['fix_azimuth']['fix_pitch']:
            pp.pitch= 0
    else:
        pp.fix_az=False

    return pp


def npp(ipix, nest = False):
    
    #neighboring pixel of the peak
    # map in nest
    return np.array(hp.get_all_neighbours(256, ipix,nest = nest))

def selectcenter(hpmap, center, delta = 3, nside = 256, nest = False,
                threshold = 3, displaycenters = False, doplot = False):
    
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
    if doplot:
        hp.gnomview(hpmap, reso = 12, rot = np.array([90,0]) - newcenter, nest = nest)
        if displaycenters:
            for each in centerarr:
                hp.projscatter(np.deg2rad(each), marker = '+', color = 'r')
        hp.projscatter(hp.pix2ang(256, pixmax, nest = nest), marker = '+', color = 'r')

    return pixmax, newcenter

def thph_qsoft(qinst, scene, soft_pix, PiRot = True):
    
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
        th_tes_all = np.zeros((sim_th.shape[1]) ) 
        ph_tes_all = np.zeros((sim_th.shape[1]) )

        #for i in range(sim_th.shape[1]):
        newuv = np.dot(sbfit.rotmatZ(np.pi), sbfit.thph2uv(sim_th[0], sim_ph[0]) )
        newthph = sbfit.uv2thph(newuv)
        th_tes_all = newthph[0]
        ph_tes_all = newthph[1]
    else: 
        th_tes_all = sim_th
        ph_tes_all = sim_ph
    
    return th_tes_all, ph_tes_all

def flat2hp_map(maps, az, el, nside = 256):
    
    auxth, auxph = np.pi/2 - np.deg2rad(el), np.deg2rad(az)#*np.cos(np.deg2rad(el))
    auxpix = hp.ang2pix(nside, auxth, auxph)
    auxmap = np.zeros((12 * nside ** 2))
    auxmap[auxpix] = maps
    
    return auxmap

class SbHealpyModel(object):
    
    """
    This class aims to search (no fit) the location and amplitude of the peaks in healpix projections for TODs 
    obtained in qubic simulations or from the data. Uses a pointing strategy adapted from
    qubicsoft using scan azimuth and elevation coordinates.     
    """
    def __init__(self, d, q, s, pixnum, az, el, dirfiles, asic = None, id_fp_sys = "FileName", 
                 npix = None, tes = None, qpix = None, verbose = False, PiRot = True,
                 nest = False, simulation = False, maps = None, startpars= None):
        
        """
        d: 
            Qubic dictionary

        pixnum:
            Could be: pixel detector number (id_fp_sys = "FileName"), 
                      QubicInstrument detector number (id_fp_sys = 'qsName')
                      TES, ASIC detector number (id_fp_sys = 'TESName')
        az, el:
            Array with azimuth and elevation coordinates. 
            len(az) = #azimuth in a fixed elevation step
            len(el) = #elevation in a fixed azimuth step

        id_fp_sys: 
            FileName --> user provides the number of TES in this convention: 1 - 256 in continuos way. This
                        convention is not equal to the qubicsoft indexing. 
                    kw needed: npix
            TESName --> user provides (tes, asic)
                    kw needed: tes, asic
            qsName --> qubicsoft indexing
                    kw needed: qpix
        startpars: 
            Array with th, ph coords of central peak. Dims: [2, npeaks]. npeaks should be grater than 8 (9 dims)
        nest: 
            Healpix parameter. NESTED or RING. Default: True. 
        """
        
        self.instrument = q
        self.scene = s
        self.nside = d['nside']
        self.npixels = 12 * self.nside ** 2
        self.verbose = verbose
        self.pirot = PiRot
        # Generate pixel, detector and TES numbers identification 
        self._init_id(id_fp_sys, pixnum, asic = asic,)
        #self._init_thph(PiRot = self.pirot)
        # Location of Data 
        self.dirfiles = dirfiles
        self.nest = nest
        #Center of the FOV
        self.elcenter = np.mean(el)
        self.azcenter = np.mean(az)

        if startpars is None:
            self.init_th, self.init_ph = thph_qsoft(self.instrument, self.scene, self.qpix, PiRot = self.pirot)
            self.cent_th_tes, self.cent_ph_tes = self.init_th[0], self.init_ph[0]
        else: 
            self.init_th, self.init_ph = startpars
            self.cent_th_tes, self.cent_ph_tes = self.init_th[0], self.init_ph[0]

        # Save vector coordinates of central peak 
        self.vec_tes = np.array([np.sin(self.cent_th_tes) * np.cos(self.cent_ph_tes),
                        np.sin(self.cent_th_tes) * np.sin(self.cent_ph_tes),
                        np.cos(self.cent_th_tes)])
        
        #vecs_fullmap = hp.pix2vec(self.nside, range(0, self.npixels), nest = self.nest)
    
    def __call__(self, filterbeam = 3, simulation = False, maps = None, doplot = False,
                threshold = 2, shift = 0.05, delta = 13, 
                refilter = False, factor_corrector = 1.2):
            
        """
        filterbeam:
            Used to smooth maps with gaussian kernels.
        simulation: 
            If False, uses data files to fit. If True, it fits simulated maps provided in 'maps'.
        threshold, threshold0: 
            Angular distance from a given center (diff for each peak) to define a sub-region to search maximum value.
            The maximum value in the region is considered the central pixel of the peak. threshold0 is the same for central peak.
        shift:
            There are some shifts in projections of two peaks at the edges. This factor try to make the fit better.
            Default: 0.05deg
        delta: 
            Linear angular separation between central peak and farest peak. This value is used for the initial guess.
            Default: 13deg


        """
        #warn("{}: This class adjusts the location and width of the peaks of the synthesized beams. \
        #    Currently, this approach can be used to adjust the 9 main peaks. For more peaks you have to update it ".format(self.__name__))
        
        if not simulation:
            # Carry synth beam from polar cap to the equatorial zone
            # Simulation from 2019 are centered in (0,0). That's the reason for this rotation. Otherwise we could use 
            # just the rotation in y-axis 90 - center_elevation_fov
            self.centerini = [hp.vec2ang(np.dot(sbfit.rotmatY(np.pi/2), self.vec_tes))[0][0],
                                    hp.vec2ang(np.dot(sbfit.rotmatY(np.pi/2), self.vec_tes))[1][0]]
            self.hpmap = sbfit.get_hpmap(self.npix, self.dirfiles)

        elif simulation:
            # Carry synth beam from polar cap to the equatorial one
            self.centerini = [hp.vec2ang(np.dot(sbfit.rotmatY(np.deg2rad(90 - self.elcenter)), self.vec_tes))[0][0],
                                hp.vec2ang(np.dot(sbfit.rotmatY(np.deg2rad(90 - self.elcenter)), self.vec_tes))[1][0]]

            #hpmap = q.get_synthbeam(s)[PIXq, newpix]
            self.hpmap = maps

        if self.nest:
            self.hpmap = hp.reorder(self.hpmap, r2n = self.nest)
        else:
            self.hpmap = self.hpmap
        self.hpmap_filt = f.gaussian_filter(self.hpmap, filterbeam)
        
        # Searching for central peak in healpix map filtered
        self.centerini = [np.rad2deg(self.centerini[0]), np.rad2deg(self.centerini[1])] 
        px, center = selectcenter(self.hpmap_filt, self.centerini, doplot = doplot, nside = self.nside)

        # Convert central pixel in th, ph coordinates 
        thetaphi = hp.pix2ang(self.nside, px, nest = self.nest)

        shift = np.deg2rad(np.array([shift]))
        # Initial angular separation where the peaks will be searched 
        _d = np.deg2rad(delta) * np.sqrt(2)
        _t = np.deg2rad(delta) / np.sqrt(2)
        dmt  =_d - _t
        _fact_ = factor_corrector
        #peaks ordering according JCh and instrument module
        thphpeaks = [thetaphi + np.array([_d, 0]),
                     thetaphi + np.array([dmt, - dmt]),
                     thetaphi + np.array([0, -_d * _fact_] ),#+ shift), 
                     thetaphi + np.array([dmt, dmt]),
                     thetaphi,
                     thetaphi + np.array([-dmt, -dmt]),
                     thetaphi + np.array([0, _d * _fact_]),# + shift),
                     thetaphi + np.array([- dmt, dmt]),
                     thetaphi + np.array([-_d, 0])
                     ]
        if self.verbose:
            print("phi distance between peaks (_d): {:.2f}deg ".format(np.degrees(_d)))
            print("th distance to next theta coord containing a peak(_t): {:.1f}deg ".format(np.degrees(_t)))
            print("difference _d - _t = {:.2f}deg".format(np.degrees(dmt)))
            print("phi distance between peaks without considering a factor_corrector {:.2f}deg".format(np.degrees(_d)) )
            print("and considering factor factor_corrector {:.2f}deg".format(np.degrees(_d*_fact_)))
        #
        #== Searching peaks considering higher signal value within a circular region
        #  (this has to be changed in future for a fitting model)
        # 
        #   realmaxpx: relative maximum in each sub-region 
        #   absmaxpx: absolute maximum --> 
        warn("QUwarn: Searching peaks considering higher signal value within a circular region. \
          this has to be changed in future for a fitting model")

        realmaxpx = np.zeros((len(thphpeaks),), dtype = int)
        absmaxpx = np.zeros(( len(thphpeaks),), dtype = int)

        thphpeaksnew = np.zeros((len(thphpeaks),4))

        fullvec = hp.pix2vec(self.nside, range(0, self.npixels), nest = self.nest)
        ifullpx = np.linspace(0, self.npixels, self.npixels, dtype = int)

        for j, ithphpx in enumerate(thphpeaks):
            c = 'b'
            #if j == 4: 
            #    threshold = threshold0
            #else:
            #    threshold = threshold
            threshold = threshold

            ivec = hp.ang2vec(ithphpx[0], ithphpx[1], )
            maskipx = np.rad2deg(np.arccos(np.dot(ivec,fullvec))) < threshold
            
            if refilter:
                mean, std = np.mean(hpnest_filt[maskipx]),np.std(hpnest_filt[maskipx])
                maskipx2 = hpnest_filt[maskipx] < mean+3*std
                maskipx3 = hpnest_filt[maskipx] > mean-3*std
                maskipx[maskipx] = maskipx2 * maskipx3
            maskidx = np.where(maskipx == True)[0]
            #useless max (just to plot in 1d not healpix)
            realmaxpx[j] = np.where(self.hpmap_filt[maskipx] == np.max(self.hpmap_filt[maskipx]))[0][0]
            #usefull max (healpix)
            absmaxpx[j] = maskidx[realmaxpx[j]]
            #Ssave coords of peaks
            thphpeaksnew[j,:2] = hp.pix2ang(self.nside, absmaxpx[j], nest = self.nest)
            #Save amplitude
            thphpeaksnew[j,2] = self.hpmap_filt[absmaxpx[j]]
            thphpeaksnew[j,3] = j

        if doplot:
            
            vli = [px,]
            xlo = px-100
            xhi = px+100
            clf()
            fig, ax = subplots(nrows = 1, ncols = 2, figsize = (12,7))
            ax[0].set_title("Center peak")
            ax[0].set_xlim(xlo,xhi)
            ax[0].plot(self.hpmap, 'b-', label = 'raw')
            ax[0].plot(self.hpmap_filt, 'b-', alpha = 0.4, label = 'filtered')
            ax[0].legend()
            for i in vli:
                ax[0].axvline(i, c = 'k', alpha = 0.4, ls = '--')
                #neig = npp(i)
            axs = axes(ax[1])
            hp.gnomview(self.hpmap, reso = 12, rot = (0, 50), nest = self.nest, hold = True)
            #hp.mollview(self.hpmap, rot = (0, 50), nest = True, hold = True)
            hp.projscatter(np.deg2rad(self.centerini), marker = '+', color = 'r')

            if self.verbose: print(np.shape(thphpeaks))
            for j, ithph in enumerate(thphpeaks):
                hp.projscatter(ithph, marker = '+', color = 'b', label = "initial guess" if j == 0 else None)

            legend()

            #fig, ax = subplots(nrows = 9, ncols = 1, figsize = (8,8),)
            #    for j, ithphpx in enumerate(thphpeaks):
            #        c = 'b'
            #        if j == 3: c = 'r'
            #        ax[j].axvline(realmaxpx[j], c = 'k', alpha = 0.4, ls = '--')
            #        ax[j].plot(self.hpmap_filt[maskipx], 'o--', color = c, alpha = 0.4, label = 'filtered')
            #        ax[j].legend()        

        return self.hpmap_filt, thphpeaksnew.T

    def _init_id(self, ident_focalplane, num, asic = None):
        """
        Generates focal plane identifications from user input
        Parameter:
            num: is the detector or pixel number. 
            asic: ASIC number if ident_focalplane = TESName and num has to be the tes number.
        """
        if ident_focalplane == 'FileName':
            self.npix = num
            self.tes, self.asic = (self.npix, 1) if (self.npix < 128) else (self.npix - 128, 2)
            self.qpix = tes2pix(self.tes, self.asic) - 1
        elif ident_focalplane == 'qsName':
            FPidentity = make_id_focalplane()
            self.qpix = num
            det_index = self.instrument.detector[self.qpix].index[0]
            self.tes = FPidentity[det_index].TES
            self.asic = FPidentity[det_index].ASIC
            self.npix = self.tes if self.asic == 1  else self.tes + 128
        elif ident_focalplane == 'TESName':
            if num > 128:
                raise ValueError("Wrong TES value. You gave a TES number greater than 128.")
            else:
                if asic == None:
                    raise ValueError("You choose {} identification but ASIC number is missing.".format(ident_focalplane))
                else: 
                    self.tes = num
                    self.asic = asic
                    self.npix = self.tes if self.asic == 1  else self.tes + 128
                    self.qpix = tes2pix(self.tes, self.asic) - 1
        if self.verbose: 
            print("You are running fitting in healpix maps.")
            print("========================================")
            print("TES number {} asic number {}".format(self.tes, self.asic))
            print("In FileName format the number of tes is {}".format(self.npix))
            print("Index number: qpack {} qsoft {} ".format(\
                                    tes2index(self.tes, self.asic), self.instrument.detector[self.qpix].index[0] ))
            print("qubicsoft number: {}".format(self.qpix))
        return

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