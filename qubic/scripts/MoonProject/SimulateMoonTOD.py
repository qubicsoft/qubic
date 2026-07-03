### Here we create the functions necessary to simulate TOD
### to test our Moon spectrum reconstruction pipeline

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support
quantity_support()
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, get_body

import gc

import healpy as hp
from qubic.lib import Qacquisition, Qsamplings

import pipeline_moon_functions as pmf


def get_TOD(d, qsmoon, y0):
    print("in get_TOD", flush=True)
    ### We use our fake moon-frame pointing qsmoon
    qsmoon.fix_az = False
    # a = Qacquisition.QubicMultiAcquisitions(q, qsmoon, s, d, nus_edge_in)
    a = Qacquisition.QubicMultiAcquisitions(d, nsub=d['nf_sub'], nrec=2, comps=[], H=None, nu_co=None, sampling=qsmoon) # because QAcquisition not fixed on this branch
    # TOD_nsub = []
    # for ind_a, a_i in enumerate(a.subacqs[:len(a.subacqs)//2]): # only need first band (to be updated with MonoBand)
    for ind_a, a_i in enumerate(a.subacqs): # only need first band (to be updated with MonoBand)
        TOD_i, _ = a_i.get_observation(y0[ind_a, :], noiseless=d['noiseless'], convolution = True)
        if ind_a == 0:
            TOD = TOD_i.copy()
        else:
            TOD += TOD_i
        # TOD_nsub.append(TOD_i)
    # H = a.H.copy()
    del a
    gc.collect()   ### Important ! Frees the memory
    return TOD#, H, TOD_nsub


def create_moon_tod(d, azt, elt, tt, Obs_Site, moon_spectrum):
    '''

    Obs_Site: astropy.coordinates.EarthLocation object
        the observation site.
    '''
    longitude = float(Obs_Site.lon/u.deg)
    latitude = float(Obs_Site.lat/u.deg)
    ### We remove tt[0] but we save tinit in date_obs
    tinit = tt[0]
    tt -= tinit
    date_obs = str(datetime.utcfromtimestamp(tinit))
    print('Observations started at: {} UTC (old)'.format(date_obs))
    ### This QubicSampling object is just useful for corrd conversion
    # qs = Qsamplings.QubicSampling(azimuth=azt, elevation=elt, time=tt,
    #                date_obs=date_obs, longitude=longitude, latitude=latitude)
    
    ############################### Using Astropy #########################
    alltimes = Time(date_obs) + tt*u.second #+ 3*u.minute

    ### Local coordinates
    frame_obs = AltAz(obstime=alltimes , location=Obs_Site)

    ### Moon
    moon_observed = get_body('Moon', alltimes, Obs_Site)

    ### RA/DEC
    moonra = moon_observed.ra
    moondec = moon_observed.dec

    # Computing the mean angular size of the Moon
    moon_dist = moon_observed.distance.value * pmf.AU_meters # m
    moon_size = 2 * 1737.4 * 1e3 # m
    moon_angsize = np.mean(np.arctan(moon_size/moon_dist)) # radians
    print("Moon angular size = {} degree".format(moon_angsize * 180/np.pi))

    ########################################################################

    ### To get the Moon moving, the idea is to subtract the movement of the Moon
    ### to fake a new referential where it doesn't move

    ### Shift in RA,DEC
    # movement = position - average(position)
    shiftra = (moonra - np.mean(moonra))/u.deg
    shiftdec = (moondec - np.mean(moondec))/u.deg

    # new observed coordinates corrected for the Moon movement
    ra, dec = Qsamplings.hor2equ(azt, elt, tt + tinit)
    newra = ra - shiftra
    newdec = dec - shiftdec

    # new position of the Moon corrected for the Moon movement
    # should be constant
    newmoonra = moonra/u.deg - shiftra
    newmoondec = moondec/u.deg - shiftdec

    ### Convert the Moon new position to Gal coordinates
    newmoonl, newmoonb = Qsamplings.equ2gal(newmoonra, newmoondec)

    ### This is just to add a shift by hand, will be removed later
    # deltazt, deltelt = 4.5658359258853585, 1.4750926931640054
    deltazt, deltelt = 0, 0

    ### Now we need to go to local coordinates az,el...
    altaz = SkyCoord(newra*u.deg, newdec*u.deg, frame='icrs').transform_to(frame_obs) 
    newaz = altaz.az.value + deltazt # trying to shift by hand the pointings
    newel = altaz.alt.value + deltelt

    print("Getting to new Qsamplings", flush=True)
    ### This new Qsamplings object is the one used for the simulated observation
    qsmoon = Qsamplings.QubicSampling(azimuth=newaz, elevation=newel, time=tt, # time since date_obs
                    date_obs=date_obs, longitude=longitude, latitude=latitude,
                    pitch=0) # put pitch to zero to test (might be close to 5 deg in real data?)
    
    ### Now we know the equatorial/Galactic location of the moon in this referential attached to the Moon. We can create an image of the moon there
    nside = d["nside"]

    lmoon_av = np.mean(newmoonl)  ### Mean but in fact it is constant
    bmoon_av = np.mean(newmoonb)  ### Mean but in fact it is constant

    print(lmoon_av, bmoon_av)
    uvmoon = hp.ang2vec(np.radians(90. - bmoon_av), np.radians(lmoon_av))

    pixmoon = hp.query_disc(nside, uvmoon, moon_angsize, inclusive=True)
    map_in = np.zeros(12*nside**2)
    map_in[pixmoon] = 1 # approximation for now

    # plt.figure()
    # hp.mollview(map_in, sub=(2, 2, 1))
    # hp.gnomview(map_in, rot=[lmoon_av, bmoon_av], reso=10, sub=(2, 2, 2))

    # ### We can calculate the coverage from the l,b of the pointing and display it
    # ipcov = hp.ang2pix(nside, np.radians(90. - qsmoon.galactic[:, 1]), np.radians(qsmoon.galactic[:, 0]))
    # mapcov = map_in.copy()
    # mapcov[ipcov] += 1
    # hp.mollview(mapcov, sub=(2, 2, 3))
    # hp.gnomview(mapcov, rot=[lmoon_av, bmoon_av], reso=10, sub=(2, 2, 4))
    # plt.show()
    # arey
    
    ### Let's simulate TOD from this

    # Create an input map with the moon in each sub-frequency
    y0 = np.empty((d['n_nu_fit'], 12 * d['nside'] ** 2))
    for i in range(d['n_nu_fit']):
        y0[i, :] = map_in * moon_spectrum[i]

    TOD = get_TOD(d, qsmoon, y0)

    return TOD
    