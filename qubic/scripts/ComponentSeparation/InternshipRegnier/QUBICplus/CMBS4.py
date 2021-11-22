from qubic import camb_interface as qc
import healpy as hp
import numpy as np
import qubicplus
import pysm3
import pysm3.units as u
from pysm3 import utils
from qubic import camb_interface as qc
import matplotlib.pyplot as plt
import os
import random as rd
import string
import qubic
from importlib import reload

def theoretical_noise_maps(sigma_sec, coverage, Nyears=4, verbose=False):
    """
    This returns a map of the RMS noise (not an actual realization, just the expected RMS - No covariance)

    Parameters
    ----------
    sigma_sec: float
        Noise level.
    coverage: array
        The coverage map.
    Nyears: int
    verbose: bool

    Returns
    -------

    """
    # ###### Noise normalization
    # We assume we have integrated for a time Ttot in seconds with a sigma per root sec sigma_sec
    Ttot = Nyears * 365 * 24 * 3600  # in seconds
    if verbose:
        print('Total time is {} seconds'.format(Ttot))
    # Oberved pixels
    thepix = coverage > 0
    # Normalized coverage (sum=1)
    covnorm = coverage / np.sum(coverage)
    if verbose:
        print('Normalized coverage sum: {}'.format(np.sum(covnorm)))

    # Time per pixel
    Tpix = np.zeros_like(covnorm)
    Tpix[thepix] = Ttot * covnorm[thepix]
    if verbose:
        print('Sum Tpix: {} s  ; Ttot = {} s'.format(np.sum(Tpix), Ttot))

    # RMS per pixel
    Sigpix = np.zeros_like(covnorm)
    Sigpix[thepix] = sigma_sec / np.sqrt(Tpix[thepix])
    if verbose:
        print('Total noise (with no averages in pixels): {}'.format(np.sum((Sigpix * Tpix) ** 2)))
    return Sigpix

def give_me_nus(nu, largeur, Nf):
    largeurq=largeur/Nf
    min=nu-largeur
    max=nu+largeur
    arr = np.linspace(min, max, Nf+1)
    mean_nu = np.zeros(Nf)

    for i in range(len(arr)-1):
        mean_nu[i]=np.mean(np.array([arr[i], arr[i+1]]))

    return mean_nu

def smoothing(maps, FWHMdeg, Nf, central_nus, verbose=True):
        """Convolve the maps to the FWHM at each sub-frequency or to a common beam if FWHMdeg is given."""
        fwhms = np.zeros(Nf)
        if FWHMdeg is not None:
            fwhms += FWHMdeg
        for i in range(Nf):
            if fwhms[i] != 0:
                maps[i, :, :] = hp.sphtfunc.smoothing(maps[i, :, :].T, fwhm=np.deg2rad(fwhms[i]),
                                                      verbose=verbose).T
        return fwhms, maps

def integrated(central_nus, bandwidth):
    min_nu = central_nus * (1 - 0.5 * bandwidth)
    max_nu = central_nus * (1 + 0.5 * bandwidth)
    nus_reconstructed = np.linspace(min_nu, max_nu, 4)

    return min_nu, max_nu, nus_reconstructed

def random_string(nchars):
    lst = [rd.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)

def get_multiple_nus(nu, bw, nf):
    nus=np.zeros(nf)
    edge=np.linspace(nu-(bw/2), nu+(bw/2), nf+1)
    for i in range(len(edge)-1):
        #print(i, i+1)
        nus[i]=np.mean([edge[i], edge[i+1]])
    return nus

def give_me_nu_fwhm_S4_2_qubic(nu, largeur, Nf, fwhmS4):

    def give_me_fwhm(nu, nuS4, fwhmS4):
        return fwhmS4*nuS4/nu

    largeurq=largeur/Nf
    min=nu*(1-largeur/2)
    max=nu*(1+largeur/2)
    arr = np.linspace(min, max, Nf+1)
    mean_nu = get_multiple_nus(nu, largeur, Nf)

    fwhm = give_me_fwhm(mean_nu, nu, fwhmS4)

    return mean_nu, fwhm

def compute_freq(band, Nfreq=None, relative_bandwidth=2.5):
    """
    Prepare frequency bands parameters
    band -- int,
        QUBIC frequency band, in GHz.
        Typical values: 150, 220
    relative_bandwidth -- float, optional
        Ratio of the difference between the edges of the
        frequency band over the average frequency of the band:
        2 * (nu_max - nu_min) / (nu_max + nu_min)
        Typical value: 0.25
    Nfreq -- int, optional
        Number of frequencies within the wide band.
        If not specified, then Nfreq = 15 if band == 150
        and Nfreq = 20 if band = 220
    """

    if Nfreq is None:
        Nfreq = {150: 15, 220: 20}[band]

    nu_min = band * (1 - relative_bandwidth / 2)
    nu_max = band * (1 + relative_bandwidth / 2)

    Nfreq_edges = Nfreq + 1
    base = (nu_max / nu_min) ** (1. / Nfreq)

    nus_edge = nu_min * np.logspace(0, Nfreq, Nfreq_edges, endpoint=True, base=base)
    nus = np.array([(nus_edge[i] + nus_edge[i - 1]) / 2 for i in range(1, Nfreq_edges)])
    deltas = np.array([(nus_edge[i] - nus_edge[i - 1]) for i in range(1, Nfreq_edges)])
    Delta = nu_max - nu_min
    Nbbands = len(nus)
    return Nfreq_edges, nus_edge, nus, deltas, Delta, Nbbands

def nus_for_iib(nu, bw):
    min=nu-(bw/2)
    max=nu+(bw/2)
    #print(min, max)
    nus = np.linspace(min, max, 4)
    return nus


class S4(object):

    def __init__(self, skyconfig, dict):
        self.dict = dict
        self.skyconfig = skyconfig
        self.nus = self.dict['frequency']
        self.fwhm_arcmin = self.dict['fwhm']
        self.fwhm = self.fwhm_arcmin/60
        self.bw = self.dict['bandwidth']
        self.edges = self.dict['edges']
        self.fsky = self.dict['fsky']
        self.depth_i = self.dict['depth_i']
        self.depth_p = self.dict['depth_p']

        self.nside=256
        self.npix = 12*self.nside**2
        self.lmax=3 * self.nside
        ell, totDL, unlensedCL = qc.get_camb_Dl(lmax=self.lmax)
        mycls = qc.Dl2Cl_without_monopole(ell, totDL)
        mymaps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        self.input_cmb_maps = mymaps
        self.input_cmb_spectra = totDL
        for k in skyconfig.keys():
            if k == 'cmb':
                self.seed = self.skyconfig['cmb']

    def get_cmb(self):
        np.random.seed(self.seed)
        ell, totDL, unlensedCL = qc.get_camb_Dl(lmax=2*self.nside+1)
        mycls = qc.Dl2Cl_without_monopole(ell, totDL)
        maps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        return maps

    def get_sky(self):
        setting = []
        iscmb=False
        for k in self.skyconfig:
            if k == 'cmb' :
                iscmb=True
                maps = self.get_cmb()

                rndstr = random_string(10)
                hp.write_map('/tmp/' + rndstr, maps)
                cmbmap = pysm3.CMBMap(self.nside, map_IQU='/tmp/' + rndstr)
                os.remove('/tmp/' + rndstr)
                #setting.append(skyconfig[k])
            else:
                setting.append(self.skyconfig[k])

        sky = pysm3.Sky(nside=self.nside, preset_strings=setting)
        if iscmb:
            sky.add_component(cmbmap)

        return sky

    def getskymaps(self, same_resol=False, verbose=False, coverage=None, iib=False, noise=False):

        """
        This function returns fullsky at S4 or QUBIC+ frequencies and resolutions.
        """

        sky=self.get_sky()
        allmaps = np.zeros(((len(self.nus), 3, self.npix)))

        if verbose:
            print("    FWHM : {} deg \n    nus : {} GHz \n    Bandwidth : {} GHz\n\n".format(self.fwhm, self.nus, self.bw))

        if iib :
            def get_nus_for_iib(edge):
                min = edge[0]
                max = edge[1]
                nus = np.linspace(min, max, 8)

                return nus

            for indi, i in enumerate(self.nus):
                nus_edges=get_nus_for_iib(self.edges[indi])
                maps_edges=np.zeros(((8, 3, self.npix)))
                if verbose:
                    print("Integrated from {:.2f} to {:.2f} GHz".format(nus_edges[0], nus_edges[-1]))
                for indj, j in enumerate(nus_edges):
                    maps_edges[indj] = sky.get_emission(j*u.GHz, np.ones(4))*utils.bandpass_unit_conversion(j*u.GHz,np.ones(4), u.uK_CMB)
                    if same_resol:
                        if verbose and indj == 1:
                            print('Reconvolution to {:.2f} deg'.format(np.max(self.fwhm)))
                        maps_edges[indj] = hp.sphtfunc.smoothing(maps_edges[indj, :, :], fwhm=np.deg2rad(np.max(self.fwhm)),verbose=False)
                    else:
                        if verbose and indj == 1:
                            print('Reconvolution to {:.2f} deg'.format(self.fwhm[indi]))
                        maps_edges[indj] = hp.sphtfunc.smoothing(maps_edges[indj, :, :], fwhm=np.deg2rad(self.fwhm[indi]),verbose=False)


                allmaps[indi] = np.mean(maps_edges, axis=0)
        else:
            for ind, i in enumerate(self.nus):
                allmaps[ind] = sky.get_emission(i*u.GHz, None)*utils.bandpass_unit_conversion(i*u.GHz,None, u.uK_CMB)

                if same_resol:
                    if verbose:
                        print('Reconvolution to {:.2f} deg'.format(np.max(self.fwhm)))
                    allmaps[ind] = hp.sphtfunc.smoothing(allmaps[ind, :, :], fwhm=np.deg2rad(np.max(self.fwhm)),verbose=False)
                else:
                    if verbose:
                        print('Reconvolution to {:.2f} deg'.format(self.fwhm[ind]))
                    allmaps[ind] = hp.sphtfunc.smoothing(allmaps[ind, :, :], fwhm=np.deg2rad(self.fwhm[ind]),verbose=False)

        def create_noisemaps():
            N = np.zeros(((len(self.nus), 3, self.npix)))
            for ind_nu, nu in enumerate(self.nus):
                sig_i=self.depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
                N[ind_nu, 0] = np.random.normal(0, sig_i, 12*self.nside**2)
                sig_p=self.depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60)
                N[ind_nu, 1] = np.random.normal(0, sig_p, 12*self.nside**2)*np.sqrt(2)
                N[ind_nu, 2] = np.random.normal(0, sig_p, 12*self.nside**2)*np.sqrt(2)

            return N

        if noise:
            noisemaps = create_noisemaps()
            maps_noisy = allmaps+noisemaps

            if coverage is not None:
                thr = 0.1
                mymask = (coverage > (np.max(coverage)*thr)).astype(int)
                pixok = mymask > 0
                maps_noisy[:, :, ~pixok] = hp.UNSEEN
                noisemaps[:, :, ~pixok] = hp.UNSEEN
                allmaps[:, :, ~pixok] = hp.UNSEEN

            return maps_noisy, allmaps, noisemaps

        if coverage is not None:
            thr = 0.1
            mymask = (coverage > (np.max(coverage)*thr)).astype(int)
            pixok = mymask > 0
            allmaps[:, :, ~pixok] = hp.UNSEEN
        return allmaps




    def create_noisemaps(self, coverage, nsub, index=None, covcut=0.1, verbose=False, eps=0, f=1):

        #seenpix = (coverage / np.max(coverage)) > covcut
        #npix = seenpix.sum()

        # Sigma_sec for each Stokes: by default they are the same unless there is non trivial covariance
        fact_I = np.ones(nsub)
        fact_Q = np.ones(nsub)
        fact_U = np.ones(nsub)

        depth_i = self.depth_i
        depth_p = self.depth_p

        if coverage is not None:
            thr = covcut
            mymask = (coverage > (np.max(coverage)*thr)).astype(int)
            pixok = mymask > 0
        else:
            pixok = np.ones(self.npix).astype(bool)

        noise_maps = np.zeros((nsub, self.npix, 3))
        for isub in range(nsub):
            #print("depth i : {}".format(depth_i[isub]))
            #print("depth p : {}".format(depth_p[isub]))
            IrndFull = np.random.normal(0, depth_i[isub], np.sum(pixok))
            QrndFull = np.random.normal(0, depth_p[isub], np.sum(pixok))
            UrndFull = np.random.normal(0, depth_p[isub], np.sum(pixok))

            ### put them into the whole sub-bandss array
            noise_maps[isub, pixok, 0] = IrndFull
            noise_maps[isub, pixok, 1] = QrndFull
            noise_maps[isub, pixok, 2] = UrndFull

        return np.transpose(noise_maps, (0, 2, 1))

    def theoretical_noise_maps(self, sigma_sec, coverage, Nyears=4, verbose=False):
        """
        This returns a map of the RMS noise (not an actual realization, just the expected RMS - No covariance)

        Parameters
        ----------
        sigma_sec: float
            Noise level.
        coverage: array
            The coverage map.
        Nyears: int
        verbose: bool

        Returns
        -------

        """

        # ###### Noise normalization
        # We assume we have integrated for a time Ttot in seconds with a sigma per root sec sigma_sec
        Ttot = Nyears * 365 * 24 * 3600  # in seconds
        if verbose:
            print('Total time is {} seconds'.format(Ttot))
        # Oberved pixels
        thepix = coverage > 0
        # Normalized coverage (sum=1)
        covnorm = coverage / np.sum(coverage)
        if verbose:
            print('Normalized coverage sum: {}'.format(np.sum(covnorm)))

        # Time per pixel
        Tpix = np.zeros_like(covnorm)
        Tpix[thepix] = Ttot * covnorm[thepix]
        if verbose:
            print('Sum Tpix: {} s  ; Ttot = {} s'.format(np.sum(Tpix), Ttot))

        # RMS per pixel
        Sigpix = np.zeros_like(covnorm)
        Sigpix[thepix] = sigma_sec / np.sqrt(Tpix[thepix])
        if verbose:
            print('Total noise (with no averages in pixels): {}'.format(np.sum((Sigpix * Tpix) ** 2)))
        return Sigpix
