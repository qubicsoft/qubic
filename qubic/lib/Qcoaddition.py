### General modules
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from lib.Calibration import Qfiber as ft
from lib.Calibration import Qselfcal as scal
from lib.Instrument.Qinstrument import QubicInstrument
from lib.Qcoaddition import get_mode
from lib.Qdictionary import qubicDict
from lib.Qutilities import progress_bar
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import medfilt

__all__ = ["MySplineFitting", "Pip1Tes", "FocalPlane", "PipAllTes"]


class MySplineFitting:
    def __init__(self, xin, yin, covarin, nbspl, logspace=False):
        # input parameters
        self.x = xin
        self.y = yin
        self.nbspl = nbspl
        if np.size(np.shape(covarin)) == 1:
            err = covarin
            self.covar = np.diag(err**2)
            self.invcovar = np.diag(1.0 / err**2)
        else:
            self.covar = covarin
            self.invcovar = np.linalg.inv(covarin)

        # Prepare splines
        xspl = np.linspace(np.min(self.x), np.max(self.x), nbspl)
        if logspace:
            xspl = np.logspace(np.log10(np.min(self.x)), np.log10(np.max(self.x)), nbspl)
        self.xspl = xspl
        F = np.zeros((np.size(xin), nbspl))
        self.F = F
        for i in np.arange(nbspl):
            self.F[:, i] = self.get_spline_tofit(xspl, i, xin)

        # solution of the chi square
        ft_cinv_y = np.dot(np.transpose(F), np.dot(self.invcovar, self.y))
        covout = np.linalg.inv(np.dot(np.transpose(F), np.dot(self.invcovar, F)))
        alpha = np.dot(covout, ft_cinv_y)
        fitted = np.dot(F, alpha)

        # output
        self.residuals = self.y - fitted
        self.chi2 = np.dot(np.transpose(self.residuals), np.dot(self.invcovar, self.residuals))
        self.ndf = np.size(xin) - np.size(alpha)
        self.alpha = alpha
        self.covout = covout
        self.dalpha = np.sqrt(np.diagonal(covout))

    def __call__(self, x):
        theF = np.zeros((np.size(x), self.nbspl))
        for i in np.arange(self.nbspl):
            theF[:, i] = self.get_spline_tofit(self.xspl, i, x)
        return np.dot(theF, self.alpha)

    def with_alpha(self, x, alpha):
        theF = np.zeros((np.size(x), self.nbspl))
        for i in np.arange(self.nbspl):
            theF[:, i] = self.get_spline_tofit(self.xspl, i, x)
        return np.dot(theF, alpha)

    def get_spline_tofit(self, xspline, index, xx):
        yspline = np.zeros(np.size(xspline))
        yspline[index] = 1.0
        tck = interpolate.splrep(xspline, yspline)
        yy = interpolate.splev(xx, tck, der=0)
        return yy


class Pip1Tes:
    """

    Instance to treat TES one by one.

    """

    def __init__(self, tt, tod, az, el, thk):
        self.tt = tt - tt[0]
        self.tod = tod - tod[0]
        self.az = az
        self.el = el
        self.thk = thk
        self.scantype_hk, self.azt, self.elt, self.scantype = self._scantype(0.035)

    def identify_scans(self, inthk, az, el, tt=None, median_size=101, thr_speedmin=0.1, plotrange=[0, 1000]):
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

        thk = inthk - inthk[0]

        ### Sampling for HK data
        timesample = np.median(thk[1:] - thk[:-1])
        ### angular velocity
        medaz_dt = medfilt(np.gradient(az, timesample), median_size)
        ### Identify regions of change
        # Low velocity -> Bad
        c0 = np.abs(medaz_dt) < thr_speedmin
        # Positive velicity => Good
        cpos = (~c0) * (medaz_dt > 0)
        # Negative velocity => Good
        cneg = (~c0) * (medaz_dt < 0)

        ### Scan identification at HK sampling
        scantype_hk = np.zeros(len(thk), dtype="int") - 10
        scantype_hk[c0] = 0
        scantype_hk[cpos] = 1
        scantype_hk[cneg] = -1

        # check that we have them all
        count_them = np.sum(scantype_hk == 0) + np.sum(scantype_hk == -1) + np.sum(scantype_hk == 1)
        if count_them != len(scantype_hk):
            print("Identify_scans: Bad Scan counting at HK sampling level - Error: {} {}".format(len(scantype_hk), count_them))
            raise RuntimeError("Bad scan counting at HK sampling level")

        ### Now give a number to each back and forth scan
        num = 0
        previous = 0
        for i in range(len(scantype_hk)):
            if scantype_hk[i] == 0:
                previous = 0
            else:
                if (previous == 0) & (scantype_hk[i] > 0):
                    # we have a change
                    num += 1
                previous = 1
                scantype_hk[i] *= num

        if tt is not None:
            ### We propagate these at TOD sampling rate  (this is an "step interpolation": we do not want intermediatee values")
            scantype = interp1d(thk, scantype_hk, kind="previous", fill_value="extrapolate")(tt)
            scantype = scantype.astype(int)
            count_them = np.sum(scantype == 0) + np.sum(scantype <= -1) + np.sum(scantype >= 1)
            if count_them != len(scantype):
                print("Bad Scan counting at data sampling level - Error")
                raise RuntimeError("Bad scan counting at data sampling level")
            ### Interpolate azimuth and elevation to TOD sampling
            azt = np.interp(tt, thk, az)
            elt = np.interp(tt, thk, el)
            ### Return evereything
            return scantype_hk, azt, elt, scantype
        else:
            ### Return scantype at HK sampling only
            return scantype_hk

    def _scantype(self, speedmin):
        """

        Method to run scantype analysis.

        Parameters :
        ------------
                - speedmin : float number that describe the mount speed.

        """

        scantype_hk, azt, elt, scantype = self.identify_scans(self.thk, self.az, self.el, tt=self.tt, plotrange=[0, 2000], thr_speedmin=speedmin)
        return scantype_hk, azt, elt, scantype

    def _remove_drifts_spline(self, nsplines=40, nresample=1000, nedge=100):
        """

        Method to remove drifts using spline interpolation.

        """

        ### Linear function using the beginning and end of the samples (nn samples)
        ### In order to approach periodicity of the signal to be resampled
        x0 = np.mean(self.tt[:nedge])
        y0 = np.mean(self.tod[:nedge])
        x1 = np.mean(self.tt[-nedge:])
        y1 = np.mean(self.tod[-nedge:])
        p = np.poly1d(np.array([(y1 - y0) / (x1 - x0), y0]))

        ### Resample TOD-linearfunction to apply spline
        tt_resample = np.linspace(self.tt[0], self.tt[-1], nresample)
        tod_resample = scipy.signal.resample(self.tod - p(self.tt), nresample) + p(tt_resample)

        ### Perform spline fitting
        splfit = MySplineFitting(tt_resample, tod_resample, tod_resample * 0 + 1, nsplines)

        return self.tod - splfit(self.tt)

    def _remove_offset_scan(self, method="meancut", apply_to_bad=True):
        """

        Method to remove offsets.

        """

        ### We remove offsets for each good scan but we also need to remove a coomparable offset for the scantype==0 reggiions in order to keep coninuity
        ### This si donee by apply_to_bad=True

        indices = np.arange(len(self.tod))
        last_index = 0
        myoffsetn = 0
        myoffsetp = 0
        donefirst = 0

        nscans = np.max(np.abs(self.scantype))
        for n in range(1, nscans + 1):
            # scan +
            ok = self.scantype == n
            if method == "meancut":
                myoffsetp, _ = ft.meancut(self.tod[ok], 3)
            elif method == "median":
                myoffsetp = np.median(self.tod[ok])
            elif method == "mode":
                myoffsetp = get_mode(self.tod[ok])
            else:
                break
            self.tod[ok] -= myoffsetp
            if apply_to_bad:
                first_index = np.min(indices[ok])
                if (n == 1) & (donefirst == 0):
                    myoffsetn = myoffsetp  ### deal with first region
                vals_offsets = myoffsetn + np.linspace(0, 1, first_index - last_index - 1) * (myoffsetp - myoffsetn)
                self.tod[last_index + 1 : first_index] -= vals_offsets
                last_index = np.max(indices[ok])
                donefirst = 1

                # scan -
            ok = self.scantype == (-n)
            if method == "meancut":
                myoffsetn, _ = ft.meancut(self.tod[ok], 3)
            elif method == "median":
                myoffsetn = np.median(self.tod[ok])
            elif method == "mode":
                myoffsetn = get_mode(self.tod[ok])
            else:
                break
            self.tod[ok] -= myoffsetn
            if apply_to_bad:
                first_index = np.min(indices[ok])
                if (n == 1) & (donefirst == 0):
                    myoffsetp = myoffsetn  ### deal with first region
                vals_offsets = myoffsetp + np.linspace(0, 1, first_index - last_index - 1) * (myoffsetn - myoffsetp)
                self.tod[last_index + 1 : first_index] -= vals_offsets
                last_index = np.max(indices[ok])
                donefirst = 1

        return self.tod

    def decorel_azel(self, nbins=50, n_el=30, degree=None, nbspl=10):
        """

        Method to remove correlation in azimuth.

        """

        ### Profiling in Azimuth and elevation
        el_lims = np.linspace(np.min(self.elt) - 0.0001, np.max(self.elt) + 0.0001, n_el + 1)
        el_av = 0.5 * (el_lims[1:] + el_lims[:-1])

        okall = np.abs(self.scantype) > 0
        okpos = self.scantype > 0
        okneg = self.scantype < 0
        oks = [okpos, okneg]
        minaz = np.min(self.azt[okall])
        maxaz = np.max(self.azt[okall])

        ### Use polynomials or spline fitting to remove drifts and large features
        if degree is not None:
            coefficients = np.zeros((2, n_el, degree + 1))
        else:
            coefficients = np.zeros((2, n_el, nbspl))

        for i in range(len(oks)):
            for j in range(n_el):
                ok = oks[i] & (self.elt >= el_lims[j]) & (self.elt < el_lims[j + 1])
                if np.sum(ok) == 0:
                    break
            xc, yc, dx, dy, _ = ft.profile(self.azt[ok], self.tod[ok], rng=[minaz, maxaz], nbins=nbins, mode=True, dispersion=True, plot=False)

            if degree is not None:
                ### Polynomial Fitting
                z = np.polyfit(xc, yc, degree, w=1.0 / dy)
                coefficients[i, j, :] = z
            else:
                ### Spline Fitting
                splfit = MySplineFitting(xc, yc, dy, nbspl)
                coefficients[i, j, :] = splfit.alpha

        ### Now interpolate this to remove it to the data
        nscans = np.max(np.abs(self.scantype))
        for i in range(1, nscans + 1):
            okp = self.scantype == i
            okn = self.scantype == (-i)
        for ok in [okp, okn]:
            the_el = np.median(self.elt[ok])
            if degree is not None:
                myp = np.poly1d([np.interp(the_el, el_av, coefficients[0, :, i]) for i in np.arange(degree + 1)])
                self.tod[ok] -= myp(self.azt[ok])
            else:
                myalpha = [np.interp(the_el, el_av, coefficients[0, :, i]) for i in np.arange(nbspl)]
                self.tod[ok] -= splfit.with_alpha(self.azt[ok], myalpha)

        ### And interpolate for scantype==0 regions
        bad_chunks = self.get_chunks(0)
        self.tod = self.linear_rescale_chunks(bad_chunks, sz=100)
        return self.tod

    def linear_rescale_chunks(self, chunks, sz=1000):
        """ """

        for i in range(len(chunks)):
            thechunk = chunks[i]
            chunklen = thechunk[1] - thechunk[0] + 1
            if thechunk[0] == 0:
                # this is the starting index => just the average
                vals = np.zeros(chunklen) + np.median(self.tod[thechunk[1] + 1 : thechunk[1] + sz]) + np.median(self.tod[thechunk[0] : thechunk[1]])
                self.tod[thechunk[0] : thechunk[1] + 1] -= vals
            elif thechunk[1] == (len(self.tod) - 1):
                # this is the last one => just the average
                vals = np.zeros(chunklen) + np.median(self.tod[thechunk[0] - 1 - sz : thechunk[0] - 1]) + np.median(self.tod[thechunk[0] : thechunk[1]])
                self.tod[thechunk[0] : thechunk[1] + 1] -= vals
            else:
                left = np.median(self.tod[thechunk[0] - 1 - sz : thechunk[0] - 1])
                right = np.median(self.tod[thechunk[1] + 1 : thechunk[1] + sz])
                vals = left + np.linspace(0, 1, chunklen) * (right - left)
                self.tod[thechunk[0] : thechunk[1] + 1] -= np.median(self.tod[thechunk[0] : thechunk[1] + 1]) - vals

        return self.tod

    def get_chunks(self, value):
        """ """

        ### returns chunks corresponding to a given value
        current_chunk = []
        chunk_idx = []
        inchunk = 0
        chunknum = 0
        for i in range(len(self.scantype)):
            if self.scantype[i] == value:
                inchunk = 1
                current_chunk.append(i)
            else:
                if inchunk == 1:
                    chunknum += 1
                    chunk_idx.append([current_chunk[0], current_chunk[len(current_chunk) - 1]])
                    current_chunk = []
                    inchunk = 0
        if inchunk == 1:
            chunk_idx.append([current_chunk[0], current_chunk[len(current_chunk) - 1]])
        return chunk_idx

    def healpix_map_(self, nside=128, countcut=0, unseen_val=hp.UNSEEN):
        """

        Method to project data on the sky using coaddition.

        """

        ips = hp.ang2pix(nside, self.azt, self.elt, lonlat=True)
        mymap = np.zeros(12 * nside**2)
        mapcount = np.zeros(12 * nside**2)
        for i in range(len(self.azt)):
            mymap[ips[i]] -= self.tod[i]
            mapcount[ips[i]] += 1
        unseen = mapcount <= countcut
        mymap[unseen] = unseen_val
        mapcount[unseen] = unseen_val
        mymap[~unseen] = mymap[~unseen] / mapcount[~unseen]
        return mymap

    def run(self, remove_drift=False, remove_offset=False, decorel=False, healpix_map=False):
        """

        Main method to run the pipeline.

        """

        if remove_drift:
            self.tod = self._remove_drifts_spline()
        if remove_offset:
            self.tod = self._remove_offset_scan()
        if decorel:
            self.tod = self.decorel_azel()

        if healpix_map:
            m = self.healpix_map_()
            return self.tod, m
        else:
            return self.tod


class FocalPlane:
    """

    Instance to display data on the FP (assuming Technical Demonstrator)

    """

    def __init__(self, tt, tod, az, el, thk):
        ### Save tod
        self.tt, self.tod = tt, tod
        self.ndets = self.tod.shape[0]
        self.az, self.el = az, el
        self.thk = thk

        ### Useful arguments to plot focal plane
        self.coord_thermo = np.array([17 * 11 + 1, 17 * 12 + 1, 17 * 13 + 1, 17 * 14 + 1, 275, 276, 277, 278])
        self.numbering_tes = np.arange(1, 129, 1)
        x = np.linspace(-0.0504, -0.0024, 17)
        y = np.linspace(-0.0024, -0.0504, 17)

        ### Detector coordinates
        self.X, self.Y = np.meshgrid(x, y)

    def _get_instrument(self):
        """

        Method to initiate QUBIC instrument.

        """

        dictfilename = "global_source_oneDet.dict"
        d = qubicDict()
        d.read_from_file(dictfilename)
        d["synthbeam"] = "CalQubic_Synthbeam_Calibrated_Multifreq_FI.fits"
        q = QubicInstrument(d)
        return q

    def _get_place_ONAFP_with_TES(self, tes, asic):
        """

        Method to compute coordintates for FP plot using TES and ASIC numbering.

        """

        xtes, ytes, FP_index, index_q = scal.TES_Instru2coord(TES=tes, ASIC=asic, q=self._get_instrument(), frame="ONAFP", verbose=False)
        ind = np.where((np.round(xtes, 4) == np.round(self.X, 4)) & (np.round(ytes, 4) == np.round(self.Y, 4)))
        return ind  # ind[0][0]*17+ind[1][0]+1

    def create_subplots(self, tes, asic, ax, **kwargs):
        """

        Method to make many subplots for each TES.

        """

        if np.sum(tes == np.array([4, 36, 68, 100])) != 0:
            i = self.coord_thermo[self.k_thermo]
            self.k_thermo += 1

        else:
            i = self._get_place_ONAFP_with_TES(tes, asic)

            if asic == 2:
                tes += 128
            idx = tes - 1

            ax[i[0][0], i[1][0]].plot(self.tt, self.tod[idx], **kwargs)
            ax[i[0][0], i[1][0]].annotate(
                "{}".format(tes), xy=(0, 0), xycoords="axes fraction", fontsize=9, color="black", fontstyle="italic", fontweight="bold", xytext=(0.05, 0.85), backgroundcolor="w"
            )
        self.bar.update()

    def create_subplots_healpix(self, tes, asic, m, center, reso, ax, fig, **kwargs):
        """

        Method to make many subplots for each TES with healpix projection.

        """

        if np.sum(tes == np.array([4, 36, 68, 100])) != 0:
            i = self.coord_thermo[self.k_thermo]
            self.k_thermo += 1

        else:
            i = self._get_place_ONAFP_with_TES(tes, asic)

            if asic == 2:
                tes += 128

            mproj = hp.gnomview(m, rot=center, reso=reso, title="", notext=True, cbar=False, return_projected_map=True, bgcolor="black", badcolor="gray", cmap="jet")
            ax[i[0][0], i[1][0]].imshow(mproj, origin="lower", cmap="jet", **kwargs)
            ax[i[0][0], i[1][0]].annotate(
                "{}".format(tes), xy=(0, 0), xycoords="axes fraction", fontsize=9, color="black", fontstyle="italic", fontweight="bold", xytext=(0.05, 0.85), backgroundcolor="w"
            )

        self.bar.update()

    def _get_FP_line(self, type_of_data="line", color=None, save=None, center=None, reso=None, nside=None, **kwargs):
        """

        Method to display data in the FP (assuming TD configuration).

        """

        self.bar = progress_bar(256, "Display focal plane")

        if color is None:
            color = ["black"] * self.ndets

        fig, ax = plt.subplots(17, 17, figsize=(30, 30))
        ### Remove all x and y axis
        for i in range(17):
            for j in range(17):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        self.k_thermo = 0

        ### Main loop
        k = 0
        for j in [1, 2]:
            for i in self.numbering_tes:
                # plt.rc('font',size=6)
                if type_of_data == "line":
                    self.create_subplots(i, j, ax, **kwargs)
                elif type_of_data == "healpix":
                    ### Herit some methods
                    pipeline = Pip1Tes(self.tt, self.tod[k], self.az, self.el, self.thk)
                    m = pipeline.healpix_map_(nside=nside)
                    self.create_subplots_healpix(i, j, m, center, reso, ax, fig, **kwargs)

                k += 1

        ### Saving
        if save is not None:
            fig.savefig(save)

        plt.close()


class PipAllTes:
    """

    Instance to analyse and display all TES.

    """

    def __init__(self, tt, tod, az, el, thk):
        self.tt, self.tod = tt, tod
        self.ndets = self.tod.shape[0]
        self.az, self.el = az, el
        self.thk = thk
        self._create_folder("plots")

    def _create_folder(self, folder_name):
        """

        Method to create folder to save plots.

        Parameters :
        ------------
                - folder_name : str

        """

        if not os.path.exists(folder_name):
            try:
                os.makedirs(folder_name)
            except OSError:
                pass
        else:
            pass

    def run(self, remove_drift=False, remove_offset=False, decorel=False, plot_FP=False, plot_FP_healpix=False, center=None, reso=None, color=None, nside=128, **kwargs):
        """

        Method to run loop for all TES. You can add useful arguments using **kwargs keyword.

        Parameters :
        ------------
                - remove_drift    : bool
                - remove_offset   : bool
                - decorel         : bool
                - plot_FP         : bool to plot FP with TOD
                - plot_FP_healpix : bool to plot FP with projected maps
                - center          : tuple
                - reso            : Int number for healpix resolution
                - color           : list of color for each TES
                - nside           : Int number for map pixelization

        """

        for idet in range(self.ndets):
            ### If treatment is needed, it initiate pipeline for a single TES
            if remove_drift or remove_offset or decorel:
                print(f"============ Treating TES #{idet + 1} ============")
                pipeline = Pip1Tes(self.tt, self.tod[idet], self.az, self.el, self.thk)

            if remove_drift:
                self.tod[idet] = pipeline._remove_drifts_spline()

            if remove_offset:
                self.tod[idet] = pipeline._remove_offset_scan()

            if decorel:
                self.tod[idet] = pipeline.decorel_azel()

        ### Initiate FP instance
        fp = FocalPlane(self.tt, self.tod, self.az, self.el, self.thk)

        if plot_FP:
            fp._get_FP_line(type_of_data="line", color=color, save="plots/FP.png")
        if plot_FP_healpix:
            fp._get_FP_line(type_of_data="healpix", color=color, save="plots/FP_healpix.png", center=center, reso=reso, nside=nside, **kwargs)
