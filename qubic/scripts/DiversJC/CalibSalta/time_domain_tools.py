import numpy as np
from pylab import *
import bottleneck as bn
from scipy.signal import medfilt
import scipy.signal
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from scipy import interpolate


from qubicpack.qubicfp import qubicfp
from qubic import fibtools as ft


class MySplineFitting:
    def __init__(self,xin,yin,covarin,nbspl,logspace=False):
        # input parameters
        self.x=xin
        self.y=yin
        self.nbspl=nbspl
        covar=covarin
        if np.size(np.shape(covarin)) == 1:
            err=covarin
            covar=np.zeros((np.size(err),np.size(err)))
            covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
        
        self.covar=covar
        self.invcovar=np.linalg.inv(covar)
        
        # Prepare splines
        xspl=np.linspace(np.min(self.x),np.max(self.x),nbspl)
        if logspace==True: xspl=np.logspace(np.log10(np.min(self.x)),np.log10(np.max(self.x)),nbspl)
        self.xspl=xspl
        F=np.zeros((np.size(xin),nbspl))
        self.F=F
        for i in np.arange(nbspl):
            self.F[:,i]=self.get_spline_tofit(xspl,i,xin)
        
        # solution of the chi square
        ft_cinv_y=np.dot(np.transpose(F),np.dot(self.invcovar,self.y))
        covout=np.linalg.inv(np.dot(np.transpose(F),np.dot(self.invcovar,F)))
        alpha=np.dot(covout,ft_cinv_y)
        fitted=np.dot(F,alpha)
        
        # output
        self.residuals=self.y-fitted
        self.chi2=np.dot(np.transpose(self.residuals), np.dot(self.invcovar, self.residuals))
        self.ndf=np.size(xin)-np.size(alpha)
        self.alpha=alpha
        self.covout=covout
        self.dalpha=np.sqrt(np.diagonal(covout))
    
    def __call__(self,x):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,self.alpha))

    def with_alpha(self,x,alpha):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,alpha))
            
    def get_spline_tofit(self,xspline,index,xx):
        yspline=zeros(np.size(xspline))
        yspline[index]=1.
        tck=interpolate.splrep(xspline,yspline)
        yy=interpolate.splev(xx,tck,der=0)
        return(yy)


def remove_drifts_spline(tt, tod, nsplines=20, nresample=1000, nedge=100, doplot=False):
	### Linear function using the beginning and end of the samples (nn samples)
	### In order to approach periodicity of the signal to be resampled
	nnedge = 100
	x0 = np.mean(tt[:nedge])
	y0 = np.mean(tod[:nedge])
	x1 = np.mean(tt[-nedge:])
	y1 = np.mean(tod[-nedge:])	
	p = np.poly1d(np.array([(y1-y0)/(x1-x0), y0]))
	if doplot:
		subplot(2,1,1)
		plot(tt, tod, label='Input TOD')
		plot(tt, p(tt), label='Linear function from edges')
	### Resample TOD-linearfunction to apply spline
	tt_resample = np.linspace(tt[0], tt[-1], nresample)
	tod_resample = scipy.signal.resample(tod-p(tt), nresample) + p(tt_resample)
	if doplot:
		plot(tt_resample, tod_resample, label='Resampled TOD')

	### Perfomr spline fitting
	splfit = MySplineFitting(tt_resample, tod_resample, tod_resample*0+1, nsplines)
	fitted = splfit(tt_resample)
	if doplot:
		plot(tt_resample, fitted, label='Fitted spline')
		legend()
		subplot(2,1,2)
		plot(tt, tod-splfit(tt))

	return tod-splfit(tt)



def get_mode(y, nbinsmin=51):
    mm, ss = ft.meancut(y, 4)
    hh = np.histogram(y, bins=int(np.min([len(y) / 30, nbinsmin])), range=[mm - 5 * ss, mm + 5 * ss])
    idmax = np.argmax(hh[0])
    mymode = 0.5 * (hh[1][idmax + 1] + hh[1][idmax])
    return mymode


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
	
	### Sampling for HK data
	timesample = np.median(thk[1:]-thk[:-1])
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
	scantype_hk = np.zeros(len(thk), dtype='int')-10
	scantype_hk[c0] =0
	scantype_hk[cpos] = 1
	scantype_hk[cneg] = -1
	# check that we have them all
	count_them = np.sum(scantype_hk==0) + np.sum(scantype_hk==-1) + np.sum(scantype_hk==1)
	if count_them != len(scantype_hk):
		print('Identify_scans: Bad Scan counting at HK sampling level - Error')
		stop
	
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
			
	dead_time = np.sum(c0) / len(thk)
	
	if doplot:
		### Some plotting
		subplot(2,2,1)
		title('Angular Velocity Vs. Azimuth - Dead time = {0:4.1f}%'.format(dead_time*100))
		plot(az, medaz_dt)
		plot(az[c0], medaz_dt[c0], 'ro', label='Slow speed')
		plot(az[cpos], medaz_dt[cpos], '.', label='Scan + : mean={:4.2f}deg/s max={:4.2f}deg/s'.format(np.mean(medaz_dt[cpos]), np.max(medaz_dt[cpos])))
		plot(az[cneg], medaz_dt[cneg], '.', label='Scan - : mean={:4.2f}deg/s max={:4.2f}deg/s'.format(np.mean(medaz_dt[cneg]), np.min(medaz_dt[cneg])))
		xlabel('Azimuth [deg]')
		ylabel('Ang. velocity [deg/s]')
		legend(loc='upper left')

		subplot(2,2,2)
		title('Angular Velocity Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
		plot(thk, medaz_dt)
		plot(thk[c0], medaz_dt[c0], 'ro', label='speed=0')
		plot(thk[cpos], medaz_dt[cpos], '.', label='Scan +')
		plot(thk[cneg], medaz_dt[cneg], '.', label='Scan -')
		legend(loc='upper left')
		xlabel('Time [s]')
		ylabel('Ang. velocity [deg/s]')
		xlim(plotrange[0],plotrange[1])

		subplot(2,3,4)
		title('Azimuth Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
		plot(thk, az)
		plot(thk[c0], az[c0], 'ro', label='speed=0')
		plot(thk[cpos], az[cpos], '.', label='Scan +')
		plot(thk[cneg], az[cneg], '.', label='Scan -')
		legend(loc='upper left')
		xlabel('Time [s]')
		ylabel('Azimuth [deg]')
		xlim(plotrange[0],plotrange[1])

		subplot(2,3,5)
		title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
		plot(thk, el)
		plot(thk[c0], el[c0], 'ro', label='speed=0')
		plot(thk[cpos], el[cpos], '.', label='Scan +')
		plot(thk[cneg], el[cneg], '.', label='Scan -')
		legend(loc='lower left')
		xlabel('Time [s]')
		ylabel('Elevtion [deg]')
		xlim(plotrange[0],plotrange[1])
		elvals = el[(thk > plotrange[0]) & (thk < plotrange[1])]
		deltael = np.max(elvals) - np.min(elvals)
		ylim(np.min(elvals) - deltael/5, np.max(elvals) + deltael/5)
		
		allnums = np.unique(np.abs(scantype_hk))
		for n in allnums[allnums > 0]:
			ok = np.abs(scantype_hk) == n
			xx = np.mean(thk[ok]) 
			yy = np.mean(el[ok])
			if (xx > plotrange[0])  & (xx < plotrange[1]):
				text(xx, yy+deltael/20, str(n))
				
		subplot(2,3,6)
		title('Elevation Vs. time - Dead time = {0:4.1f}%'.format(dead_time*100))
		thecol = (arange(len(allnums))*256/(len(allnums)-1)).astype(int)
		for i in range(len(allnums)):
			ok = np.abs(scantype_hk) == allnums[i]
			plot(az[ok], el[ok], color=get_cmap(rcParams['image.cmap'])(thecol[i]))
		ylim(np.min(el), np.max(el))
		xlabel('Azimuth [deg]')
		ylabel('Elevtion [deg]')
		scatter(-allnums*0, -allnums*0-10,c=allnums)
		aa=colorbar()
		aa.ax.set_ylabel('Scan number')        

		tight_layout()

	if tt is not None:
		### We propagate these at TOD sampling rate  (this is an "step interpolation": we do not want intermediatee values")
		scantype = interp1d(thk, scantype_hk, kind='previous', fill_value='extrapolate')(tt)
		scantype = scantype.astype(int)
		count_them = np.sum(scantype==0) + np.sum(scantype<=-1) + np.sum(scantype>=1)
		if count_them != len(scantype):
			print('Bad Scan counting at data sampling level - Error')
			stop
		### Interpolate azimuth and elevation to TOD sampling
		azt = np.interp(tt, thk, az)
		elt = np.interp(tt, thk, el)
		### Return evereything
		return scantype_hk, azt, elt, scantype
	else:
		### Return scantype at HK sampling only
		return scantype_hk
		

def haar(x, size=51):
	out = np.zeros(x.size)
	xf = bn.move_median(x, size)[size:]   # fen??tre glissante de taille size de la m??diane selon l'axe x
	#xf = medfilt(x, size)[size:] # same but much slower...
	out[size+size//2:-size+size//2] = xf[:-size] - xf[size:]
	return out


def move_median_jc(dd, sz, min_count=1):
	### Avoid the shift in move_median
	return np.nan_to_num(np.roll(bn.move_median(dd, sz, min_count=min_count), -sz//2+1))

def move_std_jc(dd, sz, min_count=1):
	### Avoid the shift in move_median
	return np.nan_to_num(np.roll(bn.move_std(dd, sz, min_count=min_count), -sz//2))


def jumps_finder(dd, size_haar=51, threshold=5, size_fact=3., doplot=False, verbose=False):
	### Haar filter
	hdd = haar(dd, size=size_haar)
	### We remove a 3 times largen median filter version of the hhar filtered data
	### This allows to leave only square jumps of size_haar size corresponding to sharp jumps
	### while larger modulation (example SB peaks) are removed by this operation.
	hdd -= move_median_jc(hdd, int(size_haar*2.5))
	mhdd, shdd = ft.meancut(hdd,3)
	hdd /= shdd

	### Derivative
	dhdd = np.gradient(hdd)
	# dhdd -= move_median_jc(dhdd, size_haar)
	mdhdd, sdhdd = ft.meancut(dhdd,3)
	dhdd /= sdhdd
	dhdd = nan_to_num(dhdd)

	### Sort of moving "median"-std that will discard solitary peaks and be large when there are many n-std points 
	### (typically on synthesized beam peaks)
	truc = bn.move_mean(np.sqrt(np.abs(move_median_jc(dhdd**2, size_haar//2) - move_median_jc(dhdd, size_haar//2)**2)) / sdhdd, size_haar//2)
 
	
	### Thereshold to find jumps 
	# On Derivative of Haar filtered data
	jumps = (np.abs(dhdd) > threshold) & (truc < threshold)
		
	if verbose:
		print('Before DBScan, we have {} points above threshold'.format(np.sum(jumps)))
	### Find clusters in jumps samples in order to separate the jumps
	idx = np.arange(len(dd))
	jumps_tt = idx[jumps]
	clust = DBSCAN(eps=size_haar//5, min_samples=1).fit(np.reshape(jumps_tt, (len(jumps_tt),1)))
	nc = np.max(clust.labels_)+1
	xc = np.zeros(nc+1, dtype=int)
	szc = np.zeros(nc+1, dtype=int)
	for i in range(nc):
		xc[i] = np.min(jumps_tt[clust.labels_ == i])
		szc[i] = (size_fact*(np.max(jumps_tt[clust.labels_ == i])-xc[i])).astype(int)
	### Add last element to clusters
	xc[nc] = idx[-1]
	szc[nc] = 1

	### We remove cluster of zero size
	ok = szc > 0
	xc = xc[ok]
	szc = szc[ok]
	nc = len(xc)
		
	if verbose: print('Detected {} cluster'.format(len(xc)))
	if verbose: print(xc)
	
	### Jumps are only meaningful if they are followed by another one size_haar samples behind
	xj = []
	szj = []
	for i in range(nc-1):
		alldeltas = xc[i+1:]-xc[i]-size_haar
		delta = np.min(np.abs(alldeltas))
		if delta <= (size_haar//5):
			if verbose: 
				print('Jump detected at {} with delta={}'.format(xc[i], delta))
			xj.append(xc[i])
			szj.append(szc[i])
		else:
			if verbose:
				print('**** Jump REJECTED at {} with delta={}'.format(xc[i], delta))
	xj = np.array(xj)
	szj = np.array(szj)
	if verbose: 
		print()
		print('Found {} jumps'.format(len(xj)))
		for i in range(len(xj)):
			print('Initial Jumps are {}: [{}-{}]'.format(i,xj[i],xj[i]+szj[i]))
		
	### It happens that successive jumps are on top of each other because of noise in dhdd, so we remove the next on in that case
	ok = np.ones(len(xj)).astype(bool)
	for i in range(len(xj)-1):
		if ok[i]:
			### difference wrt ulterior jumps
			diff_to_subsequent = xj[i+1:]-(xj[i]+szj[i])
			nums = np.arange(len(diff_to_subsequent))+1
			### if negative then jumps are within the current one
			within = diff_to_subsequent <= 0
			if np.sum(within) > 0:
				if verbose:
					print('We have overlap for xj=[{}-{}]'.format(xj[i], xj[i]+szj[i]))
					print('Overlapping jumps are:')
				for k in nums[within]:
					if verbose: 
						print('xj=[{}-{}]'.format(xj[i+k], xj[i+k]+szj[i+k]))
					ok[i+nums[within]] = False
				szj[i] = np.max(xj[i+k]+szj[i+k]- xj[i])
				if verbose: 
					print('New jump is xj=[{}-{}]'.format(xj[i], xj[i]+szj[i]))
	xj = np.array(xj)[ok]
	szj = np.array(szj)[ok]
	if verbose: 
		print()
		print('Kept {} jumps'.format(len(xj)))
		for i in range(len(xj)):
			print('Final Jumps are {}: [{}-{}]'.format(i,xj[i],xj[i]+szj[i]))
		
			
	if doplot:
		figure()
		subplot(4,1,1)
		title('Jumps Finder: input data')
		xlabel('Index')
		ylabel('Data')
		plot(idx, dd)
		plot(idx[jumps], dd[jumps], 'r.', label='trigger')
		if len(xc)> 0: plot(idx[xc], dd[xc], 'go', label='clusters')
		if len(xj)> 0: 
			plot(idx[xj], dd[xj], 'mo', label='Final Jumps')
			for i in range(len(xj)):
				axvspan(xj[i], xj[i]+szj[i], color='r', alpha=0.5)  
		legend(loc='upper left')
		xlim(0, idx[-1])
		# xlim(276000, 284000)

		
		subplot(4,1,2)
		title('Jumps Finder: Normalized Haar Filtered data')
		xlabel('Index')
		ylabel('H/$\sigma$')            
		plot(idx, hdd)
		plot(idx[jumps], hdd[jumps], 'r.')
		if len(xc)> 0: plot(idx[xc], hdd[xc], 'go')
		if len(xj)> 0: 
			plot(idx[xj], hdd[xj], 'mo')
			for i in range(len(xj)):
				axvspan(xj[i], xj[i]+szj[i], color='r', alpha=0.5) 
		axhline(y=threshold, color='k', ls=':')
		axhline(y=-threshold, color='k', ls=':')
		xlim(0, idx[-1])
		# xlim(276000, 284000)

		   
		subplot(4,1,3)
		title('Jumps Finder: Derivative of Haar Filtered data')
		xlabel('Index')
		ylabel('$\partial$H/$\sigma$')            
		plot(idx, dhdd)
		plot(idx, move_median_jc(dhdd, size_haar//2), label='median')
		plot(idx, move_std_jc(move_median_jc(dhdd, size_haar//2), size_haar*5), label='std(med)')
		plot(idx[jumps], dhdd[jumps], 'r.')
		if len(xc)> 0: plot(idx[xc], dhdd[xc], 'go')
		if len(xj)> 0: 
			plot(idx[xj], dhdd[xj], 'mo')
			for i in range(len(xj)):
				axvspan(xj[i], xj[i]+szj[i], color='r', alpha=0.5)    
		axhline(y=threshold, color='k', ls=':')
		axhline(y=-threshold, color='k', ls=':')
		xlim(0, idx[-1])
		legend()
		# xlim(276000, 284000)

		subplot(4,1,4)
		title('Jumps Finder: Moving "median"-std of Derivative')
		xlabel('Index')
		plot(idx, truc)
		plot(idx[jumps], truc[jumps], 'r.')
		if len(xc)> 0: plot(idx[xc], truc[xc], 'go')
		if len(xj)> 0: 
			plot(idx[xj], truc[xj], 'mo')
			for i in range(len(xj)):
				axvspan(xj[i], xj[i]+szj[i], color='r', alpha=0.5)    
		axhline(y=threshold, color='k', ls=':')
		xlim(0, idx[-1])
		ylim(0, np.max(truc[isfinite(truc)])*1.1)
		legend()
		# xlim(276000, 284000)

		tight_layout()
		
	return list(xj), list(szj)

def correct_jumps_edges(dd, xj, szj, zone_size=100, flag_margin=5, flag_value=2**8, doplot=True):
	doplot = True
	newdd = dd.copy()
	### Removes the median between successive regions
	iinit = -1
	szinit = 0
	flags = np.zeros(len(dd), dtype=int)
	# Add last element to have the full range
	xj.append(len(dd))
	szj.append(0)
	# now do the loop over jumps regions
	for ij, sj in zip(xj, szj):
		myimin = iinit+szinit+1
		#myimax = ij-1
		myimax = myimin + 100
		print(ij, sj)
		print(myimin, myimax)
		print()
		newdd[iinit+1:ij+1] -= np.median(newdd[myimin:myimax])
		# newdd[iinit+1:] -= np.median(newdd[myimin:myimax])
		flags[ij-flag_margin:ij+sj+flag_margin] = flag_value
		iinit = ij
		szinit = sj
	if doplot:
		figure()
		idx = np.arange(len(dd))
		plot(idx, newdd)
		bad = flags != 0
		plot(idx[bad], newdd[bad], 'r.')
		title('Jump correction - correct_jumps_flat()')

	return newdd-np.median(newdd), flags

def correct_jumps_flat(dd, xj, szj, flag_margin=5, flag_value=2**8, doplot=True):
	newdd = dd.copy()
	### Only removes the median of each region in between jumps
	iinit = -1
	szinit = 0
	flags = np.zeros(len(dd), dtype=int)
	# Add last element to have the full range
	xj.append(len(dd))
	szj.append(0)
	# now do the loop over jumps regions
	for ij, sj in zip(xj, szj):
		myimin = iinit+szinit+1
		myimax = myimin + 100
		newdd[iinit+1:ij+1] -= np.median(dd[myimin:myimax])
		flags[ij-flag_margin:ij+sj+flag_margin] = flag_value
		iinit = ij
		szinit = sj
	if doplot:
		figure()
		idx = np.arange(len(dd))
		plot(idx, newdd)
		bad = flags != 0
		plot(idx[bad], newdd[bad], 'r.')
		title('Jump correction - correct_jumps_flat()')

	return newdd-np.median(newdd), flags

def correct_jumps_lin(dd, xj, szj, flag_margin=5, flag_value=2**8, doplot=False):
	doplot = True
	newdd = dd.copy()
	### removes a line in between jumps
	idx = np.arange(len(dd))
	iinit = -1
	szinit = 0
	flags = np.zeros(len(dd), dtype=int)
	# Add last element to have the full range
	xj.append(len(dd))
	szj.append(0)
	# now do the loop over jumps regions
	for ij, sj in zip(xj, szj):
		myimin = np.min([iinit+szinit+1, ij-1])
		myimax = np.max([iinit+szinit+1, ij-1])
		#### Fit a line with those points (removing outliers)
		xx = idx[myimin:myimax]
		yy = newdd[myimin:myimax]
		mm, ss = ft.meancut(yy,3)
		okfit = np.abs(yy-mm) < (4*ss)
		p = np.poly1d(polyfit(xx[okfit], yy[okfit], 1))
		newdd[iinit+1:ij+1] -= p(idx[iinit+1:ij+1])
		flags[ij-flag_margin:ij+sj+flag_margin] = flag_value
		iinit = ij
		szinit = sj
	if doplot:
		figure()
		plot(idx, newdd)
		bad = flags != 0
		plot(idx[bad], newdd[bad], 'r.')
		title('Jump correction - correct_jumps_lin()')
	return newdd-np.median(newdd), flags

def correct_jumps_continuity(dd, xj, szj, flag_margin=5, flag_value=2**8, zone_size=100, doplot=True):
	doplot = True
	newdd = dd.copy()
	flags = np.zeros(len(dd), dtype=int)
	### ensures coninuity after jumps
	# now do the loop over jumps regions
	for ij, sj in zip(xj, szj):
		mm_before, ss_before = ft.meancut(newdd[ij-zone_size:ij-1], 3)
		mm_after, ss_after = ft.meancut(newdd[ij+sj:ij+sj+zone_size+1], 3)
		newdd[ij:] -= -(mm_before-mm_after)
		flags[ij-flag_margin:ij+sj+flag_margin] = flag_value
	if doplot:
		figure()
		idx = np.arange(len(dd))
		plot(idx, newdd)
		bad = flags != 0
		plot(idx[bad], newdd[bad], 'r.')
		title('Jump correction - correct_jumps_continuity()')
	return newdd-np.median(newdd), flags

def correct_jumps_continuity_lin(dd, xj, szj, flag_margin=5, flag_value=2**8, zone_size=100, doplot=True):
	newdd = dd.copy()
	flags = np.zeros(len(dd), dtype=int)
	idx = np.arange(len(dd))
	### ensures coninuity after jumps and also removes a linear function
	# now do the loop over jumps regions
	xj.append(len(dd))
	szj.append(0)

	for i in range(len(xj)-1):
		ij = xj[i]
		sj = szj[i]
		ijp = xj[i+1]
		sjp = szj[i+1]
		mm_before, ss_before = ft.meancut(newdd[ij-zone_size:ij-1], 3)
		mm_beginning, ss_beginning = ft.meancut(newdd[ij+sj:ij+sj+zone_size+1], 3)
		mm_end, ss_end = ft.meancut(newdd[ijp-zone_size-1:ijp-1], 3)
		vals_lin = mm_beginning + (mm_end-mm_beginning) * np.arange(ijp-ij) / (ijp-ij)
		newdd[ij:ijp] -= vals_lin - mm_before
		flags[ij-flag_margin:ij+sj+flag_margin] = flag_value
	if doplot:
		figure()
		plot(idx, dd)
		plot(idx, newdd)
		bad = flags != 0
		plot(idx[bad], newdd[bad], 'r.')
		title('Jump correction - correct_jumps_continuity_lin()')
	return newdd-np.median(newdd), flags


def fill_bad_regions(dd, flags, flaglimit=1, zone_size=30, fill_baselines='linear', noisetype='normal'):
	idx = np.arange(len(dd))
	newdd = dd.copy()
	badidx = idx[flags >= flaglimit]
	clust = DBSCAN(eps=3, min_samples=2).fit(np.reshape(badidx, (len(badidx),1)))
	nc = np.max(clust.labels_)+1
	for i in range(nc):
		imin = np.min(badidx[clust.labels_==i])
		imax = np.max(badidx[clust.labels_==i])
		mm_before, ss_before = ft.meancut(newdd[imin-zone_size:imin-1], 3)
		mm_after, ss_after = ft.meancut(newdd[imax:imax+zone_size+1], 3)
		### Fabricate constrained noise realiztion
		if noisetype == 'normal':
			mynoise = np.random.randn(imax-imin+1)*(ss_before+ss_after)/2

		if fill_baselines == 'linear':
			### We fill the missing data with a linear interpolation between average before and after
			filler = mm_before + (mm_after-mm_before) / (imax-imin) * np.arange(imax-imin+1)
			newdd[imin:imax+1] =  filler + mynoise
		elif fill_baselines == 'equalize':
			### We equalize the levels before and after, modifying the baseline up to the end of the TOD each time
			newdd[imin:] -= (mm_after-mm_before)
			newdd[imin:imax+1] = mm_before + mynoise

	return newdd

def jumps_correction(dd, threshold=5, size_haar=51, doplot=False, verbose=False, method='lin'):
	### Find the jumps
	xjumps, szjumps = jumps_finder(dd, threshold=threshold, size_haar=size_haar, doplot=doplot, verbose=verbose)

	### Correct for the jumps
	if method == 'flat':
		newdd, flags = correct_jumps_flat(dd, xjumps, szjumps, doplot=doplot)
	if method == 'edges':
		newdd, flags = correct_jumps_edges(dd, xjumps, szjumps, doplot=doplot)
	elif method == 'lin':
		newdd, flags = correct_jumps_lin(dd, xjumps, szjumps, doplot=doplot)
	elif method == 'continuity':
		newdd, flags = correct_jumps_continuity(dd, xjumps, szjumps, doplot=doplot)
	elif method == 'continuity_lin':
		newdd, flags = correct_jumps_continuity_lin(dd, xjumps, szjumps, doplot=doplot)
		# newdd, flags = correct_jumps_continuity(dd, xjumps, szjumps, doplot=doplot)
		# newdd, flags = correct_jumps_lin(dd, xjumps, szjumps, doplot=doplot)

	### Fill bad regions with constrained random noise realization
	if ((flags > 0).sum()) != 0:
		newdd = fill_bad_regions(newdd, flags)

	### Return corrected TOD
	return newdd, flags



