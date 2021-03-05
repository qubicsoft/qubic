"""
This script can be called after run angular_resolution.py as it is if you want to display the measured fwhm 
by FitMethod or SigmaMetod (see below 'methodused' parameter)

"""

import numpy as np
import matplotlib.pyplot as mp
from matplotlib import gridspec
#To use latex in matplotlib
from matplotlib import rc
import sys
from matplotlib import gridspec
from scipy.interpolate import interp1d
from resolution import *

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
# Prepare dictionary to decently run (because RAM memory and time)
d = qubic.qubicdict.qubicDict()
#d.read_from_file('test_resolution.dict')
d.read_from_file(sys.argv[1])

###			All 	 in 	plot 		###
###			All 	 in 	plot 		###
###			All freq in one plot 		### 
###			All 	 in 	plot 		###
###			All 	 in 	plot 		###

methodused = 'sigma'
#name = NameRun(d)
name = '20191011testing'
bla = np.arange(2,8)
nu = 150
if nu == 150:
    nus3, _fwhm_3, fwhm_m3 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[0],methodused), unpack = True)
    nus4, _fwhm_4, fwhm_m4 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[1],methodused), unpack = True)
    nus5, _fwhm_5, fwhm_m5 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[2],methodused), unpack = True)
    nus6, _fwhm_6, fwhm_m6 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[3],methodused), unpack = True)
    nus7, _fwhm_7, fwhm_m7 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[4],methodused), unpack = True)
    nus8, _fwhm_8, fwhm_m8 = np.loadtxt(name+'-nf16nrec{}-{}.txt'.format(bla[5],methodused), unpack = True)
    nus_in = np.array([132.28891971, 134.38320643, 136.51064814, 138.6717697 ,
        140.86710433, 143.09719364, 145.36258785, 147.66384587,
        150.00153547, 152.3762334 , 154.78852555, 157.23900707,
        159.72828256, 162.25696615, 164.82568174, 167.43506306])
    xpos = 163.
    ypos = 0.42

if nu == 220:
    nus3, _fwhm_3, fwhm_m3 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec3-{}.txt'.format(methodused), unpack = True)
    nus4, _fwhm_4, fwhm_m4 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec4-{}.txt'.format(methodused), unpack = True)
    nus5, _fwhm_5, fwhm_m5 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec5-{}.txt'.format(methodused), unpack = True)
    nus6, _fwhm_6, fwhm_m6 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec6-{}.txt'.format(methodused), unpack = True)
    nus7, _fwhm_7, fwhm_m7 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec7-{}.txt'.format(methodused), unpack = True)
    nus8, _fwhm_8, fwhm_m8 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec8-{}.txt'.format(methodused), unpack = True)
    nus_in = np.array([194.02374891, 197.09536943, 200.21561727, 203.38526223,
       206.60508635, 209.87588401, 213.19846218, 216.57364061,
       220.00225202, 223.48514232, 227.0231708 , 230.61721037,
       234.26814775, 237.97688369, 241.74433321, 245.57142582])
    xpos = 240.
    ypos = 0.29

if methodused == 'fit':
    calibname = NameCalib(method = 'fit')
    interpNusSig, interpDeltaFwhmSig, interpStdSig = np.loadtxt(calibname, unpack = True)#'20190111_{}Calib.txt'.format(methodused), unpack=True)
elif methodused == 'sigma':
    #calibname = NameCalib(method = 'sigma')
    calibname = '20191007_sigmacalibration1-5-256.txt'
    interpNusSig, interpDeltaFwhmSig, interpStdSig = np.loadtxt(calibname, unpack = True)#'20190111_{}Calib.txt'.format(methodused), unpack=True)

f_fit = interp1d(interpNusSig, interpDeltaFwhmSig, kind = 'cubic')
f_err = interp1d(interpNusSig, interpStdSig, kind = 'cubic')

delta3 = f_fit(nus3)
delta4 = f_fit(nus4)
delta5 = f_fit(nus5)
delta6 = f_fit(nus6)
delta7 = f_fit(nus7)
delta8 = f_fit(nus8)

fig = mp.figure(figsize=(12, 8)) 

ymin = np.min([np.min((fwhm_m3 - delta3 - _fwhm_3)/_fwhm_3),
               np.min((fwhm_m4 - delta4 - _fwhm_4)/_fwhm_4),
               np.min((fwhm_m5 - delta5 - _fwhm_5)/_fwhm_5),
               np.min((fwhm_m6 - delta6 - _fwhm_6)/_fwhm_6),
               np.min((fwhm_m6 - delta6 - _fwhm_6)/_fwhm_6),
               np.min((fwhm_m7 - delta7 - _fwhm_7)/_fwhm_7),
               np.min((fwhm_m8 - delta8 - _fwhm_8)/_fwhm_8)])#-0.02 
ymax = np.max([np.max((fwhm_m3 - delta3 - _fwhm_3)/_fwhm_3),
               np.max((fwhm_m4 - delta4 - _fwhm_4)/_fwhm_4),
               np.max((fwhm_m5 - delta5 - _fwhm_5)/_fwhm_5),
               np.max((fwhm_m6 - delta6 - _fwhm_6)/_fwhm_6),
               np.max((fwhm_m6 - delta6 - _fwhm_6)/_fwhm_6),
               np.max((fwhm_m7 - delta7 - _fwhm_7)/_fwhm_7),
               np.max((fwhm_m8 - delta8 - _fwhm_8)/_fwhm_8)])#0.02
ymin=1.2*ymin
ymax =1.2*ymax

grid = mp.GridSpec(9, 2)
if methodused == 'fit':
    frmat_ = 'rs'
    frmat = 'rs--'
elif methodused == 'sigma':
    frmat_ = 'bs'
    frmat = 'bs--'

#fig.suptitle('{} method (band {}GHz)'.format(methodused, nu))
ax0 = mp.subplot(grid[0:2,0])
ax0.set_ylabel(r'FWHM [deg]')
ax0.plot(nus3,_fwhm_3,'ko', label = r'FWHM real')
ax0.plot(nus3,fwhm_m3 - delta3, frmat, label = r'FWHM {}'.format(methodused))
ax0.legend(loc = 'lower left')
ax0.text(xpos,ypos, r'N$_{\rm rec} =$ 1', size=12, ha="right")

ax00 = mp.subplot(grid[2:3,0])
ax00.set_ylabel(r'Diff.')
ax00.set_ylim([ymin,ymax])
ax00.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax00.errorbar(nus3,(fwhm_m3 - delta3 - _fwhm_3)/_fwhm_3, yerr = f_err(nus3), fmt=frmat)

ax1 = mp.subplot(grid[0:2,1])
ax1.plot(nus4,_fwhm_4,'ko')
ax1.plot(nus4,fwhm_m4 - delta4, frmat)
ax1.text(xpos,ypos, r'N$_{\rm rec} =$ 2', size=12, ha="right")

ax01 = mp.subplot(grid[2:3,1])
ax01.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax01.set_ylim([ymin,ymax])
ax01.errorbar(nus4, (fwhm_m4 - delta4 - _fwhm_4)/_fwhm_4, yerr = f_err(nus4), fmt= frmat)

ax2 = mp.subplot(grid[3:5,0])
ax2.set_ylabel(r'FWHM [deg]')
ax2.plot(nus5,_fwhm_5,'ko')
ax2.plot(nus5,fwhm_m5 - delta5, frmat)
ax2.text(xpos,ypos, r'N$_{\rm rec} = $3', size=12, ha="right")

ax02 = mp.subplot(grid[5:6,0])
ax02.set_ylabel(r'Diff.')
ax02.set_ylim([ymin,ymax])
ax02.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax02.errorbar(nus5,(fwhm_m5 - delta5 - _fwhm_5)/_fwhm_5, yerr = f_err(nus5), fmt= frmat)

ax3 = mp.subplot(grid[3:5,1])
ax3.plot(nus6,_fwhm_6,'ko')
ax3.plot(nus6,fwhm_m6 - delta6, frmat)
ax3.text(xpos,ypos, r'N$_{\rm rec} = $4', size=12, ha="right")

ax03 = mp.subplot(grid[5:6,1])
ax03.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax03.set_ylim([ymin,ymax])
ax03.errorbar(nus6, (fwhm_m6 - delta6 - _fwhm_6)/_fwhm_6, yerr = f_err(nus6), fmt= frmat)

ax4 = mp.subplot(grid[6:8,0])
ax4.set_ylabel(r'FWHM [deg]')
ax4.plot(nus7,_fwhm_7,'ko')
ax4.plot(nus7,fwhm_m7 - delta7, frmat)
ax4.text(xpos,ypos, r'N$_{\rm rec} = $5', size=12, ha="right")

ax04 = mp.subplot(grid[8:9,0])
ax04.set_xlabel(r'$\nu$[GHz]')
ax04.set_ylabel(r'Diff.')
ax04.set_ylim([ymin,ymax])
ax04.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax04.errorbar(nus7,(fwhm_m7 - delta7 - _fwhm_7)/_fwhm_7, yerr = f_err(nus7), fmt= frmat)

ax5 = mp.subplot(grid[6:8,1])
ax5.plot(nus8,_fwhm_8,'ko')
ax5.plot(nus8,fwhm_m8 - delta8, frmat)
ax5.text(xpos,ypos, r'N$_{\rm rec} = $6', size=12, ha="right")

ax05 = mp.subplot(grid[8:9,1])
ax05.set_xlabel(r'$\nu$[GHz]')
ax05.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax05.set_ylim([ymin,ymax])
ax05.errorbar(nus8,(fwhm_m8 - delta8 - _fwhm_8)/_fwhm_8, yerr = f_err(nus8), fmt= frmat)

ax0.text(xpos,ypos, r'N$_{\rm rec} = 3$', size=12, ha="right")

alph = 0.2
ls=':'
for i in nus_in:
    ax0.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax1.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax2.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax3.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax4.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax5.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax00.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax01.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax02.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax03.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax04.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax05.axvline(x= i, color='k', linestyle=ls, alpha = alph)

ax1.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax2.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax3.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax4.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax5.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax0.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])

ax01.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax02.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax00.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax03.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax04.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])
ax05.set_xlim([nus_in[0]*0.99, nus_in[-1]*1.0])

ax1.set_yticklabels([])
ax3.set_yticklabels([])
ax5.set_yticklabels([])
ax01.set_yticklabels([])
ax03.set_yticklabels([])
ax05.set_yticklabels([])

ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax02.set_xticklabels([])
ax03.set_xticklabels([])

#ax0.legend(bbox_to_anchor=(0.,1.02,1,0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=3)
mp.subplots_adjust(hspace=0.,wspace=0.)
mp.show()


todo = False
if todo:
    print('siguee')
elif not todo:
    sys.exit()
#
#
#
#
#
#
#
#

methodused = 'sigma'
nu2 = 220
nu = 150
if nu == 150:
    nus3_150, _fwhm_3_150, fwhm_m3_150 = np.loadtxt('../20190207-nf16nrec3-{}.txt'.format(methodused), unpack = True)
    nus4_150, _fwhm_4_150, fwhm_m4_150 = np.loadtxt('../20190207-nf16nrec4-{}.txt'.format(methodused), unpack = True)
    nus5_150, _fwhm_5_150, fwhm_m5_150 = np.loadtxt('../20190207-nf16nrec5-{}.txt'.format(methodused), unpack = True)
    nus6_150, _fwhm_6_150, fwhm_m6_150 = np.loadtxt('../20190207-nf16nrec6-{}.txt'.format(methodused), unpack = True)
    nus7_150, _fwhm_7_150, fwhm_m7_150 = np.loadtxt('../20190207-nf16nrec7-{}.txt'.format(methodused), unpack = True)
    nus8_150, _fwhm_8_150, fwhm_m8_150 = np.loadtxt('../20190207-nf16nrec8-{}.txt'.format(methodused), unpack = True)
    nus_in_150 = np.array([132.28891971, 134.38320643, 136.51064814, 138.6717697 ,
        140.86710433, 143.09719364, 145.36258785, 147.66384587,
        150.00153547, 152.3762334 , 154.78852555, 157.23900707,
        159.72828256, 162.25696615, 164.82568174, 167.43506306])
    xpos_150 = 163.
    ypos_150 = 0.42

if nu2 == 220:
    nus3_220, _fwhm_3_220, fwhm_m3_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec3-{}.txt'.format(methodused), unpack = True)
    nus4_220, _fwhm_4_220, fwhm_m4_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec4-{}.txt'.format(methodused), unpack = True)
    nus5_220, _fwhm_5_220, fwhm_m5_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec5-{}.txt'.format(methodused), unpack = True)
    nus6_220, _fwhm_6_220, fwhm_m6_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec6-{}.txt'.format(methodused), unpack = True)
    nus7_220, _fwhm_7_220, fwhm_m7_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec7-{}.txt'.format(methodused), unpack = True)
    nus8_220, _fwhm_8_220, fwhm_m8_220 = np.loadtxt('../result_20190213/20190213_00_220-nf16nrec8-{}.txt'.format(methodused), unpack = True)
    nus_in_220 = np.array([194.02374891, 197.09536943, 200.21561727, 203.38526223,
       206.60508635, 209.87588401, 213.19846218, 216.57364061,
       220.00225202, 223.48514232, 227.0231708 , 230.61721037,
       234.26814775, 237.97688369, 241.74433321, 245.57142582])
    xpos_220 = 240.
    ypos_220 = 0.29

interpNusSig, interpDeltaFwhmSig, interpStdSig = np.loadtxt('../20190111_{}Calib.txt'.format(methodused), unpack=True)
f_fit = interp1d(interpNusSig, interpDeltaFwhmSig, kind = 'cubic')
f_err = interp1d(interpNusSig, interpStdSig, kind = 'cubic')

delta3_150 = f_fit(nus3_150)
delta4_150 = f_fit(nus4_150)
delta5_150 = f_fit(nus5_150)
delta6_150 = f_fit(nus6_150)
delta7_150 = f_fit(nus7_150)
delta8_150 = f_fit(nus8_150)

delta3_220 = f_fit(nus3_220)
delta4_220 = f_fit(nus4_220)
delta5_220 = f_fit(nus5_220)
delta6_220 = f_fit(nus6_220)
delta7_220 = f_fit(nus7_220)
delta8_220 = f_fit(nus8_220)

fig = mp.figure(figsize=(12, 8)) 

ymin_150 = np.min([np.min((fwhm_m3_150 - delta3_150 - _fwhm_3_150)/_fwhm_3_150),
               np.min((fwhm_m4_150 - delta4_150 - _fwhm_4_150)/_fwhm_4_150),
               np.min((fwhm_m5_150 - delta5_150 - _fwhm_5_150)/_fwhm_5_150),
               np.min((fwhm_m6_150 - delta6_150 - _fwhm_6_150)/_fwhm_6_150),
               np.min((fwhm_m6_150 - delta6_150 - _fwhm_6_150)/_fwhm_6_150),
               np.min((fwhm_m7_150 - delta7_150 - _fwhm_7_150)/_fwhm_7_150),
               np.min((fwhm_m8_150 - delta8_150 - _fwhm_8_150)/_fwhm_8_150)])#-0.02 
ymax_150 = np.max([np.max((fwhm_m3_150 - delta3_150 - _fwhm_3_150)/_fwhm_3_150),
               np.max((fwhm_m4_150 - delta4_150 - _fwhm_4_150)/_fwhm_4_150),
               np.max((fwhm_m5_150 - delta5_150 - _fwhm_5_150)/_fwhm_5_150),
               np.max((fwhm_m6_150 - delta6_150 - _fwhm_6_150)/_fwhm_6_150),
               np.max((fwhm_m6_150 - delta6_150 - _fwhm_6_150)/_fwhm_6_150),
               np.max((fwhm_m7_150 - delta7_150 - _fwhm_7_150)/_fwhm_7_150),
               np.max((fwhm_m8_150 - delta8_150 - _fwhm_8_150)/_fwhm_8_150)])#0.02

ymin_220 = np.min([np.min((fwhm_m3_220 - delta3_220 - _fwhm_3_220)/_fwhm_3_220),
               np.min((fwhm_m4_220 - delta4_220 - _fwhm_4_220)/_fwhm_4_220),
               np.min((fwhm_m5_220 - delta5_220 - _fwhm_5_220)/_fwhm_5_220),
               np.min((fwhm_m6_220 - delta6_220 - _fwhm_6_220)/_fwhm_6_220),
               np.min((fwhm_m6_220 - delta6_220 - _fwhm_6_220)/_fwhm_6_220),
               np.min((fwhm_m7_220 - delta7_220 - _fwhm_7_220)/_fwhm_7_220),
               np.min((fwhm_m8_220 - delta8_220 - _fwhm_8_220)/_fwhm_8_220)])#-0.02 
ymax_220 = np.max([np.max((fwhm_m3_220 - delta3_220 - _fwhm_3_220)/_fwhm_3_220),
               np.max((fwhm_m4_220 - delta4_220 - _fwhm_4_220)/_fwhm_4_220),
               np.max((fwhm_m5_220 - delta5_220 - _fwhm_5_220)/_fwhm_5_220),
               np.max((fwhm_m6_220 - delta6_220 - _fwhm_6_220)/_fwhm_6_220),
               np.max((fwhm_m6_220 - delta6_220 - _fwhm_6_220)/_fwhm_6_220),
               np.max((fwhm_m7_220 - delta7_220 - _fwhm_7_220)/_fwhm_7_220),
               np.max((fwhm_m8_220 - delta8_220 - _fwhm_8_220)/_fwhm_8_220)])#0.02



ymin=1.2*np.min([ymin_150,ymin_220])

ymax =1.2*np.max([ymax_220,ymax_220])

grid = mp.GridSpec(9, 2)
if methodused == 'fit':
    frmat_ = 'rs'
    frmat = 'rs--'
elif methodused == 'sigma':
    frmat_ = 'bs'
    frmat = 'bs--'

fig.suptitle('{} method (bands {}GHz + {}GHz)'.format(methodused, nu, nu2))
ax0 = mp.subplot(grid[0:2,0])
ax0.set_ylabel(r'FWHM [deg]')
ax0.plot(nus3_150,_fwhm_3_150,'ko')
ax0.plot(nus3_150,fwhm_m3_150 - delta3_150, frmat)
ax0.plot(nus3_220,_fwhm_3_220,'ko', label = r'FWHM real')
ax0.plot(nus3_220,fwhm_m3_220 - delta3_220, frmat, label = r'FWHM')
ax0.legend(loc = 'lower left')
ax0.text(xpos,ypos, r'N$_{\rm rec} = 3$', size=12, ha="right")

ax00 = mp.subplot(grid[2:3,0])
ax00.set_ylabel(r'Diff.')
ax00.set_ylim([ymin,ymax])
ax00.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax00.errorbar(nus3_150,(fwhm_m3_150 - delta3_150 - _fwhm_3_150)/_fwhm_3_150, yerr = f_err(nus3_150), fmt=frmat)
ax00.errorbar(nus3_220,(fwhm_m3_220 - delta3_220 - _fwhm_3_220)/_fwhm_3_220, yerr = f_err(nus3_220), fmt=frmat)

ax1 = mp.subplot(grid[0:2,1])
ax1.plot(nus4_150,_fwhm_4_150,'ko')
ax1.plot(nus4_150,fwhm_m4_150 - delta4_150, frmat)
ax1.plot(nus4_220,_fwhm_4_220,'ko')
ax1.plot(nus4_220,fwhm_m4_220 - delta4_220, frmat)
ax1.text(xpos,ypos, r'N$_{\rm rec} = 4$', size=12, ha="right")

ax01 = mp.subplot(grid[2:3,1])
ax01.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax01.set_ylim([ymin,ymax])
ax01.errorbar(nus4_150, (fwhm_m4_150 - delta4_150 - _fwhm_4_150)/_fwhm_4_150, yerr = f_err(nus4_150), fmt= frmat)
ax01.errorbar(nus4_220, (fwhm_m4_220 - delta4_220 - _fwhm_4_220)/_fwhm_4_220, yerr = f_err(nus4_220), fmt= frmat)


ax2 = mp.subplot(grid[3:5,0])
ax2.set_ylabel(r'FWHM [deg]')
ax2.plot(nus5_150,_fwhm_5_150,'ko')
ax2.plot(nus5_150,fwhm_m5_150 - delta5_150, frmat)
ax2.plot(nus5_220,_fwhm_5_220,'ko')
ax2.plot(nus5_220,fwhm_m5_220 - delta5_220, frmat)
ax2.text(xpos,ypos, r'N$_{\rm rec} = 5$', size=12, ha="right")

ax02 = mp.subplot(grid[5:6,0])
ax02.set_ylabel(r'Diff.')
ax02.set_ylim([ymin,ymax])
ax02.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax02.errorbar(nus5_150,(fwhm_m5_150 - delta5_150 - _fwhm_5_150)/_fwhm_5_150, yerr = f_err(nus5_150), fmt= frmat)
ax02.errorbar(nus5_220,(fwhm_m5_220 - delta5_220 - _fwhm_5_220)/_fwhm_5_220, yerr = f_err(nus5_220), fmt= frmat)

ax3 = mp.subplot(grid[3:5,1])
ax3.plot(nus6_150,_fwhm_6_150,'ko')
ax3.plot(nus6_150,fwhm_m6_150 - delta6_150, frmat)
ax3.plot(nus6_220,_fwhm_6_220,'ko')
ax3.plot(nus6_220,fwhm_m6_220 - delta6_220, frmat)
ax3.text(xpos,ypos, r'N$_{\rm rec} = 6$', size=12, ha="right")

ax03 = mp.subplot(grid[5:6,1])
ax03.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax03.set_ylim([ymin,ymax])
ax03.errorbar(nus6_150, (fwhm_m6_150 - delta6_150 - _fwhm_6_150)/_fwhm_6_150, yerr = f_err(nus6_150), fmt= frmat)
ax03.errorbar(nus6_220, (fwhm_m6_220 - delta6_220 - _fwhm_6_220)/_fwhm_6_220, yerr = f_err(nus6_220), fmt= frmat)

ax4 = mp.subplot(grid[6:8,0])
ax4.set_ylabel(r'FWHM [deg]')
ax4.plot(nus7_150,_fwhm_7_150,'ko')
ax4.plot(nus7_150,fwhm_m7_150 - delta7_150, frmat)
ax4.plot(nus7_220,_fwhm_7_220,'ko')
ax4.plot(nus7_220,fwhm_m7_220 - delta7_220, frmat)
ax4.text(xpos,ypos, r'N$_{\rm rec} = 7$', size=12, ha="right")

ax04 = mp.subplot(grid[8:9,0])
ax04.set_xlabel(r'$\nu$[GHz]')
ax04.set_ylabel(r'Diff.')
ax04.set_ylim([ymin,ymax])
ax04.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax04.errorbar(nus7_150,(fwhm_m7_150 - delta7_150 - _fwhm_7_150)/_fwhm_7_150, yerr = f_err(nus7_150), fmt= frmat)
ax04.errorbar(nus7_220,(fwhm_m7_220 - delta7_220 - _fwhm_7_220)/_fwhm_7_220, yerr = f_err(nus7_220), fmt= frmat)

ax5 = mp.subplot(grid[6:8,1])
ax5.plot(nus8_150,_fwhm_8_150,'ko')
ax5.plot(nus8_150,fwhm_m8_150 - delta8_150, frmat)
ax5.plot(nus8_220,_fwhm_8_220,'ko')
ax5.plot(nus8_220,fwhm_m8_220 - delta8_220, frmat)
ax5.text(xpos,ypos, r'N$_{\rm rec} = 8$', size=12, ha="right")

ax05 = mp.subplot(grid[8:9,1])
ax05.set_xlabel(r'$\nu$[GHz]')
ax05.axhline(y= 0., color='k', linestyle=':', alpha = 0.6)
ax05.set_ylim([ymin,ymax])
ax05.errorbar(nus8_150,(fwhm_m8_150 - delta8_150 - _fwhm_8_150)/_fwhm_8_150, yerr = f_err(nus8_150), fmt= frmat)
ax05.errorbar(nus8_220,(fwhm_m8_220 - delta8_220 - _fwhm_8_220)/_fwhm_8_220, yerr = f_err(nus8_220), fmt= frmat)
ax0.text(xpos,ypos, r'N$_{\rm rec} = 3$', size=12, ha="right")

alph = 0.2
ls=':'
for i in nus_in_150:
    ax0.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax1.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax2.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax3.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax4.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax5.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax00.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax01.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax02.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax03.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax04.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax05.axvline(x= i, color='k', linestyle=ls, alpha = alph)
for i in nus_in_220:
    ax0.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax1.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax2.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax3.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax4.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax5.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax00.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax01.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax02.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax03.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax04.axvline(x= i, color='k', linestyle=ls, alpha = alph)
    ax05.axvline(x= i, color='k', linestyle=ls, alpha = alph)

ax1.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax2.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax3.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax4.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax5.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax0.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])

ax01.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax02.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax00.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax03.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax04.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])
ax05.set_xlim([nus_in_150[0]*0.99, nus_in_220[-1]*1.0])

ax1.set_yticklabels([])
ax3.set_yticklabels([])
ax5.set_yticklabels([])
ax01.set_yticklabels([])
ax03.set_yticklabels([])
ax05.set_yticklabels([])

ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])
ax00.set_xticklabels([])
ax01.set_xticklabels([])
ax02.set_xticklabels([])
ax03.set_xticklabels([])

#ax0.legend(bbox_to_anchor=(0.,1.02,1,0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=3)
mp.subplots_adjust(hspace=0.,wspace=0.)
mp.show()




