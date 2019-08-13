from __future__ import division, print_function
from matplotlib import rc
from qubicpack import qubicpack as qp
import fibtools as ft
import plotters as p
import lin_lib as ll
import demodulation_lib as dl
import qubic.io

from pysimulators import FitsArray

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
import string
import scipy.signal as scsig
from scipy import interpolate
import os
import healpy as hp
from matplotlib.backends.backend_pdf import PdfPages


method = sys.argv[1]
dirfiles = sys.argv[2]
names = sys.argv[3]
freq = sys.argv[4]



#### Find files
nu = int(str.split(freq,'-')[0][0:3])
allfiles = np.sort(glob.glob(dirfiles+'alltes*_'+names+'_*.fits'))
allfiles = np.sort(allfiles)

allfiles_ang = np.sort(glob.glob(dirfiles+'angles*_'+names+'_*.fits'))
allfiles_ang = np.sort(allfiles_ang)

allels = np.zeros(len(allfiles))
for i in range(len(allfiles)):
    allels[i] = str.split(allfiles[i],'_')[-1][:-5]

print 'Found {} Files'.format(len(allfiles))


#### Read files
alldata = []
allaz = []
allel = []
allang_az = []
for j in range(len(allfiles)):
    data = np.array(FitsArray(allfiles[j]))
    sh = np.shape(data)
    alldata.append((data.T-np.median(data,axis=1)).T)
    bla = np.array(FitsArray(allfiles_ang[j]))
    allaz.append(bla[0,:])
    allel.append(bla[1,:]+124.35)
    allang_az.append(bla[2,:])

#### Make a fake TOD for healpix mapping
tod = dl.make_tod(alldata)
az = dl.make_tod(allaz, axis=0)
el = dl.make_tod(allel, axis=0)


#### Make directories
dir_files = '/Volumes/Data/Qubic/Calib-TD/Synthesized_Beams_Files/' + freq
dir_img = '/Volumes/Data/Qubic/Calib-TD/Synthesized_Beams_Images/' + freq
dir_healpix = dir_files + '/Healpix'
dir_flat = dir_files + '/Flat'
dir_img_healpix = dir_img + '/Healpix'
dir_img_flat = dir_img + '/Flat'
dirs = [dir_files, dir_healpix, dir_flat, dir_img, dir_img_flat, dir_img_healpix]
for d in dirs:
    try: 
        os.mkdir(d)
    except:
        print ''


#### Make images for each TES
nside = 256
nbins_x = 40
x_max = np.max(az)+0.5
x_min = -x_max
nsig_lo = 3
nsig_hi = 10



#### Now make the all in a page files

## First for Flat
nn1 = 6  ## Vertical
nn2 = 4  ## Horizonthal
ic=0
nseries = 256/(nn1*nn2)+1
#nseries = 1
fs = 10
#rc('figure',figsize=(20,28))
rcParams.update({'font.size': fs})

with PdfPages(dir_img+'/allTES_flat.pdf') as pdf:
    for serie in range(nseries):
        print 'Doing Flat all-in-one image: page {} out of {}'.format(serie, 256/(nn1*nn2)+1)
        rc('figure',figsize=(20,28))
        for i in range(nn1*nn2):
            TESNum = serie*nn1*nn2+i+1
            TESIndex = TESNum-1
            if TESNum <= 256:
			subplot(nn1,nn2,i+1)
			img, xx, yy = dl.bin_image_elscans(allaz, allels, alldata, [x_min, x_max], nbins_x, TESIndex)
			mm, ss = ft.meancut(img[img != 0], 3)
			the_img = img.copy()
			the_img[img < (-ss*nsig_lo)] = -ss*nsig_lo
			imshow(the_img, 
			       extent=[x_min*np.cos(np.radians(50)),x_max*np.cos(np.radians(50)), np.min(allels), np.max(allels)], 
			       aspect='equal',vmin=-ss*nsig_lo, vmax=ss*nsig_hi)
			title('Nu={} GHz - TESNum={}'.format(nu,TESIndex+1), fontsize=fs)
			ylabel('Elevation', fontsize=fs)
			xlabel('Azimuth x cos(50)', fontsize=fs)    
        tight_layout()        
        pdf.savefig()
    close()




## Then for Healpix
nn1 = 6  ## Vertical
nn2 = 4  ## Horizonthal
ic=0
nseries = 256/(nn1*nn2)+1
#nseries = 1
fs = 9
rcParams.update({'font.size': fs})
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(dir_img+'/allTES_Healpix.pdf') as pdf:
    for serie in range(nseries):
        print 'Doing Healpix all-in-one image: page {} out of {}'.format(serie, 256/(nn1*nn2)+1)
        rc('figure',figsize=(16,28))
        clf()
        for i in range(nn1*nn2):
            TESNum = serie*nn1*nn2+i+1
            TESIndex = TESNum-1
            if TESNum <= 256:
				ok = np.abs(az) <= x_max 
				newtod = tod.copy()
				newtod[:, ~ok] = 0
				sbmap = dl.scan2hpmap(nside, az*np.cos(np.radians(50)), el-50, newtod[TESIndex,:])
				the_sbmap = sbmap.copy()
				mm, ss = ft.meancut(sbmap[sbmap != 0], 3)
				the_sbmap[sbmap < (-ss*nsig_lo)] = -ss*nsig_lo
				hp.gnomview(sbmap, rot=[0,np.mean(allels)-50], 
				            reso=10,min=-ss*nsig_lo, max=+ss*nsig_hi,
				            title='Nu={} GHz - TESNum={}'.format(nu,TESIndex+1), sub=(nn1,nn2,i+1))      
        tight_layout()        
        pdf.savefig()
    close()


    
for TESIndex in range(256):
    print 'Doing individual images (Flat and Healpix) for TES {} out of {}'.format(TESIndex,256)
    # Flat image
    img, xx, yy = dl.bin_image_elscans(allaz, allels, alldata, [x_min, x_max], nbins_x, TESIndex)
    FitsArray(img).save(dir_flat+'/imgflat_TESNum_{}.fits'.format(TESIndex+1))
    if TESIndex==0:
        FitsArray(xx).save(dir_flat+'/azimuth.fits')
        FitsArray(yy).save(dir_flat+'/elevation.fits')
    
    # Healpix Image
    ok = np.abs(az) <= x_max 
    newtod = tod.copy()
    newtod[:, ~ok] = 0
    sbmap = dl.scan2hpmap(nside, az*np.cos(np.radians(50)), el-50, newtod[TESIndex,:])
    #FitsArray(sbmap).save(dir_healpix+'/healpix_TESNum_{}.fits'.format(TESIndex+1))
    qubic.io.write_map(dir_healpix+'/healpix_TESNum_{}.fits'.format(TESIndex+1), sbmap, compress=True)

    # Plot Flat Images and save them
    rc('figure',figsize=(8,8))
    clf()
    mm, ss = ft.meancut(img[img != 0], 3)
    the_img = img.copy()
    the_img[img < (-ss*nsig_lo)] = -ss*nsig_lo
    imshow(the_img, 
           extent=[x_min*np.cos(np.radians(50)),x_max*np.cos(np.radians(50)), np.min(allels), np.max(allels)], 
           aspect='equal',vmin=-ss*nsig_lo, vmax=ss*nsig_hi)
    colorbar()
    title(freq+'\n FlatMap - TESNum={}'.format(TESIndex+1))
    ylabel('Elevation')
    xlabel('Azimuth x cos(50)')
    savefig(dir_img_flat+'/imgflat_TESNum_{}.pdf'.format(TESIndex+1))

    
#     rc('figure',figsize=(8,8))
#     clf()
#     the_sbmap = sbmap.copy()
#     the_sbmap[sbmap < (-ss*nsig_lo)] = -ss*nsig_lo
#     hp.gnomview(the_sbmap, rot=[0,np.mean(allels)-50], 
#             reso=10,min=-ss*nsig_lo, max=+ss*nsig_hi,
#             title=freq+' - Healpix - TESNum={}'.format(TESIndex+1))
#     savefig(dir_img_healpix+'/imghp_TESNum_{}.pdf'.format(TESIndex+1))






