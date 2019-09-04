from __future__ import division, print_function
import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys

def select_det(q,id):
    id=[id]
    detector_i = q.detector[id]
    q.detector = detector_i
    return(q)


mpl.style.use('classic')
name='test_scan_source'
resultDir='%s'%name
alaImager=True
component=1
sel_det=True
id_det=232
oneComponent=False

try:
    os.makedirs(resultDir)
except:
    pass


# INSTRUMENT
d = qubic.qubicdict.qubicDict()
d.read_from_file(sys.argv[1])

q = qubic.QubicInstrument(d)

if sel_det:
    q=select_det(q,id_det)


p= qubic.get_pointing(d)


plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
plt.plot(p.time[:10000],p.azimuth[:10000])
plt.ylabel('Azimuth',fontsize=22)
plt.subplot(3,1,2)
plt.plot(p.time[:10000],p.elevation[:10000])
plt.ylabel('Elevation',fontsize=22)
plt.subplot(3,1,3)
plt.plot(p.time[:10000],p.angle_hwp[:10000])
plt.ylabel('HWP',fontsize=22)
plt.show()

s = qubic.QubicScene(d)

sb = q.get_synthbeam(s, 0)
xr=0.1*np.max(sb)
hp.gnomview(sb, rot=[0,90], xsize=500, reso=5, min=-xr, max=xr,title='Input ', return_projected_map=True,hold=True)
plt.show()
fix_azimuth=d['fix_azimuth']



m0=np.zeros(12*d['nside']**2)
x0=np.zeros((len(m0),3))
id=hp.pixelfunc.ang2pix(d['nside'], fix_azimuth['az'], fix_azimuth['el'],lonlat=True)
source=m0*0
source[id]=1
arcToRad=np.pi/(180*60.)
source=hp.sphtfunc.smoothing(source,fwhm=30*arcToRad)
x0[:,component]=source

if p.fix_az:
    center = (fix_azimuth['az'],fix_azimuth['el'])
else:
    center = qubic.equ2gal(d['RA_center'], d['DEC_center'])

a = qubic.QubicAcquisition(q, p, s, d)
TOD = a.get_observation(x0, noiseless=True)

plt.plot(TOD[0,:])
plt.show()



if alaImager==True:
    
    d['synthbeam_kmax']=0
    if oneComponent==True:
        d['kind']='I'
    q = qubic.QubicInstrument(d)
    if sel_det:
        q=select_det(q,id_det)
    
    p= qubic.get_pointing(d)
    s = qubic.QubicScene(d)
    arec = qubic.QubicAcquisition(q, p, s,d)
else:
    arec = qubic.QubicAcquisition(q, p, s,d)


maps_recon = arec.tod2map(TOD, d)
print(maps_recon.shape)

cov = arec.get_coverage()
cov[id]=np.nan
hp.mollview(cov)
plt.show()
im_old=hp.gnomview(cov, rot=center, reso=5,title='cov ', return_projected_map=True,hold=True, xsize=500)
plt.show()

cov = np.sum(cov, axis=0)
maxcov = np.max(cov)
unseen = cov < maxcov*0.1
maps_recon[:,unseen] = hp.UNSEEN

stokes = ['I', 'Q', 'U']


plt.figure(figsize=(15,8))
count=1

if d['kind']=='I':
    xr=0.01*np.max(maps_recon[:])
    im_old=hp.gnomview(maps_recon[:], rot=center, reso=5, min=-xr, max=xr,title='Output ', return_projected_map=True,hold=True, xsize=500)
    plt.show()
else:
    for istokes in [0,1,2]:
        plt.subplot(1,3,count)
        xr=0.009
        im_old=hp.gnomview(maps_recon[:,istokes], xsize=500, rot=center, reso=5, min=-xr, max=xr,title='Output '+stokes[istokes], return_projected_map=True,hold=True)
        count+=1
    plt.show()

    P=np.sqrt(maps_recon[:,1]**2+maps_recon[:,2]**2)

    plt.figure(figsize=(15,8))

    plt.subplot(1,2,1)
    hp.gnomview(P, xsize=500, rot=center, reso=5,title='Output P' , return_projected_map=True,hold=True)
    plt.subplot(1,2,2)
    hp.gnomview(sb, rot=[0,90], xsize=500, reso=5,title='Input ', return_projected_map=True,hold=True)
    plt.show()

