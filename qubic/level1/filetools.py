'''
$Id: filetools.py
$auth: Steve Torchinsky <satorchi@apc.in2p3.fr>
$created: Mon 23 Jan 2023 14:37:03 CET
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

utilities to write and read QUBIC Level-1 data files
'''
import os
import numpy as np
from astropy.io import fits

from qubic.level1.flags import flag_definition

hdr_comment = {}
hdr_comment['TELESCOP'] ='Telescope used for the observation'
hdr_comment['FILETYPE'] = 'File identification'
hdr_comment['DATASET'] = 'QubicStudio dataset name'
hdr_comment['FILENAME'] = 'name of this file'
hdr_comment['FILEDATE'] = 'UT date this file was created'
hdr_keys = hdr_comment.keys()

def write_level1(fpobject,todarray,flagarray):
    '''
    write a fits file with Level-1 data
    '''
    
    # initialize
    filename_suffix = '_level1.fits'
    filename = fpobject.dataset_name
    hdr = {}
    for key in hdr_keys:
        hdr[key] = None
    hdr['TELESCOP'] = 'QUBIC'
    hdr['FILETYPE'] = 'LEVEL1'
    hdr['FILEDATE'] = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    hdr['DATASET'] = fpobject.dataset_name    
    hdr['FILENAME'] = filename+filename_suffix

    # Primary header
    prihdr = fits.Header()
    for key in hdr.keys():
        val = hdr[key]
        if key in hdr_comment.keys():
            comment = hdr_comment[key]
        else:
            comment = ''
        prihdr[key] = (val,comment)

    # add flag definitions in header
    for idx,flagdef in enumerate(flag_definition):
        if flagdef!='available':
            key = 'FLAG%04i' % idx
            prihdr[key] = flagdef

    # create the primary HDU
    prihdu = fits.PrimaryHDU(header=prihdr)

    # prepare the hdulist
    hdulist = [prihdu]

    # HDU with the level-1 array and the flagarray
    # formats defined at https://docs.astropy.org/en/stable/io/fits/usage/table.html
    col1 = fits.Column(name='TODarray', format='D', unit='ADU', array=todarray)
    col2 = fits.Column(name='flags', format='K', unit='UINT', array=flagarray)
    cols  = fits.ColDefs([col1,col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist.append(tbhdu)

    # HDU with dark TES "temperature" detectors
    
    # make another HDU with the housekeeping data
    # 300mK stage Temperature
    # 1K stage Temperature
    # HWP position
    # azimuth
    # elevation
    # bore sight rotation
    # calsource info and value interpolated to data time axis
    # carbon fibre parameters
    # horn status
    
    # write the FITS file
    thdulist = fits.HDUList(hdulist)
    thdulist.writeto(hdr['FILENAME'],overwrite=True)
    print('Level-1 data written to file: %s' % hdr['FILENAME'])
    return


def read_level1(filename):
    '''
    read a QUBIC Level-1 fits file
    '''

    if not os.path.exists(filename):
        print('ERROR! File does not exist: %s' % filename)
        return None
              
    if not os.path.isfile(filename):
        print('ERROR! Not a file: %s' % filename)
        return None

    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    if nhdu < 3:
        print('ERROR! File does not have the necessary data: %s' % filename)
        return None

    prihdr = hdulist[0].header
    if 'TELESCOP' not in prihdr.keys() or prihdr['TELESCOP']!='QUBIC':
        print('ERROR! Not a QUBIC file: %s' % filename)
        return None

    if 'FILETYPE' not in prihdr.keys() or prihdr['FILETYPE']!='LEVEL1':
        print('ERROR! Not a QUBIC Level-1 file: %s' % filename)
        return None


              
    hdulist.close()
    return ok
