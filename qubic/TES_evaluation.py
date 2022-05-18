#!/usr/bin/env python3
'''
$Id: TES_evaluation.py
$auth: Steve Torchinsky <satorchi@apc.in2p3.fr>
$created: Tue 17 May 2022 16:57:19 CEST
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

write a fits file with a boolean array for all TES good/bad
The FITS header had info about how the determination was made including:
 dataset name
 analysis type (I-V, NEP, timeconstant, carbon fibre, linearity,...)
 analyser (person who did the analysis)
 elog entry (link to elog entry for the data set)
 wiki page (link to the wiki page with more information)

'''
import os
from astropy.io import fits
import numpy as np
import datetime as dt
import qubic

hdr_keys = ['TELESCOP',
            'FILETYPE',
            'DATASET',
            'ANALYSIS',
            'ANALYSER',
            'ELOG',
            'WIKIPAGE',
            'FILENAME',
            'FILEDATE']

hdr_comment = {}
hdr_comment['TELESCOP'] ='Telescope used for the observation'
hdr_comment['FILETYPE'] = 'File identification'
hdr_comment['DATASET'] = 'QubicStudio dataset name'
hdr_comment['ANALYSIS'] = 'Analysis type used to determine TES quality'
hdr_comment['ANALYSER'] = 'the person who did the analysis'
hdr_comment['ELOG'] = 'link to the elog entry for the data'
hdr_comment['WIKIPAGE'] = 'link to the wiki page where there is more information'
hdr_comment['FILENAME'] = 'name of this file'
hdr_comment['FILEDATE'] = 'UT date this file was created'

def get_header_values(dataset=None,
                      analysis=None,
                      analyser=None,
                      elog=None,
                      wikipage=None):
    '''
    get the descriptive information
    '''

    # initialize
    filename_suffix = '_GoodBad-TES.fits'
    filename = ''
    hdr = {}
    for key in hdr_keys:
        hdr[key] = None
    hdr['TELESCOP'] = 'QUBIC'
    hdr['FILETYPE'] = 'GOODBAD'
    hdr['DATASET'] = dataset
    hdr['ANALYSIS'] = analysis
    hdr['ANALYSER'] = analyser
    hdr['ELOG'] = elog
    hdr['WIKIPAGE'] = wikipage
    hdr['FILEDATE'] = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

    if dataset is None:
        ans = input('Enter dataset name: ')
        if len(ans.strip())>0:            
            if ans.find(',')>0: ans = ans.split(',')
            hdr['DATASET'] = ans
        else:
            hdr['DATASET'] = 'UNKNOWN-dataset'

    if isinstance(hdr['DATASET'],list):
        first_dataset = os.path.basename(hdr['DATASET'][0])
    else:
        first_dataset = os.path.basename(hdr['DATASET'])
    filename = first_dataset.strip().replace(' ','_')

    if analysis is None:
        ans = input('Enter analysis method (eg. IV, NEP, timeconstant,...): ')
        hdr['ANALYSIS'] = ans
    filename += '_%s' % (hdr['ANALYSIS'].strip().replace(' ','_'))

    if analyser is None:
        ans = input('Who did the analysis? ')
        hdr['ANALYSER'] = ans

    if elog is None:
        ans = input('Enter the elog entry for the dataset (if any): ')
        if len(ans)==0:
            elog_url = ''
        else:
            try:
                elog_num = int(ans)
                elog_url = 'https://elog-qubic.in2p3.fr/demo/%i' % elog_num
            except:
                elog_url = ans
        hdr['ELOG'] = elog_url

    if wikipage is None:
        ans = input('Enter wikipage where we can find more information: ')
        hdr['WIKIPAGE'] = ans

    hdr['FILENAME'] = filename+filename_suffix
    return hdr


def check_goodbad(goodbad):
    '''
    check that the given array is an array of boolean
    '''
    if isinstance(goodbad,list) or isinstance(goodbad,tuple):
        goodbad = np.array(goodbad)

    if not isinstance(goodbad,np.ndarray):
        print('Inappropriate input.  Please input a 1-dimensional array of Boolean')
        return None

    return goodbad


def write_goodbad(goodbad,
                  dataset=None,
                  analysis=None,
                  analyser=None,
                  elog=None,
                  wikipage=None):
    '''
    write a FITS file with a good/bad info
    '''

    hdrval = get_header_values(dataset,
                               analysis,
                               analyser,
                               elog,
                               wikipage)

    goodbad = check_goodbad(goodbad)
    if goodbad is None: return


    
    prihdr = fits.Header()
    toolong = False
    for key in hdrval.keys():
        cardlen_nocomment = len(key)+len(hdrval[key])
        if cardlen_nocomment>80:
            toolong = True
            prihdr[key] = (hdrval[key],'')
            continue
        
        if key in hdr_comment.keys() and (cardlen_nocomment+len(hdr_comment[key]))<=80:
            prihdr[key] = (hdrval[key],hdr_comment[key])
        else:
            prihdr[key] = (hdrval[key],'')
            

    if toolong:
        prihdr['WARNING'] = ('CUTOFF','go to HDU3 for details')
        print('Truncated information can be found in full in the 3rd HDU')
            
    prihdu = fits.PrimaryHDU(header=prihdr)

    # prepare the hdulist
    hdulist = [prihdu]

    # the good/bad array is in its own hdu
    col = fits.Column(name='GOODBAD', format='bool', unit='Bool', array=goodbad)
    cols  = fits.ColDefs([col])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist.append(tbhdu)

    # make another HDU with additional info because it gets cut off in the primary header
    # this is like a secondary primary header
    columnlist = []
    for key in hdrval.keys():
        if key in hdr_comment.keys():
            comment = hdr_comment[key]
        else:
            comment = ''
        val = hdrval[key]
        fmt = 'A%i' % max((len(val),len(comment)))
        col = fits.Column(name=key, format=fmt, unit='text', array=[val,comment])
        columnlist.append(col)
    
    
    cols  = fits.ColDefs(columnlist)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist.append(tbhdu)

    # write the FITS file
    thdulist = fits.HDUList(hdulist)
    thdulist.writeto(hdrval['FILENAME'],overwrite=True)
    print('good/bad TES information written to file: %s' % hdrval['FILENAME'])
    return

def read_goodbad(filename,verbosity=1):
    '''
    read a FITS file with good/bad information
    '''

    filename_fullpath = None
    if os.path.exists(filename):
        filename_fullpath = filename
        if not os.path.isfile(filename_fullpath):
            print('Not a file: %s' % filename_fullpath)
            return None

    # if not found, try to read from the default location
    if filename_fullpath is None:
        pkg_dir = os.path.dirname(qubic.__file__)
        basename = os.path.basename(filename)
        filename_fullpath = os.sep.join([pkg_dir,'TES',basename])
        if not os.path.exists(filename_fullpath):
            print('File not found: %s' % filename_fullpath)
            return None

    filename = filename_fullpath

    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    if nhdu < 3:
        print('File does not have the necessary data: %s' % filename)
        return None

    prihdr = hdulist[0].header
    if 'TELESCOP' not in prihdr.keys() or prihdr['TELESCOP']!='QUBIC':
        print('Not a QUBIC file: %s' % filename)
        return None

    if 'FILETYPE' not in prihdr.keys() or prihdr['FILETYPE']!='GOODBAD':
        print('Not a QUBIC TES good/bad file: %s' % filename)
        return None

    ok = np.array(hdulist[1].data.field(0),dtype=bool)
    if verbosity<1: return ok

    info = {}
    infokeys = ['Dataset name','Analysis type','Analyser','elog','wikipage','Filename','File created']
    for idx,key in enumerate(infokeys):
        field_num = idx + 2
        info[key] = hdulist[2].data.field(field_num)[0]
        print('%s: %s' % (key,info[key]))
              
    
    return ok
