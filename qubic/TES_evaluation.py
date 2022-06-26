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
from glob import glob
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
            dataset = ans
        else:
            dataset = 'UNKNOWN-dataset'

    if isinstance(dataset,str) and dataset.find(',')>0:
        dataset = dataset.split(',')

    hdr['DATASET'] = dataset
            
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
        val = hdrval[key]
        if key in hdr_comment.keys():
            comment = hdr_comment[key]
        else:
            comment = ''
        
        if isinstance(val,list):
            for idx,dset in enumerate(val):
                dset_num = idx + 1
                subkey = 'SET%02i' % dset_num
                cardlen = len(dset) + len(comment)
                if cardlen<80:
                    prihdr[subkey] = (dset,comment)
                else:
                    prihdr[subkey] = (dset,'')
                    toolong = True
        else:
            cardlen = len(val) + len(comment)
            if cardlen>80: comment = ''
            prihdr[key] = (val,comment)
            

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

        # dataset could be a list
        if isinstance(val,list):
            for idx,dset in enumerate(val):
                dset_num = idx + 1
                subkey = 'SET%02i' % dset_num
                
                fmt = 'A%i' % max((len(dset),len(comment)))
                col = fits.Column(name=subkey, format=fmt, unit='text', array=[dset,comment])
                columnlist.append(col)
        else:
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

    info = {} # maybe should return this?
    infotitle = {}
    for key in hdr_comment.keys():
        infotitle[key] = hdr_comment[key]
    infotitle['DATASET'] ='Dataset name'
    infotitle['SET'] = 'Dataset name'
    infotitle['ANALYSIS'] = 'Analysis type'
    infotitle['ANALYSER'] = 'Analyser'
    infotitle['ELOG'] = 'elog'
    infotitle['WIKIPAGE'] = 'wikipage'
    infotitle['FILENAME'] = 'Filename'
    infotitle['FILEDATE'] = 'File created'
    hdr = hdulist[2].header
    dat = hdulist[2].data
    nfields = hdr['TFIELDS']
    for idx in range(nfields):
        field_num = idx + 1
        field_key = 'TTYPE%i' % field_num
        field_name = hdr[field_key].strip()
        if field_name.find('SET')==0:
            ttl = infotitle['SET']
        elif field_name in infotitle.keys():
            ttl = infotitle[field_name]
        elif field_name in hdr_comment.keys():
            ttl = hdr_comment[field_name]
        else:
            ttl = "What is this? "
        info[field_name] = dat.field(idx)[0]
        print('%s: %s' % (ttl,info[field_name]))
              
    hdulist.close()
    return ok

def list_goodbad():
    '''
    list all the available TES good/bad files
    '''
    files = []
    
    pattern = '*_GoodBad-TES.fits'
    localfiles = glob(pattern)
    if len(localfiles)>0:
        localfiles.sort()
        files += localfiles

    pkg_dir = os.path.dirname(qubic.__file__)    
    pkgpattern = os.sep.join([pkg_dir,'TES',pattern])
    pkgfiles = glob(pkgpattern)
    if len(pkgfiles)>0:
        pkgfiles.sort()
        for F in pkgfiles:
            files.append(os.path.basename(F))

    if len(files)>0:
        print('The following files are available:\n   %s' % '\n   '.join(files))
        return files

    print('No Good/Bad files found!')
    return files

