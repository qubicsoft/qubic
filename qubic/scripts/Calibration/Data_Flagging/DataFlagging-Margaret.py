'''
$Id: DataFlag.py
$auth: DQA team lead by Steve Torchinsky and JCh <satorchi@apc.in2p3.fr>
$created: Fri 23 Jan 2022 10:00:39 CEST
$tec. contact: mgamboa@fcaglp.unlp.edu.ar
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

Create a pipeline to flag and mask data
'''

import glob
import sys
import os
import warnings
import gc

# To fit===================================
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
#==========================================
#import string
import numpy as np
import scipy.signal as scsig
from scipy import interpolate
from astropy.io import fits as pyfits
import datetime as dt
import pickle
from importlib import reload
import datetime
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
rc('figure',figsize=(9,4.5))
rc('font',size=12)
rc('text',usetex=False)
rc('axes', facecolor = 'white')
rc('savefig', facecolor = 'white')

from qubicpack.qubicfp import qubicfp
import qubicpack as qp
from pysimulators import FitsArray
from qubicpack.utilities import TES_index, figure_window_title
from qubicpack.timeline import timeline_timeaxis
from qubicpack.utilities import qc_utc_date

import matplotlib.cm as cm

from sklearn.preprocessing import MinMaxScaler

__all__ = ['FlagData',
           'FlagToMask']

# The program begins
def verbosity_files(day, dirs, kwd0, ifile, setup = True):
    if setup:
        if ifile == 0:
            print('===================================')
            print('day {} - has {} files of {} test'. format(day, 
                                                         len(dirs), 
                                                         kwd0))
        else:
            if dirs[ifile][57:70] != kwd0:
                kwd0 = dirs[ifile][57:70]
                print('===================================')
                print('day {} - has {} files of {} test'. format(day, 
                                                         len(dirs), 
                                                         dirs[ifile][57:])) 
        return
    else:
        return
# Fit
#model
# our line model, unicode parameter names are supported :)
def line(x, a, b):
    return b + x * a

def ReadDataSet(datestart = None, dateend = None,
                keywords = None,
                verbose = False):

    """
    
    datestart, dateend: Dates. DD-MM-YYYY format
    keywords: list of strings. name of the scanning (provided on eLog)
    verbose: boolean.

    Hard coded variables: 
        data_dir: '/sps/qubic/Data/Calib-TD/'+day+'/'
        Path where the data is.

    Return: 
        DataContainer: dictionary. Keywords: 
            'YYYY-MM-DD' for each day --> value: list of qubicfp. One for each dataset within the day
            'fnamesYYYY-MM-DD' --> value: list of string. Name of each experiment stored in 'YYYY-MM-DD'
            'kwdays' --> value: list of days. 
    """

    # Crete list of days in which I will read the data
    def validate(date_text):
        try:
            datetime.datetime.strptime(date_text, '%d-%m-%Y')
        except ValueError:
            raise ValueError("Incorrect data format, should be DD-MM-YYYY")
    validate(datestart)
    validate(dateend)
    #assert isinstance(datestart, str) & isinstance(dateend, str),\
    #    'The format of the date is not correct. It should be a an instance of datetime.datetime'
    start = datetime.datetime.strptime(datestart, "%d-%m-%Y")
    end = datetime.datetime.strptime(dateend, "%d-%m-%Y")
    
    date_generated = pd.date_range(start, end = end)# periods=5)
    days = date_generated.strftime("%Y-%m-%d").to_list() 

    # Check    
    if verbose: print('Days to read data', days)

    # Create list of keyword to select experiment/s
    for i in range(len(keywords)):
        keywords[i] = '*' + keywords[i] + '*' #['Responsivity']#, 'hwp']
    # Create glob's format to read systematically the dataset

    # container of data in Dictionary. We don't use pd.DataFrame because each
    # variable has to have the same length and here we don't have it.
    addfile = 0
    DataContainer = {}
    alldays = []
    for keyword in keywords:
        for day in days:
            data_dir = '/sps/qubic/Data/Calib-TD/'+day+'/'
            dirs = np.sort(glob.glob(data_dir+keyword))
            if len(dirs) == 0:
                pass
            else:
                # Create a keyword with the day within the container
                DataContainer['{}'.format(day)] = {}
                auxdata = []

                # Load the focal plane
                #loop in dirs
                filenames = []
                for ifile in range(0, len(dirs)):
                    print(ifile)
                    #loop in keyword for same day
                    # Printout
                    if verbose: 
                        kwd0 = dirs[0][57:]
                        verbosity_files(day, dirs, kwd0, ifile, setup = True)

                    addfile += 1

                    thedir = dirs[ifile]
                    if keyword == '*':
                        #print('================', thedir[57:],) if ifile == 0 else None
                        auxdata = None
                        filenames.append(thedir[57:])
                    else:
                        qfpName = 'qfp{}_{}'.format(day.replace('-',''),ifile)
                        locals()[qfpName] = qubicfp()
                        locals()[qfpName].assign_verbosity(1)
                        locals()[qfpName].read_qubicstudio_dataset(thedir)
                        auxdata.append(locals()[qfpName])
                        #save memory
                        if verbose:
                            bytetoMB = 9.537e-7
                            print('Each data array has approximatelly',
                            sys.getsizeof(locals()[qfpName]) * \
                                bytetoMB, 'MB') # in MB
                        
                        # Free memory
                        del locals()[qfpName]
                        gc.collect()

                        # Save the filenames
                        filenames.append(thedir)
                # Save date
                alldays.append(day)
                # Save data
                DataContainer.update({'{}'.format(day): auxdata, 'fnames{}'.format(day): filenames})
    if verbose: print('There are {} files'.format(addfile))
    DataContainer.update({'kwdays': alldays})

    return DataContainer 

##### =================
####
###     FLAG DATA
##
#   ===================

class FlagData(object):
    def __init__(self, iDataContainer):
        """
        Initialization of the data to be flagged. 

        Arguments:
            iDataContainer: qubicfp object. It can also be used the output of the
            ReadDataSet by doing ReadDataSet['YYYY-MM-DD'][i], where 'YYYY-MM-DD' is a day
            and 'i' is an index of the list.

            channel: str. This argument is used to extract house keeping data from a particular
            channel ['AVS47_1_CH6'] = '0.3K fridge CH' or ['AVS47_1_ch1'] = '1K stage'.
        """

        self.data = iDataContainer
        # Extract the regular time grid
        self.regtime, _ = iDataContainer.tod()

        return

    def interpolate_flag(self, rawt, flags):
        """"
        Arguments:
            rawt: array. Raw time, base of the time used with for the 
            flagged array (flags)
            flags: dict. Output of the methods to flag data. They should have
            a key named 'flag' with the array of flags.
        
        Return: 
            Array with interpolated flags in the base of the regular time by .tod()
            method from qubicfp instance.
        """
        flags_interp = np.max(np.array([interp1d(rawt, flags['flag_raw'], kind='previous', 
                                               fill_value='extrapolate')(self.regtime), 
                                      interp1d(rawt, flags['flag_raw'], kind='next', 
                                               fill_value='extrapolate')(self.regtime)], dtype = np.int64), axis=0)

        return flags_interp
    
    def flag_sat(self, upper = 4.19*(10**6), lower = -4.19*(10**6), interpolate = True, verbose = False):
        """
        Author: Margaret Haun
        Created: 2023-06-06 15:45:00 CET
        Description:  Flag saturation of the detector

        ======================
        Arguments:
            dataset: TOD
            upper: upper limit of the detector in ADU
            lower: lower limit of the detector in ADU
        Returns:
            Flagged array [same size as TOD]

        This function flags any data that exceeds the thresholds of the detector's dynamic range.
        We assign the flag for saturation to the number 63. 
        """
        sat_flag = 63
        
        _, tod = self.data.tod()
            
        def bit_id_sat(ADU):
            
            flag_assign = np.zeros_like(ADU, dtype=np.int64)
            
            for i in range(ADU.shape[0]):
                is_saturated = (ADU[i] >= upper) | (ADU[i] <= lower)
                flag_assign[i] = is_saturated.astype(int)*(2**sat_flag)
            return flag_assign
    
        retval = {}
        retval['flag'] = bit_id_sat(tod)
        
        return retval

    def flag_bathtemp(self, T_bit37 = 0.330, T_bit38 = 0.340, 
                    T_bit39 = 0.350, interpolate = True,
                    verbose = False):
        """
        Author: Martín Gamboa
        Created: 2023-01-23 15:25:00 CET
        Description:  Flag bath temperature

        ======================
        Arguments:
            dataset: TOD
            interpolate: boolean. If True return the interpolated flags
            T_bitXX [in K]: Temperature from which the sample is flagged. 
                            XX is the corresponding bit for that flag.
        Returns:
            Flagged array [same size as TOD]
            
        The flags are set for each timesample and considering if the temperature is 
                                            above 330mK, above 340mK and above 350mK.
        The corresponding flagid is 37, 38 and 39 respectively and the computation is
        flag_i = bit37*2**37 + bit38*2**38 + bit39*2**39
        """
        
        #Get raw time
        rawtime = np.copy(self.data.get_hk(data='RaspberryDate',hk='EXTERN_HK'))
        #Get data from channel
        dataset = np.copy(self.data.get_hk('AVS47_1_CH6'))

        def bitid_300mK(T):
            # Return an integer
            return (int(T > T_bit37) & int(T < T_bit38)) * 2**37 + \
                (int(T > T_bit38) & int(T < T_bit39)) * 2**38 + \
                int(T > T_bit39) * 2**39

        flags300mK = list(map(bitid_300mK, dataset))

        retval = {}
        
        retval['flag_raw'] = np.array(flags300mK, dtype = np.int64)
        
        #Resample
        # ..
        #
        if interpolate:
            retval['flag'] = self.interpolate_flag(rawtime,
                                                    retval)

        return retval

    def flag_bathtemp_rise(self, interpolate = True, verbose = False):
        """
        Author: Martín Gamboa
        Created: 2023-01-23 15:25:00 CET
        Description:  Flag bath temperature raise

        ======================
        Arguments:
            dataset: TOD
            interpolate: boolean. If True return the interpolated flags
            Verbose: boolean
        Returns:
            Flagged array [same size as TOD]. The last value is equal to the 
            previous one in order to have the same dimenssion of arrays
            
        A timestamp is flagged if the timestamp has a higher temperature compared 
                    with the previous timestamp value (interpolated using tod()[0].
        The corresponding flagid is 36,
        flag_i = bit36*2**36
        """

        #Get raw time
        rawtime = np.copy(self.data.get_hk(data='RaspberryDate',hk='EXTERN_HK'))
        #Get data from channel
        dataset = np.copy(self.data.get_hk('AVS47_1_CH6'))

        signflag = np.zeros_like(dataset)
        # Compute the difference in temperature between consecutive timestamps
        #  + Get the sign (+ --> temperature rising, - --> temperature decreasing)
        signflag[:-1] = np.sign(np.diff(dataset))

        # Repeat value for the last sample to preserve the dimenssion of the arrays
        signflag[-1] = signflag[-2]
        # 
        
        def bitid_rise300mK(T):
            # Return an integer
            return (int(T > 0) * 2**36)
        
        flag300mKRise = list(map(bitid_rise300mK, signflag))
        
        retval = {}
        retval['flag_raw'] = np.array(flag300mKRise, dtype = np.int64)
        if interpolate:
            retval['flag'] = self.interpolate_flag(rawtime,
                                                    retval)
        return retval

    def flag_1Ktemp(self, T_bit31 = 1.1, T_bit32 = 1.2, T_bit33 = 1.3, 
                    interpolate = True, verbose = False):
        """
        Author: Martín Gamboa
        Created: 2023-01-23 15:25:00 CET
        Description:  Flag 1K temperature

        ======================
        Arguments:
            dataset: TOD
            interpolate: boolean. If True return the interpolated flags
            T_bitXX [in K]: Temperature from which the sample is flagged. 
                            XX is the corresponding bit for that flag.
        Returns:
            Flagged array [same size as TOD]
            
        The flags are set for each timesample and considering if the temperature is above 1.1K, 
                                                                    above 1.2K and above 1.3K.
        The corresponding flagid is 31, 32 and 33 respectively and the computation is
        flag_i = bit31*2**31 + bit32*2**32 + bit33*2**39
        """

        #Get raw time
        rawtime = np.copy(self.data.get_hk(data='RaspberryDate',hk='EXTERN_HK'))
        #Get data from channel
        dataset = np.copy(self.data.get_hk('AVS47_1_ch1'))

        def bitid_1K(T):
            # Return an integer
            return (int(T > T_bit31) & int(T < T_bit32)) * 2**31 + \
                (int(T > T_bit32) & int(T < T_bit33)) * 2**32 + \
                int(T > T_bit33) * 2**33

        flags1K = list(map(bitid_1K, dataset))
        
        retval = {}
        retval['flag_raw'] = np.array(flags1K, dtype = np.int64)
        if interpolate:
            retval['flag'] = self.interpolate_flag(rawtime,
                                                    retval)

        return retval

    def flag_1Ktemp_rise(self, interpolate = True, verbose = False):    
        """
        Author: Martín Gamboa
        Created: 2023-01-23 15:25:00 CET
        Description:  Flag 1K temperature raise

        ======================
        Arguments:
            dataset: TOD
            interpolate: boolean. If True return the interpolated flags
            Verbose: boolean
        Returns:
            Flagged array [same size as TOD]. The last value is equal to the previous one 
            in ordcer to have the same dimenssion of arrays
            
        A timestamp is flagged if the timestamp has a higher temperature compared with the 
        previous timestamp value (interpolated using tod()[0].
        The corresponding flagid is 30,
        flag_i = bit30*2**30
        """

        #Get raw time
        rawtime = np.copy(self.data.get_hk(data='RaspberryDate',hk='EXTERN_HK'))
        #Get data from channel
        dataset = np.copy(self.data.get_hk('AVS47_1_CH6'))

        signflag = np.zeros_like(dataset)
        # Compute the difference in temperature between consecutive timestamps
        #  + Get the sign (+ --> temperature rising, - --> temperature decreasing)
        signflag[:-1] = np.sign(np.diff(dataset))
        signflag[-1] = signflag[-2]
        # 
        
        def bitid_rise1K(T):
            # Return an integer
            return (int(T > 0) * 2**30)
        
        flag1KRise = list(map(bitid_rise1K, signflag))
        
        retval = {}
        retval['flag_raw'] = np.array(flag1KRise, dtype = np.int64)
        if interpolate:
            retval['flag'] = self.interpolate_flag(rawtime,
                                                    retval)

        return retval

    def flag_1Ktemp_rise_beta(self, step = 80, interpolate = True, verbose = False):    
        
        """
        Author: Margaret Haun
        Created: 2023-07-07 17:00:00 CET
        Description:  Flag 1K temperature raise

        ======================
        Arguments:
            dataset: TOD
            step: int divisible by 2. Number of timestamps over which to compute slope angle
            interpolate: boolean. If True return the interpolated flags
            Verbose: boolean
        Returns:
            Flagged array [same size as TOD]. The final set of timestamps (of count equal to step) are set equal to the previous one 
            in order to have the same dimension of arrays
            
        A timestamp is flagged if the slope of the line between the timestamp[i] and timestamp[i+step] is greater than 7.5e-06 
        (interpolated using tod()[0]).
        The corresponding flagid is 30,
        flag_i = bit30*2**30
        """
        
        #Get raw time
        rawtime = np.copy(self.data.get_hk(data='RaspberryDate',hk='EXTERN_HK'))
        #Get data from channel
        dataset = np.copy(self.data.get_hk('AVS47_1_ch1'))
        
        #Alter step if not divisible by 2
        if step%2 == 1:
            step += 1
            print('Warning: Provided "step" arg is not divisible by 2. Provided value has been added to 1 for a step of ', step)
        
        #Slopes found using difference in y over difference in x
        slopes = []
        for i in range(len(rawtime)-step):
            diffx = rawtime[i+step]-rawtime[i]
            #Check for spikes and ignore them
            #Learn from experience and take necessary precautions to avoid hardcoding
            if dataset[i+step] > 2:
                dataset[i+step] = dataset[i+step-1]
            if dataset[i] > 2:
                if i==0:
                    dataset[i] = dataset[i+1]
                else:
                    dataset[i] = dataset[i-1]
            diffy = dataset[i+step]-dataset[i]
            if diffx != 0:
                slope = diffy/diffx
                slopes.append(slope)
            #Not sure how or why it might come up, but just in case, we have a way to avoid dividing by zero
            else:
                slope = 0
                slopes.append(slope)
        #Copy values on each end to maintain shape and symmetry
        for i in range(int(step/2)):
            slopes.append(slopes[-1])
            slopes.insert(0, slopes[0])
        
        def bitid_rise1K(M):
            # Return an integer
            return (int(M > 0.0000075) * 2**30)
        
        flag1KRise = list(map(bitid_rise1K, slopes))
        
        retval = {}
        retval['flag_raw'] = np.array(flag1KRise, dtype = np.int64)
        if interpolate:
            retval['flag'] = self.interpolate_flag(rawtime,
                                                    retval)
        return retval


class FlagToMask(): 
    def __init__(self):
        """
        This class aims to mask the data according to user requirements. 

        The posible flags are:  'saturation', 'cosmic ray', 'uncorrelated flux jumps', 'end of scan',
                    'Tbath above 330mK', 'Tbath above 340mK', 'Tbath above 350mK', 'Tbath rising',
                    '1K above 1.1K', '1K above 1.2K', '1K above 1.3K', '1K rising',
                    'correlated flux jumps'

        ==============
        Arguments:
            flagarray: 
                    Array of 64-bit integers for a specific flag. It can be use an array of flags.
            userflag: Default values for each key = False
                    Dictionary with boolean values with user requirements 
        =========================
        Return:
            maskedata: 
                    Masked dataset

        =========================
        Example:
            If we have just five timesamples with the following features: 
                    saturated + Tbath 335mK, Tbath 380mK, 1K rising, no flag data, Tbath 345mK

            #Consider you read the files with the flagged arrays and add each array properly (TBD), you would get 

            flagarray = np.array([2**63 + 2**37, 2**39, 2**30, 0, 2**38])

            #or

            flagarray = np.array([9223372174293729280, 549755813888, 1073741824, 0, 274877906944])

            # Now, you requirements for the flagged array are: Not saturated TES and Tbath below 350mK:

            maskdict = {'saturation': True, 'Tbath above 350mK': True}

            maskarray = MaskDataWithFlags(flagarray, maskdict) 
            #--> returns a mask with the timesamples that satisfy the requirements
            print(maskarray)
            out: np.array([0, 0, 1, 1, 1])
        """

        # Testing initialization 
        self.FullFlags = {'saturation': False, 'cosmic ray': False, 
                    'uncorrelated flux jumps': False, 'end of scan': False,
                    'Tbath above 330mK': False, 'Tbath above 340mK': False, 
                    'Tbath above 350mK': False, 'Tbath rising': False,
                    '1K above 1.1K': False, '1K above 1.2K': False, 
                    '1K above 1.3K': False, '1K rising': False,
                    'correlated flux jumps': False}
        # Bit correspondance
        self.BitFlags = {'saturation': 63, 'cosmic ray': 57, 
                    'uncorrelated flux jumps': 51, 'end of scan': 45,
                    'Tbath above 350mK': 39, 'Tbath above 340mK': 38, 
                    'Tbath above 330mK': 37, 'Tbath rising': 36,
                    '1K above 1.1K': 31, '1K above 1.2K': 32, 
                    '1K above 1.3K': 33, '1K rising': 30,
                    'correlated flux jumps': 27}
    def __call__(self, flagarray, useropts):#, **newflags):)
        # Update dictionary with user requirements
        
        self.FullFlags.update(useropts)

        # Create an empty mask as it's all data OK (= 0 for each timesample)
        # Look at the requirements
        BitIds = []
        for iflag in self.FullFlags.keys():
            if self.FullFlags[iflag]:
                BitIds.append(self.BitFlags[iflag])
        
        
        #Mask one sample through a function
        def masksample(iflagarray, BitIds = BitIds):
            return [not bool((int(iflagarray) & int(2**bit)) >> int(bit)) for bit in BitIds]
        
        MaskData = list(map(masksample, flagarray))

        return np.prod(MaskData, axis = 1)

class FlagToMask_beta(): 
    def __init__(self):
        """
        i think something could be better here
        """

        # Testing initialization 
        self.FullFlags = {'saturation': False, 'cosmic ray': False, 
                    'uncorrelated flux jumps': False, 'end of scan': False,
                    'Tbath above 330mK': False, 'Tbath above 340mK': False, 
                    'Tbath above 350mK': False, 'Tbath rising': False,
                    '1K above 1.1K': False, '1K above 1.2K': False, 
                    '1K above 1.3K': False, '1K rising': False,
                    'correlated flux jumps': False}
        # Bit correspondance
        self.BitFlags = {'saturation': 63, 'cosmic ray': 57, 
                    'uncorrelated flux jumps': 51, 'end of scan': 45,
                    'Tbath above 350mK': 39, 'Tbath above 340mK': 38, 
                    'Tbath above 330mK': 37, 'Tbath rising': 36,
                    '1K above 1.1K': 31, '1K above 1.2K': 32, 
                    '1K above 1.3K': 33, '1K rising': 30,
                    'correlated flux jumps': 27}
    def __call__(self, flagarray, useropts):#, **newflags):)
        # Update dictionary with user requirements
        
        self.FullFlags.update(useropts)

        # Create an empty mask as it's all data OK (= 0 for each timesample)
        # Look at the requirements
        BitIds = []
        for iflag in self.FullFlags.keys():
            if self.FullFlags[iflag]:
                BitIds.append(self.BitFlags[iflag])
        
        
        #Mask one sample through a function
        def masksample(iflagarray, BitIds = BitIds):
            return [not bool((int(iflagarray) & int(2**bit)) >> int(bit)) for bit in BitIds]
        
        MaskData = list(map(masksample, flagarray))

        return np.prod(MaskData, axis = 1)
# =====================================================================
    
def detect_outliers(timeset, dataset, channel):
    
    if channel == 'AVS47_1_CH6': #300mK stage
        threshold = 10.
    elif channel == 'AVS47_1_ch1': #1K stage
        threshold = 1000.
    else: 
        raise ValueError('No channel programed into this method. Possible values: "AVS47_1_CH6" or "AVS47_1_ch1"')
    
    # Create mask
    mask = dataset > threshold
    
    # interpolation
    ynew = np.interp(timeset[mask],timeset[~mask], dataset[~mask])
    
    # supplant values
    dataset[mask] = ynew
    if sum(mask) > 0:
        warnings.warn('Outliers values detected')
    return mask, dataset