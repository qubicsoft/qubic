'''
$Id: saturation.py
$auth: Steve Torchinsky <satorchi@apc.in2p3.fr>
$created: Mon 23 Jan 2023 14:57:27 CET
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

check data for saturation and assign flag
'''
import numpy as np
from qubic.level1.flags import flag_bit, set_flag

# This is the value where data flattens out.  See for example: 2019-12-23_19.00.20__HWP_Scanning_Vtes_2TimePerPos_60
saturation_value = 2**22 - 2**7
positive_saturation_value = saturation_value
negative_saturation_value = -(saturation_value-1)
saturation_bit = flag_bit['saturation']

def check_saturation(todarray,flag_array=None):
    '''
    go through the data, and return the flag array adjusted for saturation
    '''

    tod_shape = todarray.shape
    
    # if no flag_array is given, then create it
    if flag_array is None: flag_array = np.zeros(tod_shape,dtype=np.uint)
    
    # check for saturation
    positive_saturation = todarray >= positive_saturation_value
    negative_saturation = todarray <= negative_saturation_value
    saturation = positive_saturation | negative_saturation

    flag_array[saturation] = set_flag('saturation',flag_array[saturation])

    return flag_array

