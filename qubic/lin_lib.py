import qubic.fibtools as ft
import qubic.plotters as p

import numpy as np
from matplotlib.pyplot import *
import matplotlib.mlab as mlab
import scipy.ndimage.filters as f
import glob
import string
import datetime as dt

def logistic(x,pars):
    '''
    This function is the logistic function to be used in Minuit fitting

    INPUTS
    x      - FLOAT - The x values
    pars   - FLOAT ARRAY - the parameters
                 pars[0] : Amplitude in Y of the efficiency curve
                 pars[1] : central value in x of the efficiency curve
                 pars[2] : width in x of the efficiency curve
                 pars[3] : offset in y of the efficiency curve

    OUTPUTS
    result  - the logistic function
    '''
    return pars[3]+pars[0]*1./(1+np.exp(-(x-pars[1])/pars[2]))

def source_cal(Vin, freq = 150., folder = '/qubic/Data/Calib-TD/calsource', force_read=False):
    import scipy as sc

    '''
    This function interpolates the source calibration curve and returns an interpolated value at any point

    INPUTS
    Vin    - FLOAT - the input voltage [V]
    freq   - FLOAT - the RF frequency (default = 150 GHz)
    folder - STRING - the folder containing the calibration curve

    OUTPUTS
    result  - FLOAT - the interpolated value at Vin

    '''

    if force_read:
        ### Initial code by Daniele
        calibration_file = '%s/source_calibration_curve_%sGHz.txt' % (folder,int(freq))
        data = np.transpose(np.loadtxt(calibration_file))
    else:
        ### Replaced by JC in order not to read the data each time
        if freq == 150.:
            data = np.array([[1.00584093e-01, 2.12830746e-01, 2.94100096e-01, 3.94684189e-01,
            4.91398314e-01, 5.91964856e-01, 6.88643879e-01, 7.85305350e-01,
            8.85801689e-01, 9.78540539e-01, 1.07900178e+00, 1.17946301e+00,
            1.27216676e+00, 1.37261045e+00, 1.47303658e+00, 1.56570523e+00,
            1.66611381e+00, 1.75493004e+00, 1.86304346e+00, 1.95956453e+00,
            2.05608559e+00, 2.15262421e+00, 2.24912772e+00, 2.34950120e+00,
            2.44989224e+00, 2.54641330e+00, 2.63906440e+00, 2.73555036e+00,
            2.83979381e+00, 2.94020240e+00, 3.03670591e+00, 3.13322698e+00,
            3.22589562e+00, 3.32630421e+00, 3.42286037e+00, 3.52328651e+00,
            3.61986022e+00, 3.71645149e+00, 3.81302521e+00, 3.91348645e+00,
            4.01007772e+00, 4.10668653e+00, 4.21105284e+00, 4.30380924e+00,
            4.40045316e+00, 4.50096705e+00, 4.59764608e+00, 4.69434265e+00,
            4.79107432e+00, 4.89549329e+00, 4.95356037e+00],
           [2.72108844e-03, 4.53514739e-04, 4.53514739e-04, 4.98866213e-03,
            9.52380952e-03, 1.63265306e-02, 2.53968254e-02, 3.67346939e-02,
            5.26077098e-02, 7.07482993e-02, 9.11564626e-02, 1.11564626e-01,
            1.34240363e-01, 1.56916100e-01, 1.81859410e-01, 2.09070295e-01,
            2.36281179e-01, 2.61224490e-01, 2.92970522e-01, 3.22448980e-01,
            3.51927438e-01, 3.79138322e-01, 4.10884354e-01, 4.42630385e-01,
            4.72108844e-01, 5.01587302e-01, 5.31065760e-01, 5.65079365e-01,
            5.96825397e-01, 6.24036281e-01, 6.55782313e-01, 6.85260771e-01,
            7.12471655e-01, 7.39682540e-01, 7.64625850e-01, 7.89569161e-01,
            8.12244898e-01, 8.32653061e-01, 8.55328798e-01, 8.75736961e-01,
            8.96145125e-01, 9.14285714e-01, 9.30158730e-01, 9.46031746e-01,
            9.59637188e-01, 9.73242630e-01, 9.82312925e-01, 9.89115646e-01,
            9.91383220e-01, 1.00045351e+00, 9.98185941e-01]])
        else:
            print('No data is known yet for the frequency you have requested: {}'.format(freq))
            return -1
    
    ##### Initial code by Daniele
    #if Vin <= data[0][0]:
    #    out = data[1][0]
    #elif Vin >= data[0][-1]:
    #    out = data[1][-1]
    #else:
    #    fint = sc.interpolate.interp1d(data[0],data[1])
    #    out = fint(Vin)
    #return out

    ##### Speedup by JC
    return np.interp(Vin, data[0], data[1], left=data[1][0], right=data[1][-1])




def sim_generator_power(time, amplitude, offset, frequency, phase, 
rf_freq = 150.):

    '''
    This function simulates the power output of the source
    taking into account the source calibration curve

    INPUTS
    time      - FLOAT - the current time [s]
    amplitude - FLOAT - the p2p amplitude of the signal in the UCA stage of the generator [V]
    offset    - FLOAT - the offset of the signal in the UCA stage of the generator [V]
    frequency - FLOAT - the modulation frequency [Hz]
    phase     - FLOAT - the wave initial phase
    rf_freq   - FLOAT - the RF frequency (chosen to select the right calibration curve, defaults to 150 GHz)

    OUTPUTS
    uncal_signal, cal_signal  - FLOAT - the uncalibrated signal and the calibrated signal
                                     (percentage of max output power)interpolated value at Vin

    '''

    if type(time) == float:
        tim = np.array([time])
    else:
        tim = time
    sine = amplitude/2. * np.sin(2.*np.pi * frequency * tim + phase) # Divide by 2 because we provide the p2p
    uncal_signal = sine + offset
    #cal_signal = np.array([source_cal(i) for i in uncal_signal])
    cal_signal = source_cal(uncal_signal)
    return cal_signal

def sine(x,pars):    
    '''
    This function is just a wrapper to a sine function to be used for fitting with Minuit. It returns a sine modulation
    of a given amplitude around a given offset
    
    INPUT
    x        - FLOAT - the array of x for which we want the sine
    pars     - FLOAT - the parameters of the sine
                          pars[0] : Peak-to-peak amplitude
                          pars[1] : Modulation frequency (in Hz)
                          pars[2] : phase (in seconds)
                          pars[3] : offset
    '''
    return 0.5*np.sqrt(pars[0]**2)*np.sin(2*np.pi*(x-pars[2])/pars[1])+pars[3]

def shift_src(xyper, pars):
    '''
    This function returns a shifted version of an array supposed to be periodic and scaled in amplitude and offset.
    the array to be shifted is supposed to be in a global variable called "array_to_shift"
    The result is interpolated on the x array abscissas
    This function is to be used for fitting with Minuit
    
    INPUT
    xyper    - LIST  - the input array and period to be shifted
                          xyper[0] : the array of x
                          xyper[1] : the array of y
                          xyper[2] : the periodicity of the signal
    pars     - FLOAT - the parameters of the shift
                          pars[0] : amplitude
                          pars[1] : offset
                          pars[2] : shift in x
    '''
    x = xyper[0]
    y = xyper[1]
    per = xyper[2]
    return np.abs(pars[0])*np.interp(x, x-pars[2], y, period=per)+pars[1]


def sinesat(x,pars):
    '''
    This function returns the model of the source signal for a given modulation configuration (amp, offset, freq, phase) and then
    multiplies it by a global amplitude (calibration factor) and adds a offset.
    This is to be used in fitting with Minuit
    
    INPUT
    x        - FLOAT - the input x
    pars     - FLOAT - the parameters of the source
                          ---------- Calib Params ---------
                          pars[0] : calibration amplitude
                          pars[1] : calibration offset
                          ---------- Modulation Params ----
                          pars[2] : peak to peak amplitude of the source modulation in Volts
                          pars[3] : Period of the modulation [s]
                          pars[4] : the wave initial phase
                          pars[5] : the offset of the signal in the UCA stage of the generator [V]
    '''
    return np.abs(pars[0])*sim_generator_power(x, pars[2], pars[5], 1./pars[3], pars[4])+pars[1]

