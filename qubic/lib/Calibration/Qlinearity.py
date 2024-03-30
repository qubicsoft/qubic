import qubic.lib.Calibration.Qfiber as ft
import qubic.lib.Qplotters as p

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
    

def source_cal(Vin, freq = 150., finterp_Vin = '', finterp_freq = ''):
    import numpy as np
    from scipy import interpolate


    '''
    This function interpolates the source calibration curve and returns an interpolated value at any point

    INPUTS
    Vin          - FLOAT - the input voltage [V]
    freq         - FLOAT - the RF frequency (default = 150 GHz)
    finterp_Vin  - MIXED - the interpolating function that returns the fraction of power
                           as a function of Vin. If it is provided externally then it is of type
                           scipy.interpolate.interp2d. Default is an empty string, in this case the interpolation
                           is carried out internally
    finterp_freq - MIXED - the interpolating function that returns the maximum power
                           as a function of the frequency in GHz. If it is provided externally then it is of type
                           scipy.interpolate.interp1d. Default is an empty string, in this case the interpolation
                           is carried out internally
    OUTPUTS
    result  - FLOAT - the interpolated value at Vin, freq

    '''

    # Check if the interpolating function is provided externally
    if type(finterp_Vin).__name__ != 'interp2d':
        
        # The array "data" contains the calibration curves at the frequencies 130, 140, 150,
        # 160 and 170 GHz. The array data[0] contains the Vin (abscissa) points, the arrays
        # data[1]-->data[6] contain the five calibration curves

        data = np.array([[1.00000000e-01, 1.98039216e-01, 2.96078431e-01, 3.94117647e-01,\
                          4.92156863e-01, 5.90196078e-01, 6.88235294e-01, 7.86274510e-01,\
                          8.84313725e-01, 9.82352941e-01, 1.08039216e+00, 1.17843137e+00,\
                          1.27647059e+00, 1.37450980e+00, 1.47254902e+00, 1.57058824e+00,\
                          1.66862745e+00, 1.76666667e+00, 1.86470588e+00, 1.96274510e+00,\
                          2.06078431e+00, 2.15882353e+00, 2.25686275e+00, 2.35490196e+00,\
                          2.45294118e+00, 2.55098039e+00, 2.64901961e+00, 2.74705882e+00,\
                          2.84509804e+00, 2.94313725e+00, 3.04117647e+00, 3.13921569e+00,\
                          3.23725490e+00, 3.33529412e+00, 3.43333333e+00, 3.53137255e+00,\
                          3.62941176e+00, 3.72745098e+00, 3.82549020e+00, 3.92352941e+00,\
                          4.02156863e+00, 4.11960784e+00, 4.21764706e+00, 4.31568627e+00,\
                          4.41372549e+00, 4.51176471e+00, 4.60980392e+00, 4.70784314e+00,\
                          4.80588235e+00, 4.90392157e+00, 5.00196078e+00],\
                         [2.73972603e-03, 4.93150685e-03, 9.31506849e-03, 1.80821918e-02,\
                          2.90410959e-02, 4.21917808e-02, 5.53424658e-02, 6.84931507e-02,\
                          8.82191781e-02, 1.03561644e-01, 1.23287671e-01, 1.38630137e-01,\
                          1.58356164e-01, 1.80273973e-01, 2.00000000e-01, 2.21917808e-01,\
                          2.48219178e-01, 2.72328767e-01, 2.96438356e-01, 3.20547945e-01,\
                          3.44657534e-01, 3.73150685e-01, 3.99452055e-01, 4.23561644e-01,\
                          4.49863014e-01, 4.76164384e-01, 5.02465753e-01, 5.26575342e-01,\
                          5.52876712e-01, 5.76986301e-01, 6.03287671e-01, 6.27397260e-01,\
                          6.53698630e-01, 6.75616438e-01, 6.99726027e-01, 7.23835616e-01,\
                          7.47945205e-01, 7.69863014e-01, 7.93972603e-01, 8.13698630e-01,\
                          8.33424658e-01, 8.55342466e-01, 8.72876712e-01, 8.92602740e-01,\
                          9.10136986e-01, 9.29863014e-01, 9.47397260e-01, 9.62739726e-01,\
                          9.75890411e-01, 9.91232877e-01, 1.00000000e+00],\
                         [1.00000000e-03, 2.09008229e-03, 4.23222335e-03, 1.29677880e-02,\
                          2.17074761e-02, 3.70385261e-02, 4.79739603e-02, 6.77006260e-02,\
                          8.30296142e-02, 9.83606642e-02, 1.18085268e-01, 1.37813996e-01,\
                          1.59740531e-01, 1.79465135e-01, 2.01389608e-01, 2.21116274e-01,\
                          2.45238555e-01, 2.71560706e-01, 2.95678864e-01, 3.21994830e-01,\
                          3.48316981e-01, 3.74639132e-01, 4.00959221e-01, 4.29477119e-01,\
                          4.57995016e-01, 4.86512913e-01, 5.12833002e-01, 5.41348838e-01,\
                          5.67670989e-01, 5.96186824e-01, 6.22506913e-01, 6.46629195e-01,\
                          6.75147092e-01, 6.99269373e-01, 7.21191785e-01, 7.47511874e-01,\
                          7.69436348e-01, 7.93558629e-01, 8.13285295e-01, 8.35209768e-01,\
                          8.54936434e-01, 8.72465292e-01, 8.94389766e-01, 9.11916562e-01,\
                          9.29445419e-01, 9.44776469e-01, 9.60109581e-01, 9.73238700e-01,\
                          9.86374004e-01, 9.95111630e-01, 9.99472196e-01],\
                         [5.47945205e-04, 5.47945205e-04, 2.73972603e-03, 2.73972603e-03,\
                          9.31506849e-03, 1.58904110e-02, 2.46575342e-02, 3.56164384e-02,\
                          5.09589041e-02, 6.84931507e-02, 8.82191781e-02, 1.10136986e-01,\
                          1.32054795e-01, 1.56164384e-01, 1.82465753e-01, 2.08767123e-01,\
                          2.32876712e-01, 2.63561644e-01, 2.89863014e-01, 3.18356164e-01,\
                          3.49041096e-01, 3.79726027e-01, 4.12602740e-01, 4.43287671e-01,\
                          4.71780822e-01, 5.02465753e-01, 5.37534247e-01, 5.66027397e-01,\
                          5.98904110e-01, 6.25205479e-01, 6.55890411e-01, 6.82191781e-01,\
                          7.10684932e-01, 7.39178082e-01, 7.63287671e-01, 7.85205479e-01,\
                          8.11506849e-01, 8.31232877e-01, 8.55342466e-01, 8.75068493e-01,\
                          8.94794521e-01, 9.14520548e-01, 9.29863014e-01, 9.47397260e-01,\
                          9.62739726e-01, 9.73698630e-01, 9.84657534e-01, 9.91232877e-01,\
                          9.97808219e-01, 1.00000000e+00, 9.97808219e-01],\
                         [5.47945205e-04, 9.31506849e-03, 1.80821918e-02, 3.12328767e-02,\
                          4.87671233e-02, 6.84931507e-02, 9.47945205e-02, 1.16712329e-01,\
                          1.45205479e-01, 1.69315068e-01, 1.97808219e-01, 2.26301370e-01,\
                          2.54794521e-01, 2.85479452e-01, 3.13972603e-01, 3.42465753e-01,\
                          3.73150685e-01, 4.03835616e-01, 4.32328767e-01, 4.60821918e-01,\
                          4.89315068e-01, 5.15616438e-01, 5.44109589e-01, 5.70410959e-01,\
                          5.94520548e-01, 6.20821918e-01, 6.44931507e-01, 6.66849315e-01,\
                          6.93150685e-01, 7.17260274e-01, 7.41369863e-01, 7.63287671e-01,\
                          7.85205479e-01, 8.04931507e-01, 8.26849315e-01, 8.48767123e-01,\
                          8.70684932e-01, 8.90410959e-01, 9.12328767e-01, 9.25479452e-01,\
                          9.43013699e-01, 9.60547945e-01, 9.73698630e-01, 9.80273973e-01,\
                          9.91232877e-01, 9.95616438e-01, 1.00219178e+00, 1.00000000e+00,\
                          9.95616438e-01, 9.86849315e-01, 9.75890411e-01],\
                         [4.98451343e-03, 2.91476729e-02, 5.55026188e-02, 8.84329241e-02,\
                          1.19171443e-01, 1.54293535e-01, 1.91607413e-01, 2.26727447e-01,\
                          2.59661868e-01, 2.96975746e-01, 3.32097838e-01, 3.67219930e-01,\
                          4.02344080e-01, 4.35270269e-01, 4.68204690e-01, 5.01132938e-01,\
                          5.31871456e-01, 5.60418189e-01, 5.95542339e-01, 6.26280857e-01,\
                          6.48250172e-01, 6.76798963e-01, 7.00962122e-01, 7.22933495e-01,\
                          7.49286383e-01, 7.69065970e-01, 7.91041459e-01, 8.15202560e-01,\
                          8.34982147e-01, 8.52569947e-01, 8.70159805e-01, 8.89939391e-01,\
                          9.00953890e-01, 9.18539632e-01, 9.29554131e-01, 9.42754242e-01,\
                          9.49387226e-01, 9.62591453e-01, 9.71412107e-01, 9.78040975e-01,\
                          9.84669843e-01, 9.91296653e-01, 9.95735792e-01, 9.95789300e-01,\
                          1.00022638e+00, 1.00466346e+00, 1.00033340e+00, 1.00038691e+00,\
                          9.96056842e-01, 9.89534991e-01, 9.80804890e-01]])
        

        x = data[0]
        y = np.arange(130,180,10)
        z = data[1:]
        
        finterp_Vin = interpolate.interp2d(x,y,z)
            
    if type(finterp_freq).__name__ != 'interp1d':
        data = np.array([[130., 130.96550379, 131.91908992, 132.93386782, 133.94873239,
                          134.93317443, 135.94829902, 136.96359697, 137.9483857 ,
                          138.93326111, 139.94829902, 140.96359697, 141.94873239,
                          142.9339545 , 143.94985915, 144.9648104 , 145.94899242,
                          146.9336078 , 147.94907909, 148.96507042, 149.98045504,
                          150.9648104 , 151.94933911, 152.96524377, 153.95011918,
                          154.96611051, 155.95089924, 156.93603467, 157.98192849,
                          158.96593716, 159.98166847, 160.96689057, 161.98279523,
                          162.96819068, 163.98452871, 164.94019502, 165.98877573,
                          166.97434453, 167.95826652, 169.00424702, 169.95748646, 170.],
                         [ 11.15492958, 11.15492958,  11.23943662,  11.43661972,  11.6056338 ,
                           11.66197183,  11.74647887,  11.77464789,  11.71830986,
                           11.63380282,  11.74647887,  11.77464789,  11.6056338 ,
                           11.4084507 ,  11.23943662,  11.38028169,  11.52112676,
                           11.52112676,  11.49295775,  11.29577465,  11.29577465,
                           11.38028169,  11.4084507 ,  11.23943662,  11.15492958,
                           10.95774648,  10.90140845,  10.73239437,  10.81690141,
                           11.01408451,  10.90140845,  10.70422535,  10.53521127,
                           10.28169014,   9.97183099,   9.38028169,   8.5915493 ,
                           8.28169014,   8.50704225,   8.56338028,   8.76056338, 8.76056338]])

        finterp_freq = interpolate.interp1d(data[0], data[1])

    # Here we treat the two cases in which Vin is a scalar or an array
    if hasattr(Vin, '__len__') == False:
        result =  finterp_freq(freq)*finterp_Vin(Vin, freq)
        result = result[0]
    else:
        result = []
        for V in Vin:
            dummy = finterp_freq(freq)*finterp_Vin(V, freq)
            result.append(dummy[0])
        
    return np.array(result)




def sim_generator_power(time, amplitude, offset, frequency, phase, rf_freq = 150., finterp_Vin = '', finterp_freq = ''):

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
    finterp_Vin  - MIXED - the interpolating function that returns the fraction of power
                           as a function of Vin. If it is provided externally then it is of type
                           scipy.interpolate.interp2d. Default is an empty string, in this case the interpolation
                           is carried out internally
    finterp_freq - MIXED - the interpolating function that returns the maximum power
                           as a function of the frequency in GHz. If it is provided externally then it is of type
                           scipy.interpolate.interp1d. Default is an empty string, in this case the interpolation
                           is carried out internally

    OUTPUTS
    cal_signal  - FLOAT - the calibrated signal
                          
    '''

    if type(time) == float:
        tim = np.array([time])
    else:
        tim = time

    sine = amplitude/2. * np.sin(2.*np.pi * frequency * tim + phase) # Divide by 2 because we provide the p2p
    uncal_signal = sine + offset
    #cal_signal = np.array([source_cal(i) for i in uncal_signal])
    cal_signal = source_cal(uncal_signal, freq = rf_freq, finterp_Vin = finterp_Vin, finterp_freq = finterp_freq)
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

