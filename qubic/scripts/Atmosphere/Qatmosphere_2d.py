import os
import os.path as op
import sys
 
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import scipy
import scipy.ndimage
from scipy.stats import linregress
import healpy as hp
import jdcal as jdcal
import random                                                                                               
import time

from multiprocessing import Pool
from getdist import plots, MCSamples
import getdist
import time

class Atmsophere:
    
    def __init__(self, params):
        
        self.params = params
