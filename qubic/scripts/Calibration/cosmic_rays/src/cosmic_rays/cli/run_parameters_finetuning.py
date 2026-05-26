import os
import sys
from cosmic_rays.crd import Crd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    # if you want to try multiple configurations, set in the config file
    # remove_files = true
    config_file = sys.argv[1]
    
    # Dictionary used to store a score for each tested parameter combination.
    # The key is a tuple of parameters, and the value is a scalar performance metric.
    scores = {}
    
    # Loop over values of the standard deviation multiplicative coefficient used in thresholding
    # for cosmic rays detection
    for std_coeff in [5]:
    
        print(f"{bcolors.HEADER}Std coefficient:{std_coeff}{bcolors.ENDC}")
    
        # Loop over the minimum number of points required to estimate
        # the vertical trend of a candidate event
        for min_pt_vertical in [1, 2, 3]:
    
            print(f"{bcolors.OKCYAN}Min point vertical trend:{min_pt_vertical}{bcolors.ENDC}")
    
            # Loop over the minimum number of points required to fit
            # the exponential decay part of the cosmic ray signal.
            for min_pt_exp in [2, 3, 4, 5, 6]:
    
                print(f"{bcolors.OKGREEN}Min point exp trend:{min_pt_exp}{bcolors.ENDC}")
    
                # Read the configuration file and initialize a CRD object
                # with default parameters defined in the TOML file
                crd = Crd.read_config(config_file)
                # Override the standard deviation coefficient parameter
                # used internally for candidate selection
                crd.std_coeff = std_coeff
                # Set the minimum number of points required to estimate
                # the vertical (linear) trend before the signal peak
                crd.points_vertical_trend = min_pt_vertical
                # Set the minimum number of points required to fit
                # the exponential decay after the signal peak
                crd.points_exp_decrease = min_pt_exp
                crd.dest = os.path.join(crd.dest, str(std_coeff) + str(min_pt_vertical) + str(min_pt_exp))
    
                # Run the cosmic-ray detection algorithm with the current
                # set of parameters and retrieve the list of detected events
                crds = crd.find_cosmic_rays()
    
                # Initialize a scalar score used to quantify the performance
                # of the current parameter combination
                score = 0
                # Loop over all detected cosmic ray events
                for crd in crds:
    
                    # Loop over all TES detectors for which time constants were estimated
                    for n_tes in crd.taus:
    
                        # Increase the score by the number of estimated
                        # time constants (taus) for this TES.
                        # The score therefore counts the total number of
                        # time constants.
                        score += len(crd.taus[n_tes]['taus'])
    
                # Store the score associated with the current parameter
                # combination in the dictionary
                scores[(std_coeff, min_pt_vertical, min_pt_exp)] = score
    
    print(scores)
