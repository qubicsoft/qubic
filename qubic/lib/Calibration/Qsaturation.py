import numpy as np
import sys,os
import glob
from typing import Tuple, Union, Optional



class saturation:

    def __init__(self, sat_value: float = 4.19e6, TES_number: int = 256):
        
        """Class for detecting saturation in Time-Ordered Data (TOD).
        
        This class identifies saturated TES in the focal plane
        by checking if the TOD values exceed a specified threshold.
        
        Parameters:
            sat_value: Limit of the TOD to be considered saturated (default: 4.19e6).
            TES_number: Number of TES in the focal plane (default: 256).
        """
        self.sat_value = sat_value
        self.TES_number = TES_number

    def detect_saturation(self, todarray: np.ndarray, return_time_indices: bool = False): 
        
        """Detect saturated TES in the TOD array.
        
        For each TES, calculates the fraction of time samples that exceed the
        saturation threshold (either above +sat_value or below -sat_value).
        A TES is considered saturated if any portion of its TOD is saturated.
        
        Parameters:
            todarray: Array of shape (TES_number, time_samples) containing the TOD data.
            return_time_indices: If True, also return per-TES sample indices where
                the TOD is saturated (either above +sat_value or below -sat_value).
            
        Returns:
                - is_valid: Boolean array (True for valid, False for saturated TES)
                - saturated_indices: Array of indices of saturated TES
                - saturation_fraction: Array of saturation fractions per TES
                - num_saturated: Number of saturated TES
                - saturated_time_indices (optional): dict mapping TES index -> 1D array
                  of time sample indices where that TES is saturated. Returned only if
                  return_time_indices=True.
                
        Raises:
            ValueError: If todarray shape doesn't match TES_number.
        """
        # Input validation
        if todarray.shape[0] != self.TES_number:
            raise ValueError(
                f"todarray first dimension ({todarray.shape[0]}) must match "
                f"TES_number ({self.TES_number})"
            )
        
        # Vectorized computation of saturation fractions
        upper_satval = self.sat_value
        lower_satval = -self.sat_value
        
        # Create masks for upper and lower saturation
        upper_mask = todarray > upper_satval
        lower_mask = todarray < lower_satval
        sat_mask = upper_mask | lower_mask
        
        # Calculate saturation fraction for each TES (vectorized)
        size = todarray.shape[1]
        saturation_fraction = (np.sum(upper_mask, axis=1) + np.sum(lower_mask, axis=1)) / size
        
        # TES is valid (not saturated) if saturation fraction is zero
        is_valid = saturation_fraction == 0.0
        
        # Get indices of saturated TES
        saturated_indices = np.where(~is_valid)[0]
        num_saturated = len(saturated_indices)

        if not return_time_indices:
            return is_valid, saturated_indices, saturation_fraction, num_saturated

        saturated_time_indices = {
            int(tes): np.where(sat_mask[tes])[0] for tes in saturated_indices
        }
        return (
            is_valid,
            saturated_indices,
            saturation_fraction,
            num_saturated,
            saturated_time_indices,
        )

