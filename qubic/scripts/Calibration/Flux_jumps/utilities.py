import numpy as np
import matplotlib.pyplot as plt
import jumps_review as jr
import pickle
import json
import os
from datetime import datetime



# ================================================================================
# Plotting functions 
# ================================================================================

def plot_no_jumps(tt, todarray, results): 
        
    # ============================================================================
    # plot TES without jumps
    # ============================================================================

    """    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    TES_no = results['TES_no']
    
    if len(TES_no) > 0:
        n_plot = len(TES_no)
        n_cols = 5
        n_rows = int(np.ceil(n_plot / n_cols))
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()
        
        for i, idx in enumerate(TES_no[:n_plot]):
            ax[i].plot(tt/60, todarray[idx], 'b-', linewidth=0.5)
            ax[i].set_title(f'TES {idx+1} (No jumps)', fontsize=10)
            ax[i].set_xlabel('Time (min)')
            ax[i].set_ylabel('Flux')
            ax[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plot, len(ax)):
            ax[i].axis('off')
        
        plt.suptitle(f'TES without Flux Jumps (showing {n_plot} of {len(TES_no)})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    return 0

def plot_jump_detections(tt, todarray, results, DT=True):


    # ============================================================================
    # Plot TES with jumps
    # ============================================================================

    """    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    
    TES_yes = results['TES_yes']
    if DT: 
        jump_data = results['dt_jump_data']        
    else:
        jump_data = results['jump_data']

    if len(TES_yes) > 0:
        n_plot = len(TES_yes)
        n_cols = 4
        n_rows = int(np.ceil(n_plot / n_cols))
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()
        
        plot_idx = 0
        for idx in TES_yes[:n_plot]:
            if idx in jump_data:
                # Original data with jump markers
                ax[plot_idx].plot(tt/60, todarray[idx], 'b-', linewidth=0.5, 
                                 label='Original', alpha=0.7)
                if DT:
                    xc = jump_data[idx]['xcdt']
                    xcf = jump_data[idx]['xcfdt']
                    nc = jump_data[idx]['ncdt']
                else: 
                    xc = jump_data[idx]['xc']
                    xcf = jump_data[idx]['xcf']
                    nc = jump_data[idx]['nc']
                if len(xc) > 0:
                    ax[plot_idx].scatter(tt[xc]/60, todarray[idx][xc], 
                                        color='red', marker='o', s=30, 
                                        label='Jump start', zorder=5)
                    ax[plot_idx].scatter(tt[xcf]/60, todarray[idx][xcf], 
                                        color='green', marker='s', s=30, 
                                        label='Jump end', zorder=5)
                ax[plot_idx].set_title(f'TES {idx+1} ({nc} jumps)', 
                                       fontsize=10)
                ax[plot_idx].set_xlabel('Time (min)')
                ax[plot_idx].set_ylabel('Flux')
                ax[plot_idx].legend(fontsize=8)
                ax[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(ax)):
            ax[i].axis('off')
        
        plt.suptitle(f'TES with Flux Jumps (showing {n_plot} of {len(TES_yes)})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    return 0
    
def plot_corrections(tt, todarray, results):


    # ============================================================================
    # Plot TES with jumps corrected
    # ============================================================================

    """    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis

    """

    
    TES_yes = results['TES_yes']
    corrected_data = results['corrected_data']
    
    if len(TES_yes) > 0:
        n_plot = len(TES_yes)
        n_cols = 4
        n_rows = int(np.ceil(n_plot / n_cols))
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            ax = ax.reshape(1, -1)
        ax = ax.flatten()
        
        plot_idx = 0
        for idx in TES_yes[:n_plot]:
                
                # Corrected data
            if idx in corrected_data:
                ax[plot_idx].plot(tt/60, corrected_data[idx], 'r-', 
                            linewidth=0.8, label='Corrected', alpha=0.8)
                
                ax[plot_idx].set_title(f'TES {idx+1}', 
                                       fontsize=10)
                ax[plot_idx].set_xlabel('Time (min)')
                ax[plot_idx].set_ylabel('Flux')
                ax[plot_idx].legend(fontsize=8)
                ax[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(ax)):
            ax[i].axis('off')
        
        plt.suptitle(f'TES with Flux Jumps Corrected (showing {n_plot} of {len(TES_yes)})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return 0



# ================================================================================
# saving functions 
# ================================================================================

def save_results(results, output_dir="./results_15_26_08", 
                 save_format="pickle", dataset_name="15.26.08"):
    """
    Save analysis results to disk in pickle format.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyzed dataset
    todarray : np.ndarray
        Original time-ordered data array
    tt : np.ndarray
        Time axis
    output_dir : str
        Directory where to save results (default: "./results_15_26_08")
    save_format : str
        Format to save: pickle"
    dataset_name : str
        Name of the dataset for file naming (default: "15.26.08")
        
    Returns:
    --------
    saved_files : list
        List of paths to saved files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = f"{dataset_name}"
    saved_files = []

    
    # ============================================================================
    # Save format: Pickle format (.pkl) 
    # ============================================================================
    if save_format in ["pickle"]:
        pkl_file = os.path.join(output_dir, f"{base_name}_results.pkl")
        
        # Prepare complete results dictionary
        save_data = {
            'results': results,
            'dataset_name': dataset_name
        }
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        saved_files.append(pkl_file)
        print(f"Saved pickle file to: {pkl_file}")
    
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"{'='*70}\n")
    
    return saved_files


def load_results(load_file, load_format="pickle"):
    """
    Load previously saved analysis results.
    
    Parameters:
    -----------
    load_file : str
        Path to the saved file
    load_format : str
        Format type: "pickle" (default: "pickle")
        
    Returns:
    --------
    loaded_data : dict
        Loaded results dictionary
    """
    
    if load_format == "pickle":
        with open(load_file, 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data
    
    else:
        raise ValueError(f"Unsupported load format: {load_format}")

