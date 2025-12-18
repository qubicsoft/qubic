import os
import pickle
from io import BytesIO
from pathlib import Path

import cairosvg
import imageio
import numpy as np
import yaml


def yaml_to_txt(yaml_file, txt_file, comm=None):
    """
    Convert a YAML file to a TXT file.
    """

    splitted_path = os.path.split(txt_file)

    ### Create the path if doesn't exist
    create_folder_if_not_exists(comm=comm, folder_name=splitted_path[0])

    try:
        with open(yaml_file, "r") as yf:
            yaml_data = yaml.safe_load(yf)

        with open(txt_file, "w") as tf:
            yaml.dump(yaml_data, tf, default_flow_style=False, sort_keys=False)

        print(f"Successfully converted {yaml_file} to {txt_file}")
    except Exception as e:
        print(f"Error converting YAML to TXT: {str(e)}")


def save_data(name, d):
    """

    Method to save data using pickle convention.

    """

    with open(name, "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_data(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def create_folder_if_not_exists(comm, folder_name):
    # Check if the folder exists
    if comm is not None:
        if comm.Get_rank() == 0:
            if not os.path.exists(folder_name):
                try:
                    # Create the folder if it doesn't exist
                    os.makedirs(folder_name)
                    # print(f"The folder '{folder_name}' has been created.")
                except OSError:
                    pass
                    # print(f"Error creating the folder '{folder_name}': {e}")
            else:
                pass
    else:
        if not os.path.exists(folder_name):
            try:
                # Create the folder if it doesn't exist
                os.makedirs(folder_name)
                # print(f"The folder '{folder_name}' has been created.")
            except OSError:
                pass
                # print(f"Error creating the folder '{folder_name}': {e}")
        else:
            pass


def do_gif(svg_folder, output="animation.gif", fps=5):
    svg_files = sorted(Path(svg_folder).glob("*.svg"))
    images = []

    for svg_path in svg_files:
        # convert SVG to PNG bytes in memory
        png_bytes = cairosvg.svg2png(url=str(svg_path))
        # read PNG into numpy array
        images.append(imageio.imread(BytesIO(png_bytes)))

    imageio.mimsave(svg_folder + output, images, fps=fps)
    print(f"GIF saved at {os.path.join(svg_folder, output)}")


class MergeAllFiles:
    def __init__(self, foldername):
        self.foldername = foldername

        self.list_files = os.listdir(self.foldername)
        self.number_of_realizations = len(self.list_files)

    def _reads_one_file(self, i, key):
        d = open_data(self.foldername + self.list_files[i])

        return d[key]

    def _reads_all_files(self, key, verbose=False):
        arr = []
        list_not_readed_files = []
        for ireal in range(self.number_of_realizations):
            if verbose:
                print(f"========= Reading realization {ireal} =========")
            try:
                arr.append(self._reads_one_file(ireal, key))
            except Exception as e:
                list_not_readed_files += [ireal]
                if verbose:
                    print(f"Warning: failed to read realization {ireal}: {e}")
        arr = np.array(arr)
        arr = np.delete(arr, list_not_readed_files, axis=0)
        return arr

    def get_frequency_comp(self, i):
        d = open_data(self.foldername + self.list_files[i])["parameters"]

        nus, comp = [], []
        print(d.keys())
        if d["CMB"]["cmb"]:
            nus.append(150)
            comp.append("CMB")
        if d["Foregrounds"]["Dust"]["Dust_out"]:
            nus.append(d["Foregrounds"]["Dust"]["nu0"])
            comp.append("Dust")
        if d["Foregrounds"]["Synchrotron"]["Synchrotron_out"]:
            nus.append(d["Foregrounds"]["Synchrotron"]["nu0"])
            comp.append("Synchrotron")
        if d["Foregrounds"]["CO"]["CO_out"]:
            nus.append(d["Foregrounds"]["CO"]["nu0"])
            comp.append("CO")

        return np.array(nus), np.array(comp)
