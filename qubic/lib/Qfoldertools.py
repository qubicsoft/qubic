import os
import pickle
import yaml
import imageio
import numpy as np

def yaml_to_txt(yaml_file, txt_file, comm=None):
    """
    Convert a YAML file to a TXT file.
    """

    splitted_path = os.path.split(txt_file)
    
    ### Create the path if doesn't exist
    create_folder_if_not_exists(comm=comm, folder_name=splitted_path[0])
    
    try:
        with open(yaml_file, 'r') as yf:
            yaml_data = yaml.safe_load(yf)
        
        with open(txt_file, 'w') as tf:
            yaml.dump(yaml_data, tf, default_flow_style=False)
        
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
                except OSError as e:
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
            except OSError as e:
                pass
                # print(f"Error creating the folder '{folder_name}': {e}")
        else:
            pass


def do_gif(input_folder, filename, output="animation.gif", duration=0.01):

    # Collect all the file paths for the images
    file_paths = []

    for n in sorted(os.listdir(input_folder)):
        if n.startswith(filename) and n.endswith(".png"):
            file_paths.append(os.path.join(input_folder, n))

    # Ensure the file_paths are sorted by the numerical part of the filenames
    file_paths = sorted(
        file_paths, key=lambda x: int(x.split(filename)[-1].split(".png")[0])
    )

    # Create the GIF
    with imageio.get_writer(
        os.path.join(input_folder, output), mode="I", duration=duration
    ) as writer:
        for file_path in file_paths:
            image = imageio.imread(file_path)
            writer.append_data(image)

    print(f"GIF saved at {os.path.join(input_folder, output)}")


class MergeAllFiles:

    def __init__(self, foldername):

        self.foldername = foldername

        self.list_files = os.listdir(self.foldername)
        self.number_of_realizations = len(self.list_files)

    def _reads_one_file(self, i, key):

        d = open_data(self.foldername + self.list_files[i])

        return d[key]

    def _reads_all_files(self, key):

        arr = np.zeros(
            (self.number_of_realizations,) + self._reads_one_file(0, key).shape
        )

        for ireal in range(self.number_of_realizations):
            arr[ireal] = self._reads_one_file(ireal, key)

        return arr
