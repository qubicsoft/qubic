import numpy as np
import os
from PIL import Image
import os

def create_folder_if_not_exists(comm, folder_name):
    # Check if the folder exists
    if comm is not None:
        if comm.Get_rank() == 0:
            if not os.path.exists(folder_name):
                try:
                    # Create the folder if it doesn't exist
                    os.makedirs(folder_name)
                    #print(f"The folder '{folder_name}' has been created.")
                except OSError as e:
                    pass
                    #print(f"Error creating the folder '{folder_name}': {e}")
            else:
                pass
    else:
        if not os.path.exists(folder_name):
            try:
                # Create the folder if it doesn't exist
                os.makedirs(folder_name)
                #print(f"The folder '{folder_name}' has been created.")
            except OSError as e:
                pass
                #print(f"Error creating the folder '{folder_name}': {e}")
        else:
            pass
