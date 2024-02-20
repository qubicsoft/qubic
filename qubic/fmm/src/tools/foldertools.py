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




def do_gif(input_folder, N, jobid):

    nmaps = np.arange(1, N+1, 1)
    image_list = []
    #for filename in sorted(os.listdir(input_folder)):
    for n in nmaps:
        image_path = os.path.join(input_folder, f'maps_{n}.png')
        image = Image.open(image_path)
        image_list.append(image)

    output_gif_path = f"gif_convergence_{jobid}/animation.gif"

    image_list[0].save(output_gif_path, save_all=True, append_images=image_list[1:], duration=100, loop=0)
