import numpy as np
from pyoperators import *
from pysimulators import *


def join_toward_rank(comm, data, target_rank):
    # print('enter', target_rank)
    gathered_data = comm.gather(data, root=target_rank)
    # print('bis')
    if comm.Get_rank() == target_rank:
        # print(' bis bis')
        return np.concatenate(gathered_data)  # [0]
    else:
        return


def join_data(comm, data):

    if comm is None:
        pass
    else:
        data = comm.gather(data, root=0)

        if comm.Get_rank() == 0:

            data = np.concatenate(data)

        data = comm.bcast(data)

    return data


def split_data(comm, theta):
    if comm is None:
        return theta
    else:
        return np.array_split(theta, comm.Get_size())[comm.Get_rank()]
