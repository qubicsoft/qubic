import numpy as np


def join_toward_rank(comm, data, target_rank):
    gathered_data = comm.gather(data, root=target_rank)
    if comm.Get_rank() == target_rank:
        return np.concatenate(gathered_data)
    else:
        return


class MpiTools:
    def __init__(self, comm):
        self.comm = comm

    def _print_message(self, message):
        if self.comm is not None:
            if self.comm.Get_rank() == 0:
                print(message)

            self.comm.Barrier()

    def _barrier(self):
        """
        Method to introduce comm._Barrier() function if MPI communicator is detected.

        """
        if self.comm is None:
            pass
        else:
            self.comm.Barrier()

    def get_random_value(self, init_seed=None):
        """Random value

        Method to build a random seed.

        Returns
        -------
        seed: int
            Random seed.

        """

        np.random.seed(init_seed)
        if self.comm.Get_rank() == 0:
            seed = np.random.randint(10000000)
        else:
            seed = None

        seed = self.comm.bcast(seed, root=0)
        return seed

    def bcast(self, arr):
        if self.comm.Get_rank() == 0:
            d = arr.copy()
        else:
            d = None

        d = self.comm.bcast(d, root=0)

        return d


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
