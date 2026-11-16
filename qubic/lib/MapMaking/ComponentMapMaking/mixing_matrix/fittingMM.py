from abc import ABC, abstractmethod

import numpy as np


class FittingMM(ABC):
    def __init__(self, selfCMM):
        self.selfCMM = selfCMM
        self.preset = selfCMM.preset
        self.plots = selfCMM.plots
        self._steps = selfCMM._steps
        self.nfev = 0
        self.chi2 = None

    @abstractmethod
    def update(self, tod_comp):
        raise NotImplementedError

    def callback(self, x):
        """Common callback for scipy optimizers."""
        self.preset.tools.comm.Barrier()

        if self.preset.tools.rank == 0:
            if self.nfev % 1 == 0:
                print(f"Iter = {self.nfev:4d}   x = {[np.round(v, 5) for v in x]}   qubic log(L) = {np.log(np.round(self.chi2.Lqubic, 5))}   planck log(L) = {np.log(np.round(self.chi2.Lplanck, 5))}")
            self.nfev += 1
