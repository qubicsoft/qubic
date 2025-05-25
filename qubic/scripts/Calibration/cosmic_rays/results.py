from typing import List
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    """
    Data classes are regular Python classes primarily designed to store state rather than implement complex logic.
    The FitResult data class defines the attributes related to the fit of a single candidate. Its method is_valid()
    returns True if the fit is valid, and False otherwise.
    """
    

    a: float # offset
    b: float # amplitude
    c: float # negative inverse of the time constant
    sigma_tau: float
    nu: int # number of degrees of freedom
    chi_square_reduced: float
    p_value: float
    slope: float
    slope_sigma: float
    slope_p_value: float
    residuals: list[float] = field(default_factory=list)
    pcov: list[list[float]] = field(default_factory=list)

    index: tuple[int, int] | None = None # start and end indexes of a candidate

    @property
    def tau(self) -> float:
        """
        Method that transforms the tau method into an attribute using the @property decorator.
        This allows for more convenient access to the tau value
        """

        return -1. / self.c

    def is_valid(self, tau_max: float = 1., sigma_max: float = 0.1, p_value_max: float = 0.05) -> bool:
        """
        Checks the validity of a given candidate.

        Parameters
        ----------
        tau_max : float
            maximum tau value allowed

        sigma_max : float
            maximum sigma value allowed

        p_value_max : float
            maximum p-value value allowed

        Returns
        -------
        bool: True if the candidate is valid, False otherwise
        """

        if self.c >= 0:
            return False

        if self.tau > tau_max:
            return False

        if self.sigma_tau > sigma_max:
            return False

        if self.p_value <= p_value_max:
            return False

        return True


@dataclass
class TESResult:
    """
    Groups all FitResult events for a single TES (identified by asic and tes).
    Provides aggregate access to taus, sigmas, chi squares, p-values, and residuals.
    """
    asic: int
    tes: int
    fit_results: list[FitResult] = field(default_factory=list)

    @property
    def taus(self) -> list[float]:
        return [fr.tau for fr in self.fit_results]

    @property
    def sigmas(self) -> list[float]:
        return [fr.sigma_tau for fr in self.fit_results]

    @property
    def chi_squares(self) -> list[float]:
        return [fr.chi_square_reduced for fr in self.fit_results]

    @property
    def p_values(self) -> list[float]:
        return [fr.p_value for fr in self.fit_results]

    @property
    def residuals_list(self) -> list[list[float]]:
        return [fr.residuals for fr in self.fit_results]


@dataclass
class TausResult:
    """
    Holds a mapping of (asic, tes) to TESResult, allowing addition of new FitResults.
    """
    by_tes: dict[int, TESResult] = field(default_factory=dict)

    def add(self, key: int, asic: int, tes: int, fr: FitResult) -> None:
        # key = (asic, tes)
        if key not in self.by_tes:
            self.by_tes[key] = TESResult(asic=asic, tes=tes)
        self.by_tes[key].fit_results.append(fr)


@dataclass
class MDTReportEntry:
    """
    Accumula per un singolo TES:
      - tau e sigma
      - energy, elevation, duration e residuals

    Espone proprietà per il calcolo di tau media e incertezza.
    """
    taus: List[tuple[float, float]] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    elevations: List[float] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)

    @property
    def avg_tau(self) -> float:
        if not self.taus:
            return float('nan')

        taus, _ = zip(*self.taus)
        return float(np.mean(taus))

    @property
    def avg_tau_err(self) -> float:
        if not self.taus:
            return float('nan')

        _, sigs = zip(*self.taus)

        # incertezza sul mean = σ / √N
        return float(np.mean(sigs) / len(sigs) ** 0.5)


@dataclass
class MDTReport:
    """
    Contiene un mapping TES -> MDTReportEntry, con helper per
    aggiungere eventi e calcolare statistiche globali.
    """
    entries: dict[int, MDTReportEntry] = field(default_factory=dict)

    def add_event(self,
                  tes: int,
                  tau: float,
                  sigma: float,
                  energy: float,
                  elevation: float,
                  duration: float,
                  residuals: List[float]) -> None:
        """
        Registra un nuovo evento (un singolo fit) sotto il TES specificato.
        """

        if tes not in self.entries:
            self.entries[tes] = MDTReportEntry()
        e = self.entries[tes]
        e.taus.append((tau, sigma))
        e.energies.append(energy)
        e.elevations.append(elevation)
        e.durations.append(duration)
        e.residuals.extend(residuals)

    @property
    def global_mean_tau(self) -> tuple[float, float]:
        """
        Calcola media globale e incertezza di tutte le taus di tutti i TES.
        """
        all_t = [t for e in self.entries.values() for t in e.taus]
        if not all_t:
            return float('nan'), float('nan')
        taus, sigs = zip(*all_t)
        mean_tau = float(np.mean(taus))
        sigma = float(np.mean(sigs) / len(sigs) ** 0.5)
        return mean_tau, sigma


@dataclass
class MultiStrategyReport:
    per_strategy: dict[str, MDTReport] = field(default_factory=dict)
    global_report: MDTReport = field(default_factory=MDTReport)
