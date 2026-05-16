from typing import List
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    a: float
    b: float
    c: float
    sigma_tau: float
    nu: int
    chi_square_reduced: float
    p_value: float
    slope: float
    intercept: float
    slope_sigma: float
    slope_p_value: float
    residuals: List[float] = field(default_factory=list)
    pcov: List[List[float]] = field(default_factory=list)

    index: tuple[int, int] | None = None

    @property
    def tau(self) -> float:
        return round(-1. / self.c, 6)

    def is_valid(self, tau_max: float = 1., sigma_max: float = 0.1, p_value_max: float = 0.05) -> bool:
        if self.c >= 0:
            return False
        if self.tau > tau_max:
            return False
        if self.sigma_tau > sigma_max:
            return False
        if self.p_value <= p_value_max:
            return False

        return True

#TODO: from this point onwards, to be implemented in the next version
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

        return float(np.mean(sigs) / len(sigs) ** 0.5)


@dataclass
class MDTReport:

    entries: dict[int, MDTReportEntry] = field(default_factory=dict)

    def add_event(self,
                  tes: int,
                  tau: float,
                  sigma: float,
                  energy: float,
                  elevation: float,
                  duration: float,
                  residuals: List[float]) -> None:


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
