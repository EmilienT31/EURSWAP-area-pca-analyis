"""Continuously compounded zero curves and par-swap bootstrapping.

Model time is measured in ACT/365F years. Discount factors are interpolated
log-linearly, which is equivalent to piecewise-constant instantaneous forwards.
The first implementation supports regular annual swap payments and maturities
that are exact multiples of the payment interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import brentq


BASIS_POINT = 1.0e-4


def bp_to_decimal(basis_points: float) -> float:
    """Convert basis points into decimal rates (1 bp = 0.0001)."""

    return float(basis_points) * BASIS_POINT


def payment_times(maturity_years: float, payment_frequency: int = 1) -> np.ndarray:
    """Return a regular payment grid under the stylised ACT/365F convention."""

    if not np.isfinite(maturity_years) or maturity_years <= 0:
        raise ValueError("Swap maturity must be a positive finite year fraction.")
    if payment_frequency <= 0:
        raise ValueError("Payment frequency must be positive.")
    payments = maturity_years * payment_frequency
    count = int(round(payments))
    if count < 1 or not np.isclose(payments, count, atol=1.0e-10):
        raise ValueError(
            "Maturity must be an exact multiple of the regular payment interval."
        )
    return np.arange(1, count + 1, dtype=float) / payment_frequency


@dataclass(frozen=True)
class YieldCurve:
    """Zero curve with continuously compounded decimal rates.

    Parameters are node times in ACT/365F years and continuously compounded
    annual zero rates in decimal form. Interpolation is performed on log
    discount factors and extrapolation beyond the final node is deliberately
    disallowed.
    """

    times: np.ndarray
    zero_rates_cc: np.ndarray
    day_count: str = "ACT/365F"
    interpolation: str = "log-linear discount factors"

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        rates = np.asarray(self.zero_rates_cc, dtype=float)
        if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates):
            raise ValueError("Curve times and zero rates must be equal-length vectors.")
        if len(times) == 0 or not np.all(np.isfinite(times)):
            raise ValueError("Curve needs at least one finite node.")
        if not np.all(np.isfinite(rates)):
            raise ValueError("Curve zero rates must be finite.")
        if np.any(times <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("Curve times must be strictly increasing and positive.")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "zero_rates_cc", rates.copy())

    @property
    def log_discount_factors(self) -> np.ndarray:
        """Log discount factors at the curve nodes."""

        return -self.times * self.zero_rates_cc

    def _validated_query(self, times: float | Iterable[float]) -> tuple[np.ndarray, bool]:
        query = np.asarray(times, dtype=float)
        scalar = query.ndim == 0
        query = np.atleast_1d(query)
        if not np.all(np.isfinite(query)) or np.any(query < 0):
            raise ValueError("Requested curve times must be finite and non-negative.")
        if np.any(query > self.times[-1] + 1.0e-12):
            raise ValueError(
                f"Requested maturity exceeds final curve node {self.times[-1]:g}Y."
            )
        return query, scalar

    def log_discount_factor(self, times: float | Iterable[float]):
        """Interpolate log discount factors, including the anchor D(0)=1."""

        query, scalar = self._validated_query(times)
        node_times = np.concatenate(([0.0], self.times))
        node_values = np.concatenate(([0.0], self.log_discount_factors))
        values = np.interp(query, node_times, node_values)
        return float(values[0]) if scalar else values

    def discount_factor(self, times: float | Iterable[float]):
        """Return discount factors at one or more model times."""

        values = np.exp(self.log_discount_factor(times))
        return float(values) if np.ndim(values) == 0 else values

    def zero_rate(self, times: float | Iterable[float]):
        """Return continuously compounded decimal zero rates."""

        query, scalar = self._validated_query(times)
        logs = np.asarray(self.log_discount_factor(query))
        values = np.empty_like(query)
        positive = query > 0
        values[positive] = -logs[positive] / query[positive]
        values[~positive] = self.zero_rates_cc[0]
        return float(values[0]) if scalar else values

    def forward_rate(self, start: float, end: float) -> float:
        """Return the continuously compounded forward rate over ``[start, end]``."""

        if end <= start:
            raise ValueError("Forward-rate end time must be after start time.")
        return (
            self.log_discount_factor(start) - self.log_discount_factor(end)
        ) / (end - start)

    def par_swap_rate(self, maturity_years: float, payment_frequency: int = 1) -> float:
        """Return the annualised par fixed rate for a spot-starting single-curve swap."""

        schedule = payment_times(maturity_years, payment_frequency)
        accrual = 1.0 / payment_frequency
        annuity = accrual * float(np.sum(self.discount_factor(schedule)))
        return (1.0 - self.discount_factor(maturity_years)) / annuity

    def parallel_bump(self, bump_bp: float) -> "YieldCurve":
        """Shift every continuously compounded zero node by ``bump_bp``."""

        return YieldCurve(
            self.times,
            self.zero_rates_cc + bp_to_decimal(bump_bp),
            self.day_count,
            self.interpolation,
        )

    def key_rate_bump(
        self,
        key_time: float,
        bump_bp: float,
        key_times: Iterable[float],
    ) -> "YieldCurve":
        """Apply one triangular key-rate bump to zero rates.

        The hat functions form a partition of unity: outside the first/last key
        the nearest endpoint receives full weight, and between adjacent keys the
        two weights sum to one. Consequently, applying all key bumps by the same
        amount reproduces a parallel node bump.
        """

        keys = np.asarray(tuple(key_times), dtype=float)
        if keys.ndim != 1 or len(keys) == 0 or np.any(np.diff(keys) <= 0):
            raise ValueError("Key maturities must be a non-empty increasing vector.")
        matches = np.flatnonzero(np.isclose(keys, key_time, atol=1.0e-12))
        if len(matches) != 1:
            raise ValueError(f"Key maturity {key_time:g}Y is not in the key grid.")
        index = int(matches[0])
        weights = np.zeros_like(self.times)
        if index == 0:
            weights[self.times <= keys[0]] = 1.0
        else:
            mask = (self.times >= keys[index - 1]) & (self.times <= keys[index])
            weights[mask] = (self.times[mask] - keys[index - 1]) / (
                keys[index] - keys[index - 1]
            )
        if index == len(keys) - 1:
            weights[self.times >= keys[-1]] = 1.0
        else:
            mask = (self.times >= keys[index]) & (self.times <= keys[index + 1])
            weights[mask] = (keys[index + 1] - self.times[mask]) / (
                keys[index + 1] - keys[index]
            )
        return YieldCurve(
            self.times,
            self.zero_rates_cc + bp_to_decimal(bump_bp) * weights,
            self.day_count,
            self.interpolation,
        )


def bootstrap_par_swap_curve(
    maturities: Iterable[float],
    par_rates: Iterable[float],
    payment_frequency: int = 1,
) -> YieldCurve:
    """Bootstrap a log-DF curve from decimal par swap quotations.

    A scalar root is solved for the continuously compounded zero rate at each
    successive par maturity. Earlier discount nodes are held fixed and log
    discount factors are interpolated between solved nodes.
    """

    terms = np.asarray(tuple(maturities), dtype=float)
    quotes = np.asarray(tuple(par_rates), dtype=float)
    if terms.ndim != 1 or len(terms) == 0 or len(terms) != len(quotes):
        raise ValueError("Par maturities and quotes must be equal non-empty vectors.")
    if np.any(np.diff(terms) <= 0) or not np.all(np.isfinite(quotes)):
        raise ValueError("Par maturities must increase and quotes must be finite.")
    for maturity in terms:
        payment_times(float(maturity), payment_frequency)

    solved_times: list[float] = []
    solved_rates: list[float] = []
    for maturity, quote in zip(terms, quotes, strict=True):
        def objective(candidate_rate: float) -> float:
            curve = YieldCurve(
                np.asarray([*solved_times, maturity]),
                np.asarray([*solved_rates, candidate_rate]),
            )
            return curve.par_swap_rate(float(maturity), payment_frequency) - quote

        solved = brentq(objective, -1.0, 1.0, xtol=1.0e-14, rtol=1.0e-14)
        solved_times.append(float(maturity))
        solved_rates.append(float(solved))
    return YieldCurve(np.asarray(solved_times), np.asarray(solved_rates))
