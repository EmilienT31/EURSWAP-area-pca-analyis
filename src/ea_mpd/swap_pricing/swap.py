"""Stylised spot-starting fixed-for-overnight swap valuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .curve import YieldCurve, payment_times


@dataclass(frozen=True)
class OISSwap:
    """Single-curve OIS with annualised fixed coupon in decimal form.

    A payer swap pays fixed and receives compounded overnight. With no payment
    lags and one projection/discount curve, the floating-leg PV is
    ``notional * (1 - D(T))``.
    """

    notional: float
    fixed_rate: float
    maturity_years: float
    side: str = "payer"
    payment_frequency: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.notional) or self.notional <= 0:
            raise ValueError("Swap notional must be positive and finite.")
        if not np.isfinite(self.fixed_rate):
            raise ValueError("Swap fixed rate must be finite.")
        if self.side not in {"payer", "receiver"}:
            raise ValueError("Swap side must be 'payer' or 'receiver'.")
        payment_times(self.maturity_years, self.payment_frequency)

    def annuity(self, curve: YieldCurve) -> float:
        schedule = payment_times(self.maturity_years, self.payment_frequency)
        return (1.0 / self.payment_frequency) * float(
            np.sum(curve.discount_factor(schedule))
        )

    def fixed_leg_pv(self, curve: YieldCurve) -> float:
        """Positive present value of fixed coupons, before payer/receiver sign."""

        return self.notional * self.fixed_rate * self.annuity(curve)

    def floating_leg_pv(self, curve: YieldCurve) -> float:
        """Positive present value of the compounded overnight floating leg."""

        return self.notional * (1.0 - curve.discount_factor(self.maturity_years))

    def value(self, curve: YieldCurve) -> float:
        """Signed swap PV; payer means receive floating less pay fixed."""

        payer_value = self.floating_leg_pv(curve) - self.fixed_leg_pv(curve)
        return payer_value if self.side == "payer" else -payer_value

    @classmethod
    def at_par(
        cls,
        curve: YieldCurve,
        notional: float,
        maturity_years: float,
        side: str = "payer",
        payment_frequency: int = 1,
    ) -> "OISSwap":
        rate = curve.par_swap_rate(maturity_years, payment_frequency)
        return cls(notional, rate, maturity_years, side, payment_frequency)


def par_rate_annuity_sensitivity(
    swap: OISSwap, curve: YieldCurve, per_bp: bool = False
) -> float:
    """Return signed sensitivity to the swap's own par quote.

    At inception ``PV = direction * N * A * (S - K)``. Holding the annuity
    fixed, ``dPV/dS = direction * N * A`` where payer direction is positive.
    Set ``per_bp`` to express the result in EUR for a one-basis-point change in
    that par quote instead of EUR per unit decimal rate.
    """

    direction = 1.0 if swap.side == "payer" else -1.0
    sensitivity = direction * swap.notional * swap.annuity(curve)
    return sensitivity * 1.0e-4 if per_bp else sensitivity


def parallel_dv01(swap: OISSwap, curve: YieldCurve, bump_bp: float = 1.0) -> float:
    """Central first-order PV change for an upward parallel zero-rate bump.

    With the default one-basis-point bump, the return value is EUR per +1 bp.
    A par payer swap normally has a positive value under this sign convention.
    """

    if bump_bp <= 0:
        raise ValueError("DV01 bump must be positive.")
    up = swap.value(curve.parallel_bump(bump_bp))
    down = swap.value(curve.parallel_bump(-bump_bp))
    return (up - down) / 2.0


def key_rate_sensitivities(
    swap: OISSwap,
    curve: YieldCurve,
    key_times: Iterable[float],
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """Central EUR PV changes for triangular +1 bp zero key-rate bumps."""

    keys = tuple(float(value) for value in key_times)
    rows: list[dict[str, float]] = []
    for key in keys:
        up = swap.value(curve.key_rate_bump(key, bump_bp, keys))
        down = swap.value(curve.key_rate_bump(key, -bump_bp, keys))
        rows.append(
            {
                "key_rate_maturity_years": key,
                "sensitivity_eur_per_bp": (up - down) / 2.0,
            }
        )
    return pd.DataFrame(rows)
