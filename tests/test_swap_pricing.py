import numpy as np
import pytest

from ea_mpd.swap_pricing import (
    OISSwap,
    YieldCurve,
    bootstrap_par_swap_curve,
    bp_to_decimal,
    key_rate_sensitivities,
    parallel_dv01,
    par_rate_annuity_sensitivity,
)


@pytest.fixture()
def curve() -> YieldCurve:
    times = np.arange(1.0, 11.0)
    rates = np.array(
        [0.0200, 0.0210, 0.0220, 0.0230, 0.0240, 0.0245, 0.0250, 0.0254, 0.0257, 0.0260]
    )
    return YieldCurve(times, rates)


def test_one_basis_point_is_one_ten_thousandth() -> None:
    assert bp_to_decimal(1.0) == 0.0001


@pytest.mark.parametrize("maturity", [2.0, 5.0, 10.0])
def test_new_par_swap_has_zero_pv(curve: YieldCurve, maturity: float) -> None:
    swap = OISSwap.at_par(curve, 10_000_000, maturity)
    assert swap.value(curve) == pytest.approx(0.0, abs=1.0e-8)


def test_payer_and_receiver_values_are_opposite(curve: YieldCurve) -> None:
    coupon = curve.par_swap_rate(5.0) - 0.002
    payer = OISSwap(10_000_000, coupon, 5.0, "payer")
    receiver = OISSwap(10_000_000, coupon, 5.0, "receiver")
    assert payer.value(curve) == pytest.approx(-receiver.value(curve), abs=1.0e-9)


def test_par_payer_gains_when_curve_shifts_up(curve: YieldCurve) -> None:
    payer = OISSwap.at_par(curve, 10_000_000, 10.0, "payer")
    assert payer.value(curve.parallel_bump(1.0)) > payer.value(curve)
    assert parallel_dv01(payer, curve) > 0


def test_bootstrap_reproduces_input_par_quotes(curve: YieldCurve) -> None:
    maturities = np.arange(1.0, 11.0)
    quotes = np.array([curve.par_swap_rate(term) for term in maturities])
    bootstrapped = bootstrap_par_swap_curve(maturities, quotes)
    reproduced = np.array(
        [bootstrapped.par_swap_rate(term) for term in maturities]
    )
    np.testing.assert_allclose(reproduced, quotes, atol=1.0e-12, rtol=0.0)


def test_key_rate_first_orders_sum_to_parallel_bump(curve: YieldCurve) -> None:
    payer = OISSwap.at_par(curve, 10_000_000, 10.0, "payer")
    keys = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
    key_sensitivities = key_rate_sensitivities(payer, curve, keys)
    parallel = parallel_dv01(payer, curve)
    assert key_sensitivities["sensitivity_eur_per_bp"].sum() == pytest.approx(
        parallel, abs=0.25
    )


def test_par_annuity_parallel_and_individual_key_risk_are_distinct(
    curve: YieldCurve,
) -> None:
    payer = OISSwap.at_par(curve, 10_000_000, 5.0, "payer")
    par_annuity = par_rate_annuity_sensitivity(payer, curve, per_bp=True)
    parallel = parallel_dv01(payer, curve)
    keys = key_rate_sensitivities(payer, curve, np.arange(1.0, 11.0))
    own_key = keys.loc[
        keys["key_rate_maturity_years"].eq(5.0), "sensitivity_eur_per_bp"
    ].item()

    assert par_annuity != pytest.approx(parallel, abs=0.01)
    assert own_key != pytest.approx(par_annuity, abs=0.01)
    assert keys["sensitivity_eur_per_bp"].sum() == pytest.approx(
        parallel, abs=0.25
    )


def test_log_discount_interpolation_and_forward_rate() -> None:
    curve = YieldCurve(np.array([1.0, 2.0]), np.array([0.02, 0.03]))
    expected_log_df = 0.5 * (-0.02) + 0.5 * (-0.06)
    assert curve.log_discount_factor(1.5) == pytest.approx(expected_log_df)
    assert curve.forward_rate(1.0, 2.0) == pytest.approx(0.04)
