from datetime import date, timedelta

import numpy as np
import pytest

from ea_mpd.swap_pricing import YieldCurve


ql = pytest.importorskip("QuantLib")


def test_quantlib_log_linear_discount_curve_matches_internal_nodes() -> None:
    internal = YieldCurve(
        np.array([1.0, 2.0, 5.0, 10.0]),
        np.array([0.0200, 0.0210, 0.0240, 0.0260]),
    )
    reference = date(2025, 1, 1)
    ql_dates = [ql.Date(reference.day, reference.month, reference.year)]
    ql_dates.extend(
        ql.Date.from_date(reference + timedelta(days=365 * int(term)))
        for term in internal.times
    )
    discounts = [1.0, *internal.discount_factor(internal.times).tolist()]
    external = ql.DiscountCurve(
        ql_dates,
        discounts,
        ql.Actual365Fixed(),
        ql.NullCalendar(),
        ql.LogLinear(),
    )
    for term, ql_date in zip(internal.times, ql_dates[1:], strict=True):
        assert external.discount(ql_date) == pytest.approx(
            internal.discount_factor(term), abs=1.0e-12
        )
