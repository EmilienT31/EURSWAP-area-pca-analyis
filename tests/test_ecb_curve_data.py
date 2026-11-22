from io import StringIO

import pandas as pd
import pytest

from ea_mpd.ecb_curve_data import (
    ECBCurveDataError,
    SPOT_TYPES,
    curve_on_date,
    parse_ecb_aaa_csv,
)


def _payload(instrument: str = "G_N_A") -> bytes:
    rows = []
    for index, data_type in enumerate(SPOT_TYPES):
        rows.append(
            {
                "KEY": f"YC.B.U2.EUR.4F.{instrument}.SV_C_YM.{data_type}",
                "FREQ": "B",
                "CURRENCY": "EUR",
                "INSTRUMENT_FM": instrument,
                "PROVIDER_FM_ID": "SV_C_YM",
                "DATA_TYPE_FM": data_type,
                "TIME_PERIOD": "2025-01-30",
                "OBS_VALUE": 2.0 + index / 10,
                "OBS_STATUS": "A",
                "UNIT": "PCPA",
            }
        )
    buffer = StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def test_parser_accepts_only_complete_aaa_spot_schema() -> None:
    frame = parse_ecb_aaa_csv(_payload())
    assert len(frame) == len(SPOT_TYPES)
    assert frame["series_key"].str.contains(".G_N_A.", regex=False).all()
    assert frame["zero_rate_decimal"].iloc[0] == pytest.approx(0.02)
    curve = curve_on_date(frame, "2025-01-30")
    assert curve.times[-1] == 10.0


def test_parser_rejects_all_ratings_curve() -> None:
    with pytest.raises(ECBCurveDataError, match="INSTRUMENT_FM"):
        parse_ecb_aaa_csv(_payload("G_N_C"))
