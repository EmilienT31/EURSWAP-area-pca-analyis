"""Download and validate a compact ECB AAA government zero-curve snapshot.

Only the nodes needed by the first swap-pricing implementation are requested:
3M, 6M, 9M and annual 1Y--10Y continuously compounded spot rates. The raw API
payload is validated in memory and reduced before it is cached, avoiding the
large all-series ECB export.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd
import requests

from .swap_pricing.curve import YieldCurve


ECB_DATA_API = "https://data-api.ecb.europa.eu/service/data"
ECB_DATAFLOW = "YC"
ECB_AAA_INSTRUMENT = "G_N_A"
ECB_CURVE_MODEL = "SV_C_YM"
SPOT_TYPES = (
    "SR_3M",
    "SR_6M",
    "SR_9M",
    "SR_1Y",
    "SR_2Y",
    "SR_3Y",
    "SR_4Y",
    "SR_5Y",
    "SR_6Y",
    "SR_7Y",
    "SR_8Y",
    "SR_9Y",
    "SR_10Y",
)
SERIES_KEY = (
    "B.U2.EUR.4F.G_N_A.SV_C_YM." + "+".join(SPOT_TYPES)
)
DEFAULT_START = "2004-09-06"
DEFAULT_END = "2025-10-30"
DEFAULT_FILENAME = "aaa_spot_rates_3m_10y.csv"
METADATA_FILENAME = "aaa_spot_rates_3m_10y_metadata.json"
_MATURITY = re.compile(r"^SR_(?P<number>\d+)(?P<unit>[MY])$")


class ECBCurveDataError(ValueError):
    """Raised when an ECB response does not match the requested AAA schema."""


def api_url() -> str:
    """Return the exact ECB SDMX data URL before query parameters."""

    return f"{ECB_DATA_API}/{ECB_DATAFLOW}/{SERIES_KEY}"


def _maturity(data_type: str) -> tuple[str, float]:
    match = _MATURITY.fullmatch(data_type)
    if match is None:
        raise ECBCurveDataError(f"Unsupported ECB spot-rate code {data_type!r}.")
    number = int(match.group("number"))
    unit = match.group("unit")
    years = number / 12.0 if unit == "M" else float(number)
    return f"{number}{unit}", years


def parse_ecb_aaa_csv(payload: bytes) -> pd.DataFrame:
    """Validate an ECB CSV payload and return a compact normalized table."""

    required = {
        "KEY",
        "FREQ",
        "CURRENCY",
        "INSTRUMENT_FM",
        "PROVIDER_FM_ID",
        "DATA_TYPE_FM",
        "TIME_PERIOD",
        "OBS_VALUE",
        "OBS_STATUS",
        "UNIT",
    }
    try:
        frame = pd.read_csv(BytesIO(payload), low_memory=False)
    except Exception as exc:
        raise ECBCurveDataError(f"Could not parse ECB CSV response: {exc}") from exc
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ECBCurveDataError(f"ECB response is missing columns: {missing_columns}.")
    if frame.empty:
        raise ECBCurveDataError("ECB response contains no observations.")

    expected_values = {
        "FREQ": "B",
        "CURRENCY": "EUR",
        "INSTRUMENT_FM": ECB_AAA_INSTRUMENT,
        "PROVIDER_FM_ID": ECB_CURVE_MODEL,
        "UNIT": "PCPA",
    }
    for column, expected in expected_values.items():
        observed = set(frame[column].dropna().astype(str).unique())
        if observed != {expected}:
            raise ECBCurveDataError(
                f"ECB {column} must be {expected!r}; observed {sorted(observed)}."
            )

    observed_types = set(frame["DATA_TYPE_FM"].astype(str).unique())
    if observed_types != set(SPOT_TYPES):
        missing = sorted(set(SPOT_TYPES).difference(observed_types))
        extra = sorted(observed_types.difference(SPOT_TYPES))
        raise ECBCurveDataError(
            f"ECB spot-node mismatch; missing={missing}, unexpected={extra}."
        )

    dates = pd.to_datetime(frame["TIME_PERIOD"], errors="coerce")
    values = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    if dates.isna().any():
        raise ECBCurveDataError("ECB response contains invalid observation dates.")
    invalid_values = values.isna() & ~frame["OBS_STATUS"].astype(str).eq("M")
    if invalid_values.any():
        raise ECBCurveDataError("ECB response contains unflagged non-numeric rates.")

    maturity = frame["DATA_TYPE_FM"].map(_maturity)
    result = pd.DataFrame(
        {
            "date": dates,
            "maturity": maturity.map(lambda item: item[0]),
            "maturity_years": maturity.map(lambda item: item[1]),
            "zero_rate_pct": values,
            "zero_rate_decimal": values / 100.0,
            "series_key": frame["KEY"].astype(str),
            "observation_status": frame["OBS_STATUS"].astype(str),
        }
    )
    duplicates = result.duplicated(["date", "maturity_years"], keep=False)
    if duplicates.any():
        raise ECBCurveDataError("ECB response has duplicate date/maturity observations.")
    return result.sort_values(["date", "maturity_years"], ignore_index=True)


def download_ecb_aaa_curve(
    output_dir: str | Path,
    start_period: str = DEFAULT_START,
    end_period: str = DEFAULT_END,
    timeout: float = 120.0,
) -> tuple[Path, Path]:
    """Download, validate and cache the compact AAA curve-node history."""

    params = {
        "format": "csvdata",
        "startPeriod": start_period,
        "endPeriod": end_period,
    }
    response = requests.get(api_url(), params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    normalized = parse_ecb_aaa_csv(payload)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / DEFAULT_FILENAME
    metadata_path = destination / METADATA_FILENAME
    normalized.to_csv(data_path, index=False, date_format="%Y-%m-%d")

    metadata = {
        "source": "ECB Data Portal SDMX 2.1 REST API",
        "dataflow": ECB_DATAFLOW,
        "request_url": response.url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "raw_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "instrument_filter": ECB_AAA_INSTRUMENT,
        "instrument_description": "AAA-rated euro area central government bonds",
        "curve_model": "ECB Svensson, continuous compounding, yield-error minimisation",
        "units_received": "percent per annum",
        "units_cached": "percent per annum and decimal annual rate",
        "frequency": "TARGET business day",
        "start_period_requested": start_period,
        "end_period_requested": end_period,
        "first_observation": normalized["date"].min().date().isoformat(),
        "last_observation": normalized["date"].max().date().isoformat(),
        "rows": len(normalized),
        "dates": normalized["date"].nunique(),
        "nodes": list(SPOT_TYPES),
        "raw_payload_cached": False,
        "note": "The large raw response is validated in memory but not retained.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return data_path, metadata_path


def load_cached_aaa_curve(path: str | Path) -> pd.DataFrame:
    """Load the normalized cache and revalidate its economic identifiers."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ECB AAA curve cache not found: {source}")
    frame = pd.read_csv(source, parse_dates=["date"])
    required = {
        "date",
        "maturity",
        "maturity_years",
        "zero_rate_pct",
        "zero_rate_decimal",
        "series_key",
        "observation_status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ECBCurveDataError(f"Cached ECB curve is missing columns: {missing}.")
    if not frame["series_key"].astype(str).str.contains(".G_N_A.", regex=False).all():
        raise ECBCurveDataError("Cached ECB curve contains a non-AAA series key.")
    if not np_allclose_percent_decimal(frame):
        raise ECBCurveDataError("Cached percent and decimal rate columns disagree.")
    return frame.sort_values(["date", "maturity_years"], ignore_index=True)


def np_allclose_percent_decimal(frame: pd.DataFrame) -> bool:
    """Avoid a hard NumPy dependency in the API parsing path's public surface."""

    percent = pd.to_numeric(frame["zero_rate_pct"], errors="coerce") / 100.0
    decimal = pd.to_numeric(frame["zero_rate_decimal"], errors="coerce")
    return bool(((percent - decimal).abs() <= 1.0e-12).all())


def curve_on_date(frame: pd.DataFrame, date: str | pd.Timestamp) -> YieldCurve:
    """Construct one complete continuously compounded AAA curve snapshot."""

    requested = pd.Timestamp(date).normalize()
    selected = frame.loc[frame["date"].dt.normalize().eq(requested)].copy()
    if selected.empty:
        raise ECBCurveDataError(f"No ECB AAA curve observation on {requested.date()}.")
    selected = selected.sort_values("maturity_years")
    expected = sorted(_maturity(code)[1] for code in SPOT_TYPES)
    observed = selected["maturity_years"].tolist()
    if observed != expected or selected["zero_rate_decimal"].isna().any():
        raise ECBCurveDataError(
            f"ECB AAA curve on {requested.date()} is not complete at requested nodes."
        )
    return YieldCurve(
        selected["maturity_years"].to_numpy(float),
        selected["zero_rate_decimal"].to_numpy(float),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a compact, AAA-only ECB zero-curve history through 10Y."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/ecb_aaa_curve"),
        help="cache directory",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()
    data_path, metadata_path = download_ecb_aaa_curve(
        args.output, args.start, args.end
    )
    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
