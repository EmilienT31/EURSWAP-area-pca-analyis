"""Load and validate the observed EA-MPD workbook schema.

The workbook's Notes sheet is treated as part of the schema. In particular, it
allows the loader to reconcile minor header inconsistencies without maintaining
a hard-coded list of OIS columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import re
from typing import Mapping

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


MATURITY_PATTERN = re.compile(r"^OIS(?P<number>\d+)(?P<unit>[WMY])$")
NOTE_DEFINITION_PATTERN = re.compile(r"^(OIS\d+[WMY])\s*:", re.IGNORECASE)


class ValidationError(ValueError):
    """Raised when a workbook does not satisfy the inferred EA-MPD schema."""


@dataclass(frozen=True)
class OISColumn:
    """An observed workbook column and its economically ordered maturity."""

    source_name: str
    canonical_name: str
    maturity_label: str
    maturity_years: float


@dataclass(frozen=True)
class SheetSchema:
    """Validated schema for one event-window sheet."""

    date_column: str
    ois_columns: tuple[OISColumn, ...]


@dataclass(frozen=True)
class EAMPDWorkbook:
    """Validated sheets and their inferred schemas."""

    source: Path
    notes: tuple[str, ...]
    sheets: Mapping[str, pd.DataFrame]
    schemas: Mapping[str, SheetSchema]


def _normalise_header(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _maturity(canonical_name: str) -> tuple[str, float]:
    match = MATURITY_PATTERN.fullmatch(canonical_name)
    if match is None:
        raise ValidationError(f"Cannot parse OIS maturity from {canonical_name!r}.")
    number = int(match.group("number"))
    unit = match.group("unit")
    years = {"W": number / 52.0, "M": number / 12.0, "Y": float(number)}[unit]
    return f"{number}{unit}", years


def _notes_text(notes_frame: pd.DataFrame) -> tuple[str, ...]:
    values: list[str] = []
    for value in notes_frame.to_numpy().ravel():
        if pd.notna(value):
            text = str(value).strip()
            if text:
                values.append(text)
    return tuple(values)


def _documented_ois_names(notes: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for line in notes:
        match = NOTE_DEFINITION_PATTERN.match(line)
        if match:
            names.append(_normalise_header(match.group(1)))
    return tuple(dict.fromkeys(names))


def _infer_date_column(frame: pd.DataFrame, sheet_name: str) -> tuple[str, pd.Series]:
    candidates: list[tuple[str, pd.Series]] = []
    for column in frame.columns:
        series = frame[column]
        converted: pd.Series | None = None
        if is_datetime64_any_dtype(series):
            converted = pd.to_datetime(series, errors="coerce")
        elif not is_numeric_dtype(series):
            # EA-MPD mixes native Excel datetimes with recent DD/MM/YYYY text.
            # Infer ordering from unambiguous slash-separated values rather than
            # letting pandas silently interpret an ambiguous 11/04 month-first.
            day_first_evidence = False
            month_first_evidence = False
            for value in series.dropna():
                match = re.fullmatch(
                    r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", str(value)
                )
                if match:
                    first, second = int(match.group(1)), int(match.group(2))
                    day_first_evidence |= first > 12 and second <= 12
                    month_first_evidence |= second > 12 and first <= 12
            if day_first_evidence and month_first_evidence:
                raise ValidationError(
                    f"{sheet_name!r}/{column!r}: conflicting day/month date ordering."
                )
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
                dayfirst=day_first_evidence,
            )
            if parsed.notna().mean() >= 0.95:
                converted = parsed
        elif series.notna().any():
            # Support raw Excel serial dates if formatting metadata has been lost.
            numeric = pd.to_numeric(series, errors="coerce")
            parsed = pd.to_datetime(
                numeric, unit="D", origin="1899-12-30", errors="coerce"
            )
            plausible = parsed.between("1990-01-01", "2100-12-31")
            if (plausible | numeric.isna()).mean() >= 0.95:
                converted = parsed.where(plausible)
        if converted is not None and converted.notna().mean() >= 0.95:
            candidates.append((str(column), converted))

    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ValidationError(
            f"{sheet_name!r}: expected one date-like column, found {names or 'none'}."
        )
    return candidates[0]


def _infer_ois_columns(
    frame: pd.DataFrame,
    documented_names: tuple[str, ...],
    sheet_name: str,
) -> tuple[OISColumn, ...]:
    inferred: list[OISColumn] = []
    used_canonical: set[str] = set()

    for raw_column in frame.columns:
        source_name = str(raw_column)
        compact = _normalise_header(source_name)
        if not compact.startswith("OIS"):
            continue

        canonical = compact if MATURITY_PATTERN.fullmatch(compact) else None
        if canonical is None and documented_names:
            remaining = [name for name in documented_names if name not in used_canonical]
            matches = get_close_matches(compact, remaining, n=2, cutoff=0.75)
            if len(matches) == 1:
                canonical = matches[0]
            elif len(matches) > 1:
                raise ValidationError(
                    f"{sheet_name!r}: ambiguous Notes match for OIS header "
                    f"{source_name!r}: {matches}."
                )
        if canonical is None:
            raise ValidationError(
                f"{sheet_name!r}: unrecognised OIS maturity header {source_name!r}."
            )
        if canonical in used_canonical:
            raise ValidationError(
                f"{sheet_name!r}: duplicate OIS maturity after normalisation: "
                f"{canonical}."
            )

        numeric = pd.to_numeric(frame[raw_column], errors="coerce")
        invalid = frame[raw_column].notna() & numeric.isna()
        if invalid.any():
            examples = frame.loc[invalid, raw_column].astype(str).head(3).tolist()
            raise ValidationError(
                f"{sheet_name!r}/{source_name!r}: non-numeric observations {examples}."
            )
        if numeric.notna().sum() == 0:
            raise ValidationError(
                f"{sheet_name!r}/{source_name!r}: column has no observations."
            )

        label, years = _maturity(canonical)
        inferred.append(OISColumn(source_name, canonical, label, years))
        used_canonical.add(canonical)

    if len(inferred) < 2:
        raise ValidationError(
            f"{sheet_name!r}: expected at least two maturity-labelled OIS columns."
        )
    return tuple(sorted(inferred, key=lambda item: item.maturity_years))


def load_ea_mpd(path: str | Path) -> EAMPDWorkbook:
    """Load an EA-MPD XLSX file and validate its inferred event-sheet schema.

    No full column list is assumed. Data sheets are discovered from their date
    values and maturity-labelled OIS headers, while the workbook's Notes text is
    used to resolve a uniquely close header typo.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EA-MPD workbook not found: {source}")
    if source.suffix.lower() != ".xlsx":
        raise ValidationError(f"Expected an .xlsx workbook, got {source.name!r}.")

    try:
        raw_sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
    except Exception as exc:  # openpyxl emits several format-specific exceptions
        raise ValidationError(f"Could not read {source.name!r}: {exc}") from exc
    if not raw_sheets:
        raise ValidationError("Workbook contains no worksheets.")

    notes_sheet_name = next(
        (name for name in raw_sheets if name.strip().casefold() == "notes"), None
    )
    if notes_sheet_name is None:
        raise ValidationError("Workbook is missing its Notes sheet.")
    notes = _notes_text(raw_sheets[notes_sheet_name])
    documented_names = _documented_ois_names(notes)
    if len(documented_names) < 2:
        raise ValidationError("Notes sheet does not document the OIS maturities.")

    sheets: dict[str, pd.DataFrame] = {}
    schemas: dict[str, SheetSchema] = {}
    for sheet_name, raw_frame in raw_sheets.items():
        if sheet_name == notes_sheet_name:
            continue
        frame = raw_frame.dropna(axis="index", how="all").dropna(axis="columns", how="all")
        if frame.empty:
            continue
        try:
            date_column, dates = _infer_date_column(frame, sheet_name)
            ois_columns = _infer_ois_columns(frame, documented_names, sheet_name)
        except ValidationError:
            raise

        clean = frame.copy()
        clean[date_column] = dates.dt.normalize()
        for column in ois_columns:
            clean[column.source_name] = pd.to_numeric(
                clean[column.source_name], errors="coerce"
            )

        if clean[date_column].isna().any():
            raise ValidationError(f"{sheet_name!r}: missing or invalid event dates.")
        if clean[date_column].duplicated().any():
            duplicates = (
                clean.loc[clean[date_column].duplicated(False), date_column]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            raise ValidationError(f"{sheet_name!r}: duplicate event dates {duplicates}.")
        if not clean[date_column].is_monotonic_increasing:
            raise ValidationError(f"{sheet_name!r}: event dates are not increasing.")

        sheets[sheet_name] = clean.reset_index(drop=True)
        schemas[sheet_name] = SheetSchema(date_column, ois_columns)

    if not sheets:
        raise ValidationError("No event-window sheets could be validated.")
    if len({len(frame) for frame in sheets.values()}) != 1:
        sizes = {name: len(frame) for name, frame in sheets.items()}
        raise ValidationError(f"Event-window sheets have inconsistent row counts: {sizes}.")

    return EAMPDWorkbook(source, notes, sheets, schemas)


def find_monetary_event_sheet(workbook: EAMPDWorkbook) -> str:
    """Find the uniquely named full monetary-event window sheet."""

    matches = [
        name
        for name in workbook.sheets
        if "monetary" in name.casefold() and "event" in name.casefold()
    ]
    if len(matches) != 1:
        raise ValidationError(
            "Expected one sheet whose name identifies the monetary event window; "
            f"found {matches or 'none'}."
        )
    return matches[0]

