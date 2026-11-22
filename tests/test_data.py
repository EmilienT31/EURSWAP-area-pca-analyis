from pathlib import Path

import pandas as pd
import pytest

from ea_mpd.data import ValidationError, load_ea_mpd


@pytest.fixture()
def miniature_workbook(tmp_path: Path) -> Path:
    """A clearly labelled test fixture; it is never used as research data."""

    path = tmp_path / "fixture.xlsx"
    notes = pd.DataFrame(
        {
            "notes": [
                "Test workbook notes",
                "OIS1W: 1 week OIS rate change in basis points.",
                "OIS1M: 1 month OIS rate change in basis points.",
            ]
        }
    )
    events = pd.DataFrame(
        {
            # Mirrors the source workbook's recent day-first text dates.
            "when": ["07/01/2020", "13/02/2020", "05/03/2020"],
            "OIS_SW": [1.0, -1.0, 0.5],
            "OIS_1M": [2.0, -2.0, 0.0],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False, header=False)
        events.to_excel(writer, sheet_name="Monetary Event Window", index=False)
    return path


def test_loader_infers_date_and_reconciles_observed_header_typo(
    miniature_workbook: Path,
) -> None:
    workbook = load_ea_mpd(miniature_workbook)
    schema = workbook.schemas["Monetary Event Window"]

    assert schema.date_column == "when"
    assert workbook.sheets["Monetary Event Window"]["when"].iloc[1] == pd.Timestamp(
        "2020-02-13"
    )
    assert [column.canonical_name for column in schema.ois_columns] == [
        "OIS1W",
        "OIS1M",
    ]


def test_loader_rejects_duplicate_dates(miniature_workbook: Path) -> None:
    frames = pd.read_excel(miniature_workbook, sheet_name=None)
    frames["Monetary Event Window"].loc[1, "when"] = frames[
        "Monetary Event Window"
    ].loc[0, "when"]
    with pd.ExcelWriter(miniature_workbook, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    with pytest.raises(ValidationError, match="duplicate event dates"):
        load_ea_mpd(miniature_workbook)

