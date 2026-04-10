from pathlib import Path

import pandas as pd

from src.utils import validate_and_clean_metadata


def test_validate_metadata_ok(tmp_path: Path) -> None:
    p = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"run_accession": "SRR1", "sample_id": "S1", "label": "healthy", "age_group": "young"},
            {"run_accession": "SRR1", "sample_id": "S1", "label": "healthy", "age_group": "young"},
        ]
    ).to_csv(p, index=False)
    out = validate_and_clean_metadata(p)
    assert len(out) == 1


def test_validate_metadata_missing_col(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame([{"run_accession": "SRR1", "sample_id": "S1"}]).to_csv(p, index=False)
    try:
        validate_and_clean_metadata(p)
        assert False
    except ValueError as exc:
        assert "required columns" in str(exc)
