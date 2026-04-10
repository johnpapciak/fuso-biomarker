from pathlib import Path

import pandas as pd

from src.features import build_features, shannon_diversity
from src.kraken import parse_kraken_report


def test_parse_kraken_report(tmp_path: Path) -> None:
    rp = tmp_path / "x.report.tsv"
    rp.write_text("50.0\t100\t100\tS\t123\tFusobacterium nucleatum\n", encoding="utf-8")
    df = parse_kraken_report(rp, "S1")
    assert df.shape[0] == 1
    assert df.iloc[0]["name"] == "Fusobacterium nucleatum"


def test_shannon() -> None:
    val = shannon_diversity([0.5, 0.5])
    assert 0.68 < val < 0.70


def test_build_features(tmp_path: Path) -> None:
    (tmp_path / "data/kraken").mkdir(parents=True)
    (tmp_path / "data/qc/reports").mkdir(parents=True)
    (tmp_path / "data/features").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)

    md = pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "cancer", "age_group": "old"}]
    )
    md_path = tmp_path / "meta.csv"
    md.to_csv(md_path, index=False)

    (tmp_path / "data/kraken/SRR1.report.tsv").write_text(
        "100.0\t100\t100\tD\t2\tBacteria\n"
        "10.0\t10\t10\tG\t848\tFusobacterium\n"
        "5.0\t5\t5\tS\t851\tFusobacterium nucleatum\n",
        encoding="utf-8",
    )

    config = {
        "paths": {
            "kraken_dir": str(tmp_path / "data/kraken"),
            "qc_dir": str(tmp_path / "data/qc"),
            "features_dir": str(tmp_path / "data/features"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "panel_taxa": ["Fusobacterium nucleatum"],
        "pseudocount": 1e-6,
        "presence_absence_threshold": 1e-4,
    }
    out = build_features(md_path, config)
    assert "F_nucleatum_species_abundance" in out.columns
    assert out.iloc[0]["F_nucleatum_species_abundance"] == 0.05
