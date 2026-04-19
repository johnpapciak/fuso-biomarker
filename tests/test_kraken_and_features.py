from pathlib import Path

import pandas as pd

from src.features import build_features, shannon_diversity
from src.kraken import parse_kraken_report, run_kraken


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
    (tmp_path / "data/bracken").mkdir(parents=True)
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
        "5.0\t5\t5\tS\t851\tFusobacterium nucleatum\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_accession": "SRR1",
                "sample_id": "S1",
                "name": "Fusobacterium nucleatum",
                "fraction_total_reads": 0.05,
            }
        ]
    ).to_csv(tmp_path / "data/bracken/bracken_abundance_species.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_accession": "SRR1",
                "sample_id": "S1",
                "name": "Fusobacterium",
                "fraction_total_reads": 0.10,
            }
        ]
    ).to_csv(tmp_path / "data/bracken/bracken_abundance_genus.csv", index=False)

    config = {
        "paths": {
            "kraken_dir": str(tmp_path / "data/kraken"),
            "bracken_dir": str(tmp_path / "data/bracken"),
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
    assert out.iloc[0]["Fusobacterium_genus_abundance"] == 0.10


def test_run_kraken_optional_memory_mapping(tmp_path: Path, monkeypatch) -> None:
    qc_dir = tmp_path / "data/qc"
    kraken_dir = tmp_path / "data/kraken"
    qc_dir.mkdir(parents=True)
    kraken_dir.mkdir(parents=True)

    md_path = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "cancer", "age_group": "old"}]
    ).to_csv(md_path, index=False)

    (qc_dir / "SRR1_qc.fastq").write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")

    captured = {}

    def fake_run_command(cmd: list[str], logger, cwd=None) -> None:
        captured["cmd"] = cmd
        report_idx = cmd.index("--report") + 1
        output_idx = cmd.index("--output") + 1
        Path(cmd[report_idx]).write_text("", encoding="utf-8")
        Path(cmd[output_idx]).write_text("", encoding="utf-8")

    monkeypatch.setattr("src.kraken.check_tool_available", lambda _tool: None)
    monkeypatch.setattr("src.kraken.run_command", fake_run_command)

    config = {
        "tools": {"kraken2_path": "kraken2"},
        "kraken_db": str(tmp_path),
        "paths": {"qc_dir": str(qc_dir), "kraken_dir": str(kraken_dir)},
        "threads": 2,
        "kraken_memory_mapping": True,
    }

    run_kraken(md_path, config, skip_existing=False)
    assert "--memory-mapping" in captured["cmd"]


def test_run_kraken_memory_mapping_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    qc_dir = tmp_path / "data/qc"
    kraken_dir = tmp_path / "data/kraken"
    qc_dir.mkdir(parents=True)
    kraken_dir.mkdir(parents=True)

    md_path = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "cancer", "age_group": "old"}]
    ).to_csv(md_path, index=False)

    (qc_dir / "SRR1_qc.fastq").write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")

    captured = {}

    def fake_run_command(cmd: list[str], logger, cwd=None) -> None:
        captured["cmd"] = cmd
        report_idx = cmd.index("--report") + 1
        output_idx = cmd.index("--output") + 1
        Path(cmd[report_idx]).write_text("", encoding="utf-8")
        Path(cmd[output_idx]).write_text("", encoding="utf-8")

    monkeypatch.setattr("src.kraken.check_tool_available", lambda _tool: None)
    monkeypatch.setattr("src.kraken.run_command", fake_run_command)

    config = {
        "tools": {"kraken2_path": "kraken2"},
        "kraken_db": str(tmp_path),
        "paths": {"qc_dir": str(qc_dir), "kraken_dir": str(kraken_dir)},
        "threads": 2,
    }

    run_kraken(md_path, config, skip_existing=False)
    assert "--memory-mapping" not in captured["cmd"]
