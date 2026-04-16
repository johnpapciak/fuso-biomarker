from pathlib import Path

import pandas as pd

from src.bracken import run_bracken


def test_run_bracken_generates_outputs(tmp_path: Path, monkeypatch) -> None:
    kraken_dir = tmp_path / "data/kraken"
    bracken_dir = tmp_path / "data/bracken"
    kraken_dir.mkdir(parents=True)
    bracken_dir.mkdir(parents=True)

    md_path = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "cancer", "age_group": "old"}]
    ).to_csv(md_path, index=False)

    (kraken_dir / "SRR1.report.tsv").write_text(
        "100.00\t100\t100\tS\t123\tFusobacterium nucleatum\n", encoding="utf-8"
    )

    def fake_run_command(cmd: list[str], logger, cwd=None) -> None:
        out_idx = cmd.index("-o") + 1
        rep_idx = cmd.index("-w") + 1
        Path(cmd[out_idx]).write_text(
            "name\ttaxonomy_id\ttaxonomy_lvl\tkraken_assigned_reads\tadded_reads\tnew_est_reads\tfraction_total_reads\n"
            "Fusobacterium nucleatum\t851\tS\t5\t2\t7\t0.07\n",
            encoding="utf-8",
        )
        Path(cmd[rep_idx]).write_text("", encoding="utf-8")

    monkeypatch.setattr("src.bracken.check_tool_available", lambda _tool: None)
    monkeypatch.setattr("src.bracken.run_command", fake_run_command)

    config = {
        "tools": {"bracken_path": "bracken"},
        "kraken_db": str(tmp_path),
        "paths": {"kraken_dir": str(kraken_dir), "bracken_dir": str(bracken_dir)},
        "threads": 2,
        "bracken_read_length": 100,
        "bracken_level": "S",
        "bracken_threshold": 1,
    }

    log_df = run_bracken(md_path, config, skip_existing=False)
    assert log_df.iloc[0]["status"] == "bracken_done"

    abundance = pd.read_csv(bracken_dir / "bracken_abundance.csv")
    assert abundance.iloc[0]["sample_id"] == "S1"
    assert float(abundance.iloc[0]["fraction_total_reads"]) == 0.07
