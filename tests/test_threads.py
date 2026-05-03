from pathlib import Path

import pandas as pd

from src.cli import build_parser
from src import qc


def test_cli_threads_option_for_qc_and_run_all() -> None:
    parser = build_parser()

    qc_args = parser.parse_args(["qc", "--metadata", "meta.csv", "--threads", "4"])
    assert qc_args.threads == 4

    download_args = parser.parse_args(["download", "--metadata", "meta.csv", "--threads", "4", "--download-workers", "2"])
    assert download_args.threads == 4
    assert download_args.download_workers == 2

    run_all_args = parser.parse_args(["run-all", "--metadata", "meta.csv", "--threads", "6", "--download-workers", "3"])
    assert run_all_args.threads == 6
    assert run_all_args.download_workers == 3


def test_run_qc_passes_thread_flag(tmp_path: Path, monkeypatch) -> None:
    in_dir = tmp_path / "subsampled"
    qc_dir = tmp_path / "qc"
    rep_dir = tmp_path / "qc_reports"
    in_dir.mkdir()
    qc_dir.mkdir()
    rep_dir.mkdir()

    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "healthy", "age_group": "young"}]
    ).to_csv(metadata, index=False)

    in_fastq = in_dir / "SRR1.fastq"
    in_fastq.write_text("@r0\nACGT\n+\nIIII\n", encoding="utf-8")

    config = {
        "tools": {"fastp_path": "fastp"},
        "paths": {
            "subsampled_dir": str(in_dir),
            "qc_dir": str(qc_dir),
            "qc_reports_dir": str(rep_dir),
        },
    }

    monkeypatch.setattr(qc, "check_tool_available", lambda _tool: None)

    recorded_cmd = {}

    def fake_run_command(cmd: list[str], _logger) -> None:
        recorded_cmd["cmd"] = cmd

    monkeypatch.setattr(qc, "run_command", fake_run_command)
    monkeypatch.setattr(qc, "read_json", lambda _path: {})

    qc.run_qc(metadata, config, threads=3, skip_existing=False)

    assert "--thread" in recorded_cmd["cmd"]
    assert recorded_cmd["cmd"][recorded_cmd["cmd"].index("--thread") + 1] == "3"
    assert recorded_cmd["cmd"][recorded_cmd["cmd"].index("--out1") + 1].endswith(".fastq.gz")
