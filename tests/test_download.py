from pathlib import Path

import pandas as pd

from src import download
from src.cli import build_parser


def _config(tmp_path: Path) -> dict:
    return {
        "tools": {"prefetch_path": "prefetch", "fasterq_dump_path": "fasterq-dump"},
        "paths": {"raw_dir": str(tmp_path / "raw"), "cleaned_metadata": str(tmp_path / "clean.csv")},
        "threads": 1,
    }


def test_download_workers_cli_option() -> None:
    parser = build_parser()
    args = parser.parse_args(["download", "--metadata", "meta.csv", "--threads", "6", "--download-workers", "3"])
    assert args.download_workers == 3


def test_skip_when_fastq_exists(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "SRR1_1.fastq.gz").write_text("x", encoding="utf-8")
    (raw / "SRR1_2.fastq.gz").write_text("x", encoding="utf-8")

    result = download.download_one_run("SRR1", _config(tmp_path), threads=2)
    assert result.status == "skipped_existing"


def test_download_runs_parallel_propagates_worker_and_threads(tmp_path: Path, monkeypatch) -> None:
    md = tmp_path / "meta.csv"
    pd.DataFrame([
        {"run_accession": "SRR1", "sample_id": "S1", "label": "healthy", "age_group": "young"},
        {"run_accession": "SRR2", "sample_id": "S2", "label": "cancer", "age_group": "old"},
    ]).to_csv(md, index=False)

    monkeypatch.setattr(download, "check_tool_available", lambda _tool: None)
    calls = []

    def fake_download_one_run(run_accession: str, config: dict, threads: int):
        calls.append((run_accession, threads))
        return download.DownloadResult(run_accession, "downloaded", "ok")

    monkeypatch.setattr(download, "download_one_run", fake_download_one_run)

    out = download.download_runs_parallel(md, _config(tmp_path), threads=5, download_workers=1)
    assert len(out) == 2
    assert calls == [("SRR1", 5), ("SRR2", 5)]


def test_missing_run_accession_column(tmp_path: Path) -> None:
    md = tmp_path / "meta.csv"
    pd.DataFrame([{"sample_id": "S1", "label": "healthy", "age_group": "young"}]).to_csv(md, index=False)
    try:
        download.download_runs_parallel(md, _config(tmp_path), threads=1, download_workers=1)
        assert False
    except ValueError as exc:
        assert "required columns" in str(exc) or "run_accession" in str(exc)
