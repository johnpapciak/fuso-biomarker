from pathlib import Path

import pandas as pd

from src import subsample


def _write_fastq(path: Path, n_reads: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n_reads):
            handle.write(f"@r{i}\n")
            handle.write("ACGT\n")
            handle.write("+\n")
            handle.write("IIII\n")


def test_subsample_retries_python_when_seqtk_output_is_empty(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    sub_dir = tmp_path / "subsampled"
    raw_dir.mkdir()
    sub_dir.mkdir()

    _write_fastq(raw_dir / "SRR1.fastq", 20)

    metadata = tmp_path / "meta.csv"
    pd.DataFrame(
        [{"run_accession": "SRR1", "sample_id": "S1", "label": "healthy", "age_group": "young"}]
    ).to_csv(metadata, index=False)

    config = {
        "paths": {"raw_dir": str(raw_dir), "subsampled_dir": str(sub_dir)},
        "tools": {"seqtk_path": "seqtk"},
    }

    monkeypatch.setattr(subsample.shutil, "which", lambda _: "/usr/bin/seqtk")

    def fake_seqtk(_seqtk: str, _inp: Path, out: Path, _fraction: float) -> None:
        out.write_text("", encoding="utf-8")

    monkeypatch.setattr(subsample, "_seqtk_subsample", fake_seqtk)

    out = subsample.subsample_runs(metadata, config, fraction=0.5)
    assert out.iloc[0]["status"] == "subsampled"
    assert out.iloc[0]["subsampled_reads"] > 0
