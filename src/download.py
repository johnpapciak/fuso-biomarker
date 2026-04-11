from __future__ import annotations

import logging
import shutil
from gzip import open as gzopen
from pathlib import Path

import pandas as pd

from .utils import (
    check_tool_available,
    detect_fastqs,
    ensure_dir,
    run_command,
    validate_and_clean_metadata,
)

LOGGER = logging.getLogger(__name__)


def _gzip_fastq(path: Path) -> Path:
    if path.suffix == ".gz":
        return path
    gz_path = path.with_suffix(f"{path.suffix}.gz")
    with path.open("rb") as src, gzopen(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return gz_path


def download_runs(
    metadata_csv: Path,
    config: dict,
    limit: int | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    tools = config["tools"]
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_dir(raw_dir)

    cleaned_meta_path = Path(config["paths"]["cleaned_metadata"])
    md = validate_and_clean_metadata(metadata_csv, cleaned_meta_path)
    if limit is not None:
        md = md.head(limit)

    prefetch = tools.get("prefetch_path", "prefetch")
    fasterq = tools.get("fasterq_dump_path", "fasterq-dump")
    check_tool_available(fasterq)
    if prefetch:
        check_tool_available(prefetch)

    records: list[dict[str, str]] = []
    for run in md["run_accession"]:
        try:
            if skip_existing:
                try:
                    detect_fastqs(run, raw_dir)
                    LOGGER.info("Skipping %s (FASTQ already exists)", run)
                    records.append({"run_accession": run, "status": "skipped_existing"})
                    continue
                except FileNotFoundError:
                    pass

            if prefetch:
                run_command([prefetch, run, "-O", str(raw_dir)], LOGGER)
                sra_path = raw_dir / run / f"{run}.sra"
                if sra_path.exists():
                    cmd = [fasterq, str(sra_path), "-O", str(raw_dir), "--split-files"]
                else:
                    cmd = [fasterq, run, "-O", str(raw_dir), "--split-files"]
            else:
                cmd = [fasterq, run, "-O", str(raw_dir), "--split-files"]

            run_command(cmd, LOGGER)
            fq1, fq2 = detect_fastqs(run, raw_dir)
            _gzip_fastq(fq1)
            if fq2 is not None:
                _gzip_fastq(fq2)
            records.append({"run_accession": run, "status": "downloaded"})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed downloading %s: %s", run, exc)
            records.append({"run_accession": run, "status": f"failed: {exc}"})

    out = pd.DataFrame(records)
    out.to_csv(raw_dir / "download_log.csv", index=False)
    return out
