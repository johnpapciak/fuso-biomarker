from __future__ import annotations

import logging
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
            detect_fastqs(run, raw_dir)
            records.append({"run_accession": run, "status": "downloaded"})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed downloading %s: %s", run, exc)
            records.append({"run_accession": run, "status": f"failed: {exc}"})

    out = pd.DataFrame(records)
    out.to_csv(raw_dir / "download_log.csv", index=False)
    return out
