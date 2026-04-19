from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .utils import check_tool_available, ensure_dir, run_command, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def run_bracken(metadata_csv: Path, config: dict, skip_existing: bool = True, threads: int | None = None) -> pd.DataFrame:
    bracken = config["tools"].get("bracken_path", "bracken")
    check_tool_available(bracken)

    db = config.get("kraken_db")
    if not db or not Path(db).exists():
        raise FileNotFoundError("kraken_db is missing/invalid in config.yaml")

    kraken_dir = Path(config["paths"]["kraken_dir"])
    bracken_dir = Path(config["paths"].get("bracken_dir", "data/bracken"))
    ensure_dir(bracken_dir)

    read_len = int(config.get("bracken_read_length", 100))
    level = str(config.get("bracken_level", "S"))
    threshold = int(config.get("bracken_threshold", 10))
    resolved_threads = int(threads if threads is not None else config.get("threads", 1))
    if resolved_threads < 1:
        raise ValueError("threads must be >= 1")

    md = validate_and_clean_metadata(metadata_csv)
    logs: list[dict[str, str]] = []

    for _, row in md.iterrows():
        run = row["run_accession"]
        sample_id = row["sample_id"]
        input_report = kraken_dir / f"{run}.report.tsv"
        output_file = bracken_dir / f"{run}.bracken.tsv"
        output_report = bracken_dir / f"{run}.bracken.report.tsv"

        try:
            if not input_report.exists() or input_report.stat().st_size == 0:
                raise FileNotFoundError(f"Missing Kraken report: {input_report}")

            if skip_existing and output_file.exists() and output_report.exists():
                output_ready = output_file.stat().st_size > 0
                report_ready = output_report.stat().st_size > 0
                if output_ready and report_ready:
                    logs.append({"run_accession": run, "sample_id": sample_id, "status": "skipped_existing"})
                    continue
                LOGGER.warning(
                    "Found existing but empty Bracken outputs for %s; re-running sample", run
                )

            cmd = [
                bracken,
                "-d",
                str(db),
                "-i",
                str(input_report),
                "-o",
                str(output_file),
                "-w",
                str(output_report),
                "-r",
                str(read_len),
                "-l",
                level,
                "-t",
                str(threshold),
            ]
            if resolved_threads != 1:
                LOGGER.warning("Bracken CLI does not support threads directly; running single process per sample")
            run_command(cmd, LOGGER)
            logs.append({"run_accession": run, "sample_id": sample_id, "status": "bracken_done"})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Bracken failed for %s: %s", run, exc)
            logs.append({"run_accession": run, "sample_id": sample_id, "status": f"failed: {exc}"})

    log_df = pd.DataFrame(logs)
    log_df.to_csv(bracken_dir / "bracken_log.csv", index=False)

    combined_frames: list[pd.DataFrame] = []
    for _, row in md.iterrows():
        run = row["run_accession"]
        sample_id = row["sample_id"]
        file_path = bracken_dir / f"{run}.bracken.tsv"
        if not file_path.exists() or file_path.stat().st_size == 0:
            continue
        df = pd.read_csv(file_path, sep="\t")
        df.insert(0, "sample_id", sample_id)
        df.insert(0, "run_accession", run)
        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    combined_df.to_csv(bracken_dir / "bracken_abundance.csv", index=False)
    return log_df
