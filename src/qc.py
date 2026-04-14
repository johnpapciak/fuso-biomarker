from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .utils import check_tool_available, detect_fastqs, ensure_dir, read_json, run_command, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def run_qc(metadata_csv: Path, config: dict, skip_existing: bool = True, threads: int | None = None) -> pd.DataFrame:
    fastp = config["tools"].get("fastp_path", "fastp")
    check_tool_available(fastp)

    in_dir = Path(config["paths"]["subsampled_dir"])
    qc_dir = Path(config["paths"]["qc_dir"])
    rep_dir = Path(config["paths"]["qc_reports_dir"])
    ensure_dir(qc_dir)
    ensure_dir(rep_dir)

    resolved_threads = int(threads if threads is not None else config.get("threads", 1))
    if resolved_threads < 1:
        raise ValueError("threads must be >= 1")

    md = validate_and_clean_metadata(metadata_csv)
    records: list[dict] = []

    for run in md["run_accession"]:
        try:
            in1, in2 = detect_fastqs(run, in_dir)
            out1 = qc_dir / f"{run}_qc_1.fastq.gz"
            out2 = qc_dir / f"{run}_qc_2.fastq.gz"
            if in2 is None:
                out1 = qc_dir / f"{run}_qc.fastq.gz"

            json_out = rep_dir / f"{run}.fastp.json"
            html_out = rep_dir / f"{run}.fastp.html"

            if skip_existing and out1.exists() and json_out.exists() and (in2 is None or out2.exists()):
                records.append({"run_accession": run, "status": "skipped_existing"})
                continue

            cmd = [
                fastp,
                "--in1",
                str(in1),
                "--out1",
                str(out1),
                "--json",
                str(json_out),
                "--html",
                str(html_out),
                "--thread",
                str(resolved_threads),
            ]
            if in2 is not None:
                cmd.extend(["--in2", str(in2), "--out2", str(out2)])
            run_command(cmd, LOGGER)

            summary = {"run_accession": run, "status": "qc_done", "json_report": str(json_out), "html_report": str(html_out)}
            try:
                report = read_json(json_out)
                summary["total_reads_before"] = report.get("summary", {}).get("before_filtering", {}).get("total_reads", 0)
                summary["total_reads_after"] = report.get("summary", {}).get("after_filtering", {}).get("total_reads", 0)
            except Exception:  # noqa: BLE001
                summary["total_reads_before"] = None
                summary["total_reads_after"] = None
            records.append(summary)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("QC failed for %s: %s", run, exc)
            records.append({"run_accession": run, "status": f"failed: {exc}"})

    out = pd.DataFrame(records)
    out.to_csv(qc_dir / "qc_summary.csv", index=False)
    return out
