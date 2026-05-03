from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from gzip import open as gzopen
from pathlib import Path

import pandas as pd

from .utils import check_tool_available, detect_fastqs, ensure_dir, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result for processing a single SRA run."""

    run_accession: str
    status: str
    message: str = ""


def _gzip_fastq(path: Path) -> Path:
    if path.suffix == ".gz":
        return path
    gz_path = path.with_suffix(f"{path.suffix}.gz")
    with path.open("rb") as src, gzopen(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return gz_path


def _run_subprocess(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        msg = (
            f"Command failed ({exc.returncode}): {' '.join(cmd)} | "
            f"stdout={exc.stdout.strip() if exc.stdout else ''} | "
            f"stderr={exc.stderr.strip() if exc.stderr else ''}"
        )
        raise RuntimeError(msg) from exc


def _fastq_exists(run_accession: str, raw_dir: Path) -> bool:
    try:
        detect_fastqs(run_accession, raw_dir)
        return True
    except FileNotFoundError:
        return False


def download_one_run(run_accession: str, config: dict, threads: int) -> DownloadResult:
    """Download and convert one SRA run to FASTQ files."""
    tools = config["tools"]
    raw_dir = Path(config["paths"]["raw_dir"])
    prefetch = tools.get("prefetch_path", "prefetch")
    fasterq = tools.get("fasterq_dump_path", "fasterq-dump")

    if _fastq_exists(run_accession, raw_dir):
        return DownloadResult(run_accession, "skipped_existing", "FASTQ already exists")

    try:
        sra_path = raw_dir / run_accession / f"{run_accession}.sra"
        if prefetch and not sra_path.exists():
            _run_subprocess([prefetch, run_accession, "-O", str(raw_dir)])

        if sra_path.exists():
            cmd = [fasterq, str(sra_path), "-O", str(raw_dir), "--split-files", "-e", str(threads)]
        else:
            cmd = [fasterq, run_accession, "-O", str(raw_dir), "--split-files", "-e", str(threads)]

        _run_subprocess(cmd)

        fq1, fq2 = detect_fastqs(run_accession, raw_dir)
        _gzip_fastq(fq1)
        if fq2 is not None:
            _gzip_fastq(fq2)

        return DownloadResult(run_accession, "downloaded", "ok")
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(run_accession, "failed", str(exc))


def download_runs_parallel(
    metadata_csv: Path,
    config: dict,
    threads: int = 1,
    download_workers: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Download SRA runs with optional sample-level parallelism."""
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_dir(raw_dir)

    cleaned_meta_path = Path(config["paths"]["cleaned_metadata"])
    md = validate_and_clean_metadata(metadata_csv, cleaned_meta_path)
    if "run_accession" not in md.columns:
        raise ValueError("Metadata must contain run_accession column")
    if limit is not None:
        md = md.head(limit)

    tools = config["tools"]
    prefetch = tools.get("prefetch_path", "prefetch")
    fasterq = tools.get("fasterq_dump_path", "fasterq-dump")
    check_tool_available(fasterq)
    if prefetch:
        check_tool_available(prefetch)

    if threads < 1:
        raise ValueError("threads must be >= 1")
    if download_workers < 1:
        raise ValueError("download_workers must be >= 1")

    LOGGER.warning(
        "Download concurrency: workers=%d, per-run threads=%d (total potential thread pressure=%d). "
        "High values can oversubscribe CPU/disk.",
        download_workers,
        threads,
        download_workers * threads,
    )

    runs = md["run_accession"].astype(str).tolist()
    results: list[DownloadResult] = []

    if download_workers == 1:
        for idx, run in enumerate(runs, start=1):
            LOGGER.info("[%d/%d] Processing %s", idx, len(runs), run)
            result = download_one_run(run, config, threads)
            if result.status == "failed":
                LOGGER.error("Run %s failed: %s", run, result.message)
            else:
                LOGGER.info("Run %s status: %s", run, result.status)
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = {executor.submit(download_one_run, run, config, threads): run for run in runs}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                run = futures[future]
                result = future.result()
                if result.status == "failed":
                    LOGGER.error("[%d/%d] Run %s failed: %s", completed, len(runs), run, result.message)
                else:
                    LOGGER.info("[%d/%d] Run %s status: %s", completed, len(runs), run, result.status)
                results.append(result)

    out = pd.DataFrame([r.__dict__ for r in results])
    out.to_csv(raw_dir / "download_log.csv", index=False)

    failures = [r for r in results if r.status == "failed"]
    failure_path = raw_dir / "download_failures.csv"
    if failures:
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_accession", "status", "message"])
            writer.writeheader()
            for row in failures:
                writer.writerow(row.__dict__)
        LOGGER.warning("%d downloads failed. See %s", len(failures), failure_path)
    elif failure_path.exists():
        failure_path.unlink()

    return out


def download_runs(
    metadata_csv: Path,
    config: dict,
    limit: int | None = None,
    skip_existing: bool = True,
    threads: int | None = None,
    download_workers: int = 1,
) -> pd.DataFrame:
    """Backward-compatible wrapper for download execution."""
    del skip_existing
    resolved_threads = int(threads if threads is not None else config.get("threads", 1))
    return download_runs_parallel(
        metadata_csv=metadata_csv,
        config=config,
        threads=resolved_threads,
        download_workers=download_workers,
        limit=limit,
    )
