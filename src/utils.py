from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_METADATA_COLUMNS = ["run_accession", "sample_id", "label", "age_group"]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def check_tool_available(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(
            f"Required tool '{tool_name}' was not found in PATH. "
            "Install it or update config.yaml tool paths."
        )


def run_command(cmd: list[str], logger: logging.Logger, cwd: Path | None = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("Command failed (%s): %s", proc.returncode, " ".join(cmd))
        if proc.stdout:
            logger.error("stdout: %s", proc.stdout.strip())
        if proc.stderr:
            logger.error("stderr: %s", proc.stderr.strip())
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def validate_and_clean_metadata(
    metadata_path: Path,
    output_path: Path | None = None,
    runs: list[str] | None = None,
) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    df = df.copy()
    df["run_accession"] = df["run_accession"].astype(str).str.strip()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["age_group"] = df["age_group"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=REQUIRED_METADATA_COLUMNS)
    df = df[df["run_accession"] != ""]

    if runs:
        run_set = set(runs)
        df = df[df["run_accession"].isin(run_set)]

    df = df.drop_duplicates(subset=["run_accession"]).reset_index(drop=True)

    if output_path is not None:
        ensure_dir(output_path.parent)
        df.to_csv(output_path, index=False)

    return df


def detect_fastqs(run_accession: str, base_dir: Path) -> tuple[Path, Path | None]:
    r1 = base_dir / f"{run_accession}_1.fastq"
    r2 = base_dir / f"{run_accession}_2.fastq"
    single = base_dir / f"{run_accession}.fastq"

    if r1.exists() and r2.exists():
        return r1, r2
    if single.exists():
        return single, None

    gz_single = base_dir / f"{run_accession}.fastq.gz"
    gz_r1 = base_dir / f"{run_accession}_1.fastq.gz"
    gz_r2 = base_dir / f"{run_accession}_2.fastq.gz"
    if gz_r1.exists() and gz_r2.exists():
        return gz_r1, gz_r2
    if gz_single.exists():
        return gz_single, None

    raise FileNotFoundError(f"No FASTQ files found for {run_accession} in {base_dir}")


def count_fastq_reads(path: Path) -> int:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        lines = sum(1 for _ in handle)
    return lines // 4


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
