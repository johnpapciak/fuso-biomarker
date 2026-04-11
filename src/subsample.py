from __future__ import annotations

import logging
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .utils import count_fastq_reads, detect_fastqs, ensure_dir, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def _python_subsample_single_end(inp: Path, out: Path, fraction: float, seed: int = 42) -> None:
    """Fallback subsampling for single-end FASTQ only."""
    import gzip

    random.seed(seed)
    opener = gzip.open if inp.suffix == ".gz" else open
    out_opener = gzip.open if out.suffix == ".gz" else open
    with opener(inp, "rt", encoding="utf-8", errors="ignore") as fin, out_opener(out, "wt", encoding="utf-8") as fout:
        while True:
            rec = [fin.readline() for _ in range(4)]
            if not rec[0]:
                break
            if random.random() < fraction:
                fout.write("".join(rec))


def _seqtk_subsample(seqtk: str, inp: Path, out: Path, fraction: float) -> None:
    cmd = f"{seqtk} sample -s42 {inp} {fraction}"
    if out.suffix == ".gz":
        cmd = f"{cmd} | gzip -c > {out}"
    else:
        cmd = f"{cmd} > {out}"
    proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"seqtk failed for {inp}")


def subsample_runs(
    metadata_csv: Path,
    config: dict,
    fraction: float | None = None,
    target_reads: int | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(config["paths"]["raw_dir"])
    out_dir = Path(config["paths"]["subsampled_dir"])
    ensure_dir(out_dir)

    seqtk = config["tools"].get("seqtk_path", "seqtk")
    has_seqtk = shutil.which(seqtk) is not None

    md = validate_and_clean_metadata(metadata_csv)
    records: list[dict[str, str | int | float]] = []

    for run in md["run_accession"]:
        try:
            in1, in2 = detect_fastqs(run, raw_dir)
            out1 = out_dir / in1.name
            out2 = out_dir / in2.name if in2 else None
            if skip_existing and out1.exists() and ((out2 and out2.exists()) or out2 is None):
                records.append({"run_accession": run, "status": "skipped_existing"})
                continue

            total = count_fastq_reads(in1)
            use_fraction = fraction if fraction is not None else config.get("subsample_fraction", 0.1)
            if target_reads is not None:
                use_fraction = min(1.0, max(0.0, target_reads / max(total, 1)))

            if in2 is not None and not has_seqtk:
                raise RuntimeError("Paired-end subsampling requires seqtk in this lightweight pipeline.")

            if has_seqtk:
                _seqtk_subsample(seqtk, in1, out1, use_fraction)
                if in2 and out2:
                    _seqtk_subsample(seqtk, in2, out2, use_fraction)
            else:
                _python_subsample_single_end(in1, out1, use_fraction)

            sub_n = count_fastq_reads(out1)
            records.append(
                {
                    "run_accession": run,
                    "status": "subsampled",
                    "original_reads": total,
                    "subsampled_reads": sub_n,
                    "fraction": use_fraction,
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Subsampling failed for %s: %s", run, exc)
            records.append({"run_accession": run, "status": f"failed: {exc}"})

    out = pd.DataFrame(records)
    out.to_csv(out_dir / "subsample_log.csv", index=False)
    return out
