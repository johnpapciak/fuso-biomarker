from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .utils import check_tool_available, detect_fastqs, ensure_dir, run_command, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


COLS = [
    "percentage",
    "clade_reads",
    "taxon_reads",
    "rank_code",
    "taxid",
    "name",
]


def parse_kraken_report(report_path: Path, sample_id: str) -> pd.DataFrame:
    if not report_path.exists() or report_path.stat().st_size == 0:
        return pd.DataFrame(columns=["sample_id", *COLS])
    df = pd.read_csv(report_path, sep="\t", header=None, names=COLS)
    df["name"] = df["name"].astype(str).str.strip()
    df["sample_id"] = sample_id
    return df[["sample_id", *COLS]]


def run_kraken(metadata_csv: Path, config: dict, skip_existing: bool = True) -> pd.DataFrame:
    kraken2 = config["tools"].get("kraken2_path", "kraken2")
    check_tool_available(kraken2)

    db = config.get("kraken_db")
    if not db or not Path(db).exists():
        raise FileNotFoundError("kraken_db is missing/invalid in config.yaml")

    qc_dir = Path(config["paths"]["qc_dir"])
    out_dir = Path(config["paths"]["kraken_dir"])
    ensure_dir(out_dir)

    conf = config.get("kraken_confidence")
    md = validate_and_clean_metadata(metadata_csv)
    logs: list[dict[str, str]] = []

    for _, row in md.iterrows():
        run = row["run_accession"]
        sample_id = row["sample_id"]
        try:
            in1, in2 = detect_fastqs(f"{run}_qc", qc_dir)
            report = out_dir / f"{run}.report.tsv"
            output = out_dir / f"{run}.kraken.tsv"
            if skip_existing and report.exists() and output.exists():
                logs.append({"run_accession": run, "status": "skipped_existing"})
                continue

            cmd = [kraken2, "--db", str(db), "--report", str(report), "--output", str(output)]
            if conf is not None:
                cmd.extend(["--confidence", str(conf)])
            if in2 is not None:
                cmd.extend(["--paired", str(in1), str(in2)])
            else:
                cmd.append(str(in1))

            run_command(cmd, LOGGER)
            logs.append({"run_accession": run, "sample_id": sample_id, "status": "kraken_done"})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Kraken failed for %s: %s", run, exc)
            logs.append({"run_accession": run, "sample_id": sample_id, "status": f"failed: {exc}"})

    out = pd.DataFrame(logs)
    out.to_csv(out_dir / "kraken_log.csv", index=False)
    return out


def collect_abundance_tables(metadata_df: pd.DataFrame, kraken_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_rows: list[pd.DataFrame] = []
    genus_rows: list[pd.DataFrame] = []
    for _, row in metadata_df.iterrows():
        run = row["run_accession"]
        sample_id = row["sample_id"]
        report_df = parse_kraken_report(kraken_dir / f"{run}.report.tsv", sample_id)
        if report_df.empty:
            continue
        sp = report_df[report_df["rank_code"] == "S"][["sample_id", "name", "percentage", "clade_reads"]].copy()
        ge = report_df[report_df["rank_code"] == "G"][["sample_id", "name", "percentage", "clade_reads"]].copy()
        sp["level"] = "species"
        ge["level"] = "genus"
        species_rows.append(sp)
        genus_rows.append(ge)

    sp_df = pd.concat(species_rows, ignore_index=True) if species_rows else pd.DataFrame()
    ge_df = pd.concat(genus_rows, ignore_index=True) if genus_rows else pd.DataFrame()
    return sp_df, ge_df
