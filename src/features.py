from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .kraken import collect_abundance_tables, parse_kraken_report
from .utils import ensure_dir, read_json, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def shannon_diversity(relative_abundances: list[float]) -> float:
    arr = np.array([x for x in relative_abundances if x > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def build_features(metadata_csv: Path, config: dict) -> pd.DataFrame:
    features_dir = Path(config["paths"]["features_dir"])
    kraken_dir = Path(config["paths"]["kraken_dir"])
    qc_dir = Path(config["paths"]["qc_dir"])
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    ensure_dir(features_dir)
    ensure_dir(reports_dir)

    md = validate_and_clean_metadata(metadata_csv)
    species_df, genus_df = collect_abundance_tables(md, kraken_dir)

    full_long = pd.concat([species_df, genus_df], ignore_index=True) if not species_df.empty or not genus_df.empty else pd.DataFrame()
    if not full_long.empty:
        full_long["relative_abundance"] = full_long["percentage"] / 100.0
        full_matrix = full_long.pivot_table(index="sample_id", columns="name", values="relative_abundance", aggfunc="sum", fill_value=0)
    else:
        full_matrix = pd.DataFrame(index=md["sample_id"])

    full_matrix.to_csv(features_dir / "full_abundance_matrix.csv")

    panel_taxa = config.get("panel_taxa", ["Fusobacterium nucleatum"])
    pseudocount = float(config.get("pseudocount", 1e-6))
    pa_thresh = float(config.get("presence_absence_threshold", 1e-4))

    rows: list[dict] = []
    for _, mrow in md.iterrows():
        run = mrow["run_accession"]
        sample_id = mrow["sample_id"]
        report = parse_kraken_report(kraken_dir / f"{run}.report.tsv", sample_id)

        species = report[report["rank_code"] == "S"].copy()
        genus = report[report["rank_code"] == "G"].copy()
        species["ra"] = species["percentage"] / 100.0
        genus["ra"] = genus["percentage"] / 100.0

        s_map = dict(zip(species["name"], species["ra"]))
        g_map = dict(zip(genus["name"], genus["ra"]))

        f_nuc = s_map.get("Fusobacterium nucleatum", 0.0)
        fuso_gen = g_map.get("Fusobacterium", 0.0)
        total_bact = report.loc[report["name"].eq("Bacteria"), "percentage"].div(100).max() if not report.empty else 0.0
        total_bact = float(0.0 if pd.isna(total_bact) else total_bact)
        n_species = int((species["ra"] > 0).sum()) if not species.empty else 0
        shannon = shannon_diversity(species["ra"].tolist())

        qc_json = qc_dir / "reports" / f"{run}.fastp.json"
        depth = 0
        if qc_json.exists():
            try:
                depth = int(read_json(qc_json).get("summary", {}).get("after_filtering", {}).get("total_reads", 0))
            except Exception:  # noqa: BLE001
                depth = 0

        rec = {
            "sample_id": sample_id,
            "run_accession": run,
            "label": mrow["label"],
            "age_group": mrow["age_group"],
            "library_name": mrow.get("library_name", ""),
            "F_nucleatum_species_abundance": f_nuc,
            "Fusobacterium_genus_abundance": fuso_gen,
            "total_bacterial_abundance": total_bact,
            "number_of_detected_species": n_species,
            "Shannon_diversity": shannon,
            "sequencing_depth_proxy": depth,
        }

        for taxon in panel_taxa:
            col = f"panel_{taxon.replace(' ', '_')}"
            val = s_map.get(taxon, g_map.get(taxon, 0.0))
            rec[col] = val
            rec[f"{col}_log10"] = np.log10(val + pseudocount)
            rec[f"{col}_present"] = int(val >= pa_thresh)

        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(features_dir / "targeted_feature_matrix.csv", index=False)

    if not out_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(out_df["F_nucleatum_species_abundance"], bins=20)
        ax.set_xlabel("F. nucleatum relative abundance")
        ax.set_ylabel("Samples")
        fig.tight_layout()
        fig.savefig(reports_dir / "f_nucleatum_hist.png", dpi=150)
        plt.close(fig)

        detect_cols = [c for c in out_df.columns if c.endswith("_present")]
        if detect_cols:
            fig_height = max(3.5, 0.35 * len(detect_cols))
            fig, ax = plt.subplots(figsize=(8, fig_height))
            out_df[detect_cols].sum().sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_ylabel("Detected samples")
            ax.set_xlabel("Panel taxa")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(reports_dir / "panel_detection_counts.png", dpi=150)
            plt.close(fig)

    return out_df


def build_stage_summary(config: dict, metadata_csv: Path) -> pd.DataFrame:
    md = validate_and_clean_metadata(metadata_csv)
    raw_dir = Path(config["paths"]["raw_dir"])
    subsampled_dir = Path(config["paths"]["subsampled_dir"])
    qc_summary = Path(config["paths"]["qc_dir"]) / "qc_summary.csv"

    recs = []
    qc_df = pd.read_csv(qc_summary) if qc_summary.exists() else pd.DataFrame()
    sub_log = subsampled_dir / "subsample_log.csv"
    sub_df = pd.read_csv(sub_log) if sub_log.exists() else pd.DataFrame()

    for _, row in md.iterrows():
        run = row["run_accession"]
        rec = {"run_accession": run, "sample_id": row["sample_id"]}
        rec["raw_exists"] = any(raw_dir.glob(f"{run}*.fastq*"))
        if not sub_df.empty and run in set(sub_df["run_accession"]):
            r = sub_df[sub_df["run_accession"] == run].iloc[0]
            rec["subsampled_reads"] = r.get("subsampled_reads", None)
        if not qc_df.empty and run in set(qc_df["run_accession"]):
            r = qc_df[qc_df["run_accession"] == run].iloc[0]
            rec["qc_reads_after"] = r.get("total_reads_after", None)
        recs.append(rec)

    out = pd.DataFrame(recs)
    out_path = Path(config["paths"].get("reports_dir", "reports")) / "stage_read_summary.csv"
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    return out
