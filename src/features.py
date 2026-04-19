from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .kraken import parse_kraken_report
from .utils import ensure_dir, read_json, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def _load_bracken_abundance_tables(bracken_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_path = bracken_dir / "bracken_abundance_species.csv"
    genus_path = bracken_dir / "bracken_abundance_genus.csv"

    species_df = pd.read_csv(species_path) if species_path.exists() and species_path.stat().st_size > 0 else pd.DataFrame()
    genus_df = pd.read_csv(genus_path) if genus_path.exists() and genus_path.stat().st_size > 0 else pd.DataFrame()

    if not species_df.empty:
        species_df = species_df.rename(columns={"fraction_total_reads": "relative_abundance", "name": "taxon"}).copy()
        species_df["level"] = "species"

    if not genus_df.empty:
        genus_df = genus_df.rename(columns={"fraction_total_reads": "relative_abundance", "name": "taxon"}).copy()
        genus_df["level"] = "genus"

    return species_df, genus_df


def shannon_diversity(relative_abundances: list[float]) -> float:
    arr = np.array([x for x in relative_abundances if x > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def _resolve_auto_panel_levels(
    auto_panel_levels: str,
    species_df: pd.DataFrame,
    genus_df: pd.DataFrame,
) -> list[str]:
    if auto_panel_levels == "species":
        return ["species"]
    if auto_panel_levels == "genus":
        return ["genus"]
    levels = []
    if not species_df.empty:
        levels.append("species")
    if not genus_df.empty:
        levels.append("genus")
    return levels


def _score_taxa(level_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if level_df.empty:
        return pd.DataFrame(columns=["taxon", "score"])
    grouped = level_df.groupby("taxon")["relative_abundance"]
    mean_abundance = grouped.mean()
    prevalence = grouped.apply(lambda s: float((s > 0).mean()))
    if metric == "mean_abundance":
        score = mean_abundance
    elif metric == "prevalence":
        score = prevalence
    else:
        score = mean_abundance * prevalence
    return (
        pd.DataFrame({"taxon": score.index, "score": score.values})
        .sort_values(["score", "taxon"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_features(
    metadata_csv: Path,
    config: dict,
    auto_panel_top_n: int = 0,
    auto_panel_levels: str = "both",
    auto_panel_metric: str = "mean_x_prevalence",
) -> pd.DataFrame:
    features_dir = Path(config["paths"]["features_dir"])
    kraken_dir = Path(config["paths"]["kraken_dir"])
    bracken_dir = Path(config["paths"].get("bracken_dir", "data/bracken"))
    qc_dir = Path(config["paths"]["qc_dir"])
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    ensure_dir(features_dir)
    ensure_dir(reports_dir)

    md = validate_and_clean_metadata(metadata_csv)
    species_df, genus_df = _load_bracken_abundance_tables(bracken_dir)
    if species_df.empty and genus_df.empty:
        LOGGER.warning("No combined Bracken abundance tables found in %s; full abundance matrix will be empty", bracken_dir)

    full_long = pd.concat([species_df, genus_df], ignore_index=True) if not species_df.empty or not genus_df.empty else pd.DataFrame()
    if not full_long.empty:
        full_matrix = full_long.pivot_table(
            index="sample_id",
            columns="taxon",
            values="relative_abundance",
            aggfunc="sum",
            fill_value=0,
        )
    else:
        full_matrix = pd.DataFrame(index=md["sample_id"])

    full_matrix.to_csv(features_dir / "full_abundance_matrix.csv")

    auto_panel_cfg = config.get("auto_panel", {})
    top_n = int(auto_panel_top_n if auto_panel_top_n is not None else auto_panel_cfg.get("top_n", 0))
    level_opt = auto_panel_levels if auto_panel_levels is not None else auto_panel_cfg.get("levels", "both")
    metric_opt = auto_panel_metric if auto_panel_metric is not None else auto_panel_cfg.get("metric", "mean_x_prevalence")

    panel_taxa = list(config.get("panel_taxa", ["Fusobacterium nucleatum"]))
    pseudocount = float(config.get("pseudocount", 1e-6))
    pa_thresh = float(config.get("presence_absence_threshold", 1e-4))
    dynamic_taxa: list[str] = []
    dynamic_recs: list[dict] = []

    selected_levels = _resolve_auto_panel_levels(level_opt, species_df, genus_df)
    if top_n > 0 and selected_levels:
        if selected_levels == ["species", "genus"]:
            species_n = int(np.ceil(top_n / 2))
            genus_n = int(np.floor(top_n / 2))
            n_by_level = {"species": species_n, "genus": genus_n}
        else:
            n_by_level = {selected_levels[0]: top_n}

        level_frames = {"species": species_df, "genus": genus_df}
        for level in selected_levels:
            scores = _score_taxa(level_frames[level], metric_opt)
            n_pick = n_by_level.get(level, 0)
            if n_pick <= 0 or scores.empty:
                continue
            chosen = scores.head(n_pick).copy()
            for rank, row in enumerate(chosen.itertuples(index=False), start=1):
                dynamic_taxa.append(str(row.taxon))
                dynamic_recs.append(
                    {
                        "level": level,
                        "taxon": str(row.taxon),
                        "metric": metric_opt,
                        "score": float(row.score),
                        "rank": rank,
                    }
                )

    merged_panel_taxa = list(dict.fromkeys(panel_taxa + dynamic_taxa))
    pd.DataFrame(dynamic_recs).to_csv(features_dir / "auto_panel_taxa.csv", index=False)

    rows: list[dict] = []
    for _, mrow in md.iterrows():
        run = mrow["run_accession"]
        sample_id = mrow["sample_id"]
        species_rows = species_df[species_df["run_accession"] == run].copy() if not species_df.empty else pd.DataFrame()
        genus_rows = genus_df[genus_df["run_accession"] == run].copy() if not genus_df.empty else pd.DataFrame()
        if genus_rows.empty and not species_rows.empty:
            genus_rows = species_rows.copy()
            genus_rows["taxon"] = genus_rows["taxon"].astype(str).str.split().str[0]
            genus_rows = (
                genus_rows.groupby(["run_accession", "sample_id", "taxon"], as_index=False)["relative_abundance"].sum()
            )

        s_map = (
            species_rows.groupby("taxon")["relative_abundance"].sum().to_dict()
            if not species_rows.empty
            else {}
        )
        g_map = (
            genus_rows.groupby("taxon")["relative_abundance"].sum().to_dict()
            if not genus_rows.empty
            else {}
        )

        f_nuc = s_map.get("Fusobacterium nucleatum", 0.0)
        fuso_gen = g_map.get("Fusobacterium", 0.0)

        report = parse_kraken_report(kraken_dir / f"{run}.report.tsv", sample_id)
        total_bact = report.loc[report["name"].eq("Bacteria"), "percentage"].div(100).max() if not report.empty else 0.0
        total_bact = float(0.0 if pd.isna(total_bact) else total_bact)
        n_species = int((species_rows["relative_abundance"] > 0).sum()) if not species_rows.empty else 0
        shannon = shannon_diversity(species_rows["relative_abundance"].tolist()) if not species_rows.empty else 0.0

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

        for taxon in merged_panel_taxa:
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
