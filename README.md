# fuso-biomarker

Lightweight Python (3.11+) pipeline for exploratory colorectal cancer stool **shotgun metagenome** analysis with a focus on **Fusobacterium nucleatum** features.

## Scientific scope and assumptions
- This pipeline targets **shotgun metagenomic** reads and uses Kraken2 taxonomy profiles (not 16S/amplicon workflows).
- Relative abundance is compositional and should be treated as a proxy signal.
- *F. nucleatum* abundance is a candidate biomarker feature, not evidence of causality.
- The panel taxa features are for exploratory modeling.

## External tools
Expected on PATH (or set in `config/config.yaml`):
- `prefetch` (optional)
- `fasterq-dump`
- `seqtk` (recommended for paired subsampling)
- `fastp`
- `kraken2`

You can control tool multithreading with `threads` in `config/config.yaml` (default `1`) or with `--threads` for `download`, `qc`, `kraken`, and `run-all`.

If tools are missing, scripts raise clear errors.

## Install
Using Conda:
```bash
conda env create -f environment.yml
conda activate fuso-biomarker
```

Using `venv` + pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Metadata format
Input CSV must include:
- `run_accession`
- `sample_id`
- `label` (`healthy`/`cancer`)
- `age_group` (`young`/`old`)
- optional `library_name`

See `metadata/example_metadata.csv`.

## Run steps
```bash
python -m src.cli validate-metadata --metadata metadata/example_metadata.csv
python -m src.cli download --metadata metadata/example_metadata.csv --threads 8
python -m src.cli subsample --metadata metadata/example_metadata.csv --fraction 0.1
python -m src.cli qc --metadata metadata/example_metadata.csv --threads 8
python -m src.cli kraken --metadata metadata/example_metadata.csv --threads 8
python -m src.cli features --metadata metadata/example_metadata.csv
```

Run full pipeline:
```bash
python -m src.cli run-all --metadata metadata/example_metadata.csv --fraction 0.1 --threads 8
```

## Main outputs
- `data/raw/` downloaded FASTQ
- `data/subsampled/` reduced FASTQ + log
- `data/qc/` cleaned gzipped FASTQ (`*.fastq.gz`) + `qc_summary.csv`
- `data/kraken/` Kraken outputs and reports
- `data/features/full_abundance_matrix.csv`
- `data/features/targeted_feature_matrix.csv`
- `reports/stage_read_summary.csv`
- `reports/f_nucleatum_hist.png`
- `reports/panel_detection_counts.png`

## Repository layout
```
README.md
requirements.txt
config/config.yaml
metadata/example_metadata.csv
src/
tests/
data/
reports/
notebooks/
```

## Limitations
- Paired-end subsampling without `seqtk` is intentionally unsupported in fallback mode.
- Kraken database must exist locally and be suitable for stool shotgun data.
- Pipeline is intentionally simple and is not a replacement for HPC-scale workflow engines.
