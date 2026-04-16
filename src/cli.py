from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .bracken import run_bracken
from .download import download_runs
from .features import build_features, build_stage_summary
from .kraken import run_kraken
from .qc import run_qc
from .subsample import subsample_runs
from .utils import load_config, setup_logging, validate_and_clean_metadata

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight CRC microbiome pipeline")
    parser.add_argument("--config", default="config/config.yaml", type=Path)

    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ["validate-metadata", "download", "qc", "kraken", "bracken", "features"]:
        s = sub.add_parser(cmd)
        s.add_argument("--metadata", required=True, type=Path)
        if cmd in {"download", "qc", "kraken", "bracken"}:
            s.add_argument("--threads", type=int, default=None)

    s_sub = sub.add_parser("subsample")
    s_sub.add_argument("--metadata", required=True, type=Path)
    s_sub.add_argument("--fraction", type=float, default=None)
    s_sub.add_argument("--target-reads", type=int, default=None)

    s_all = sub.add_parser("run-all")
    s_all.add_argument("--metadata", required=True, type=Path)
    s_all.add_argument("--fraction", type=float, default=None)
    s_all.add_argument("--target-reads", type=int, default=None)
    s_all.add_argument("--limit", type=int, default=None)
    s_all.add_argument("--threads", type=int, default=None)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "validate-metadata":
        out = Path(config["paths"]["cleaned_metadata"])
        df = validate_and_clean_metadata(args.metadata, out)
        LOGGER.info("Validated metadata rows: %d", len(df))
        return

    if args.command == "download":
        download_runs(args.metadata, config, threads=args.threads)
    elif args.command == "subsample":
        subsample_runs(args.metadata, config, fraction=args.fraction, target_reads=args.target_reads)
    elif args.command == "qc":
        run_qc(args.metadata, config, threads=args.threads)
    elif args.command == "kraken":
        run_kraken(args.metadata, config, threads=args.threads)
    elif args.command == "bracken":
        run_bracken(args.metadata, config, threads=args.threads)
    elif args.command == "features":
        build_features(args.metadata, config)
        build_stage_summary(config, args.metadata)
    elif args.command == "run-all":
        download_runs(args.metadata, config, limit=args.limit, threads=args.threads)
        subsample_runs(args.metadata, config, fraction=args.fraction, target_reads=args.target_reads)
        run_qc(args.metadata, config, threads=args.threads)
        run_kraken(args.metadata, config, threads=args.threads)
        run_bracken(args.metadata, config, threads=args.threads)
        build_features(args.metadata, config)
        build_stage_summary(config, args.metadata)


if __name__ == "__main__":
    main()
