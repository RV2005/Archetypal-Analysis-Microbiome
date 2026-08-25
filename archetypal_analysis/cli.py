"""Command-line interface for the end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _cmd_run(args) -> int:
    from .pipeline.config import load_config
    from .pipeline.runner import PipelineRunner

    cfg = load_config(args.config)
    runner = PipelineRunner(cfg, verbose=not args.quiet)
    result = runner.run(args.stages, overwrite=args.overwrite)
    if not args.quiet:
        print("\nCompleted stages:", ", ".join(result))
        print("Artifacts:", runner.store.root)
    return 0


def _cmd_status(args) -> int:
    from .pipeline.config import load_config
    from .pipeline.runner import PipelineRunner

    runner = PipelineRunner(load_config(args.config), verbose=False)
    for row in runner.status():
        state = "complete" if row["complete"] else "missing"
        duration = "" if row["duration_seconds"] is None else f"{row['duration_seconds']:.3f}s"
        print(f"{row['stage']:10s} {state:8s} {duration:>10s} {row['completed_at'] or ''}")
    return 0


def _cmd_resolve(args) -> int:
    from .pipeline.config import load_config

    cfg = load_config(args.config)
    print(yaml.safe_dump({k: v for k, v in cfg.items() if k != "_meta"}, sort_keys=False))
    return 0


def _cmd_convert(args) -> int:
    from .data.conversion import convert_compendium_dataset, convert_direct_dataset

    if args.kind == "compendium":
        otu, metadata = convert_compendium_dataset(
            args.otu,
            args.metadata,
            args.output,
            projects_file=args.projects,
            tags_file=args.tags,
        )
    else:
        otu, metadata = convert_direct_dataset(args.otu, args.metadata, args.output, args.sample_id_column)
    print(f"Wrote {otu}")
    print(f"Wrote {metadata}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aa-pipeline", description="Config-driven archetypal-analysis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run all or selected stages")
    run.add_argument("--config", required=True, help="Experiment YAML")
    run.add_argument("--stages", nargs="+", default=["all"], help="Stages (space/comma separated) or 'all'")
    run.add_argument("--overwrite", action="store_true", help="Re-run explicitly requested stages even if reusable")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=_cmd_run)

    status = sub.add_parser("status", help="Show stage completion markers")
    status.add_argument("--config", required=True)
    status.set_defaults(func=_cmd_status)

    resolve = sub.add_parser("resolve-config", help="Print inherited/merged YAML")
    resolve.add_argument("--config", required=True)
    resolve.set_defaults(func=_cmd_resolve)

    convert = sub.add_parser("convert", help="Convert a legacy dataset to otu.tsv + metadata.tsv")
    convert.add_argument("--kind", choices=["direct", "compendium"], required=True)
    convert.add_argument("--otu", required=True)
    convert.add_argument("--metadata", required=True)
    convert.add_argument("--projects")
    convert.add_argument("--tags")
    convert.add_argument("--sample-id-column", default="sample")
    convert.add_argument("--output", required=True)
    convert.set_defaults(func=_cmd_convert)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())