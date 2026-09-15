"""One current entry point for source intake, frozen pools, synthesis and release."""
import argparse
import asyncio
import json
from pathlib import Path

from data_synthesis.config import DEFAULT_CONFIG, load_config, load_sii_settings
from data_synthesis.io import dumps
from data_synthesis.integrity import hashing_enabled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Show current source targets and runtime without reading a key")
    sub.add_parser("check-connection", help="Validate shell SII settings without printing credentials or invoking a model")
    intake = sub.add_parser("ingest", help="Import normalized source candidates; preserve all caption/annotation fields")
    intake.add_argument("--manifest", required=True)
    intake.add_argument("--supply-root", required=True)
    intake.add_argument("--prefix", required=True)
    seal = sub.add_parser("seal-candidates", help="Freeze the complete image download scope after all source intakes")
    seal.add_argument("--supply-root", required=True)
    down = sub.add_parser("download", help="Independent direct image download service")
    prep = sub.add_parser("prepare", help="Independent CPU 512px preprocessing service")
    for command in (down, prep):
        command.add_argument("--supply-root", required=True)
        command.add_argument("--image-root")
        command.add_argument("--workers", type=int, default=64 if command is down else 8)
    down.add_argument("--per-host", type=int, default=8)
    down.add_argument("--request-timeout", type=float, default=90)
    prep.add_argument("--root", help="Synthesis root receiving sealed view batches")
    prep.add_argument("--exclude", required=True)
    prep.add_argument("--near-exclude-index")
    freeze = sub.add_parser("freeze", help="Join all closed inputs, deduplicate and freeze the accepted image pool")
    freeze.add_argument("--root")
    freeze.add_argument("--inbox", action="append", default=[], help="Repeat for multiple completed download cohorts")
    freeze.add_argument("--prepared-run", action="append", default=[])
    freeze.add_argument("--release", action="append", default=[])
    freeze.add_argument("--exclude")
    freeze.add_argument("--near-exclude-index")
    freeze.add_argument("--exclude-prompts")
    freeze.add_argument("--pilot", action="store_true", help="Explicit small development pool, never a production release")
    run = sub.add_parser("run", help="Current reuse-first local routing and targeted SII repair")
    run.add_argument("--root")
    run.add_argument("--qualification", type=Path)
    run.add_argument("--selection", type=Path)
    run.add_argument("--max-items", type=int)
    run.add_argument("--tokenizer")
    status = sub.add_parser("status")
    status.add_argument("--root")
    quarantine = sub.add_parser("quarantine-failed", help="Explicitly exclude terminal failures, preserving attempts; production minimum still applies")
    quarantine.add_argument("--root")
    bank = sub.add_parser("prepare-bank", help="Freeze an image-only manifest for independent KL16 encoding")
    bank.add_argument("--root")
    bank.add_argument("--output", required=True)
    export = sub.add_parser("export")
    export.add_argument("--root")
    export.add_argument("--output", required=True)
    export.add_argument("--tokenizer")
    export.add_argument("--shard-records", type=int, help="Defaults to runtime export_shard_records")
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--tokenizer")
    audit.add_argument("--require-posterior", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = getattr(args, "root", None) or config["state_root"]
    if args.command == "plan":
        value = {"runtime": config, "source_plan": json.loads(Path(config["source_plan"]).read_text())}
    elif args.command == "check-connection":
        settings = load_sii_settings()
        value = {"status": "configured", "credential_source": "SII_API_KEY / SII_BASE_URL from environment or literal shell rc exports",
                 "key_present": bool(settings.api_key), "base_url_present": bool(settings.base_url),
                 "protocol": config["sii"]["protocol"], "models": config["sii"]["vision_models"], "proxy": False}
    elif args.command == "ingest":
        from data_synthesis.sources import ingest_candidates
        value = ingest_candidates(args.manifest, args.supply_root, args.prefix,
                                  compute_hashes=hashing_enabled(config))
    elif args.command == "seal-candidates":
        from data_synthesis.sources import seal_candidates
        value = seal_candidates(args.supply_root, compute_hashes=hashing_enabled(config))
    elif args.command in {"download", "prepare"}:
        from scripts.supply_b512_images import download_service, prepare_service
        args.image_root = args.image_root or config["image_root"]
        args.compute_hashes = hashing_enabled(config)
        args.root = args.supply_root
        if args.workers < 1:
            parser.error("workers must be positive")
        if args.command == "download":
            from data_synthesis.sources import seal_candidates
            if not (Path(args.supply_root) / "candidates.closed.json").is_file():
                parser.error("seal-candidates before starting the current download cohort")
            seal_candidates(args.supply_root, compute_hashes=args.compute_hashes)
            args.direct_dns_hosts, args.direct_dns_host, args.direct_dns_rps = [], [], 2.0
            if args.per_host < 1 or args.request_timeout <= 0:
                parser.error("per-host and timeout must be positive")
            value = asyncio.run(download_service(args))
        else:
            args.farm_root = root
            Path(root).mkdir(parents=True, exist_ok=True)
            value = asyncio.run(prepare_service(args))
    elif args.command == "freeze":
        from data_synthesis.sources import freeze as freeze_pool
        value = freeze_pool(root, config, prepared=args.prepared_run, releases=args.release, inbox=args.inbox,
                            exclude=args.exclude, near_index=args.near_exclude_index, exclude_prompts=args.exclude_prompts, pilot=args.pilot)
    elif args.command == "status":
        if config.get("review_mode") == "reuse_first_targeted":
            value = json.loads((Path(root) / "status.json").read_text())
        else:
            from data_synthesis.state import State
            with State(root, readonly=True) as state:
                value = {"counts": state.counts(), "frozen_pool": state.meta("frozen"), "scheduler": state.meta("scheduler")}
    elif args.command == "run" and config.get("review_mode") == "reuse_first_targeted":
        from data_synthesis.reuse_farm import run as run_reuse
        if args.max_items is not None:
            parser.error("use a separate fixed selection for a bounded reuse-first test")
        value = asyncio.run(run_reuse(root, config, args.selection or config["prepared_selection"],
            args.qualification or Path(root) / "user_acceptance.json"))
    elif args.command == "quarantine-failed":
        from data_synthesis.state import State
        with State(root, config) as state:
            value = state.quarantine_failed(config)
    elif args.command == "prepare-bank":
        from scripts.prepare_b512_posterior_bank import prepare_bank
        value = prepare_bank(root, args.output, compute_hashes=hashing_enabled(config))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or config["tokenizer"], local_files_only=True)
        if args.command == "run":
            from data_synthesis.pipeline import run as run_pipeline
            value = asyncio.run(run_pipeline(root, config, tokenizer=tokenizer, max_items=args.max_items,
                                            qualification_path=args.qualification))
        elif args.command == "export":
            from data_synthesis.publication import export as export_text
            value = export_text(root, args.output, tokenizer=tokenizer,
                                shard_records=(args.shard_records if args.shard_records is not None
                                               else config.get("export_shard_records", 250000)))
        else:
            from data_synthesis.publication import audit as audit_text
            value = audit_text(args.dataset, tokenizer=tokenizer, require_posterior=args.require_posterior)
    print(dumps(value))
    if isinstance(value, dict) and value.get("state") in {"needs_more_images", "blocked", "incomplete", "interrupted"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
