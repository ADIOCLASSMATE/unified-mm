"""Report the Long download, migrated supplements and independent 512px services."""
import argparse
import json
from pathlib import Path
import time

from data_synthesis.config import DEFAULT_CONFIG, load_config
from data_synthesis.io import atomic_json
from data_synthesis.integrity import hashing_enabled


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def collect(config):
    root = Path(config["preparation_root"])
    legacy = root / "legacy_v3/downloads"
    archives = []
    for row in read(legacy / "active_archive_jobs.json").get("source_pools", []):
        argv = row["argv"]
        run = Path(argv[argv.index("--root") + 1])
        value = read(run / "archive_download_status.json")
        archives.append({"name": row["group"], "status_file": str(run / "archive_download_status.json"),
                         "pid_alive": alive(value.get("pid", row["pid"])), **value})
    extra = legacy / "textcaps_official_20260913/archive_download_status.json"
    if extra.exists():
        value = read(extra)
        archives.append({"name": "textcaps_metadata", "status_file": str(extra),
                         "pid_alive": alive(value.get("pid", 0)), **value})
    sources = []
    for row in read(legacy / "source_jobs.json").get("jobs", []):
        source = Path(row["status"]).parent
        sources.append({"name": row["name"], "pid_alive": alive(row["pid"]),
                        "job": read(row["status"]), "candidates": read(source / "selected.candidate_status.json"),
                        "download": read(row["download_status"]), "prepare": read(row["prepare_status"])})
    long = root / "supply/blip3o_long_v1"
    long_prepare = read(long / "prepare_status.json")
    review_launch = read(root / "active_sii_review.json")
    review = None
    if review_launch:
        controller = read(Path(review_launch["root"]) / "status.json")
        shard = read(Path(controller["active_shard"]) / "status.json") if controller.get("active_shard") else {}
        review = {"launch": review_launch, "pid_alive": alive(review_launch["pid"]),
                  "controller": controller, "active_shard": shard}
    synthesis_started = bool(review and (
        review["controller"].get("completed_images", review["controller"].get("reviewed_images", 0))
        or any(n for state, n in review["active_shard"].get("counts", {}).items() if state != "pending")))
    totals = {"archive_objects": sum(a.get("objects", 0) for a in archives),
              "archive_objects_verified": sum(a.get("counts", {}).get("verified", 0) for a in archives),
              "archive_declared_bytes": sum(a.get("declared_bytes", 0) for a in archives),
              "archive_verified_bytes_including_local_reuse": sum(a.get("verified_bytes", 0) for a in archives),
              "archive_network_bytes_recorded_this_run": sum(a.get("network_bytes", 0) for a in archives),
              "url_images_downloaded": sum(s["download"].get("counts", {}).get("downloaded", 0) for s in sources),
              "url_download_failures": sum(s["download"].get("counts", {}).get("failed", 0) for s in sources),
              "prepared_512_views_before_global_dedup": long_prepare.get("prepared", 0) + sum(s["prepare"].get("prepared", 0) for s in sources),
              "preparation_rejections": long_prepare.get("rejected", 0) + sum(s["prepare"].get("rejected", 0) for s in sources)}
    return {"updated_at": time.time(), "source_plan": config["source_plan"],
            "compute_hashes": hashing_enabled(config),
            "image_counting": "known_id_url_reference_dedup; content uniqueness not verified" if not hashing_enabled(config) else "content_hash_dedup",
            "network_policy": "direct_per_client_global_proxy_unchanged", "bulk_synthesis_started": synthesis_started,
            "sii_review": review,
            "target_accepted_non_imagenet_images": config["minimum_images"], "quotas_are_caps": False,
            "storage": {k: config[k] for k in ("image_root", "text_root", "preparation_root", "posterior_root")},
            "totals": totals, "archives": archives, "sources": sources,
            "long": {"intake": read(long / "intake_status.json"), "prepare": long_prepare,
                     "services": {k: {**v, "pid_alive": alive(v["pid"])}
                                  for k, v in read(long / "services.json").items()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["preparation_root"])
    while True:
        report = collect(config)
        atomic_json(root / "status.json", report)
        legacy = root / "legacy_v3/downloads"
        if legacy.is_dir():
            atomic_json(legacy / "status.json", report)
        print(json.dumps(report["totals"]), flush=True)
        if not args.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
