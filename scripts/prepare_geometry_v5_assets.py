"""Download pinned official comparison assets without any HTTP/SOCKS proxy.

Only explicitly selected model files are downloaded. Files are resumed to .part,
checked against the published byte length, and atomically committed. No data or
weight hashes are computed. Git revisions are public provenance identifiers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/evaluation/research/cross-model-geometry-v5-20260907"
MODELS = ROOT / "public/models"
REPOSITORIES = {
    "deepseek-ai/JanusFlow-1.3B": None,
    "showlab/show-o2-1.5B": None,
    "google/siglip-so400m-patch14-384": None,
    "Qwen/Qwen2.5-1.5B-Instruct": None,
    "facebook/dinov2-base": None,
    "facebook/vit-mae-base": None,
    "stabilityai/sdxl-vae": [
        "diffusion_pytorch_model.safetensors",
        "config.json",
        "README.md",
    ],
    "Wan-AI/Wan2.1-T2V-14B": ["Wan2.1_VAE.pth"],
}
SOURCES = ("deepseek-ai/Janus", "showlab/Show-o")


def emit(event, **kw):
    print(
        json.dumps(
            {
                "event": event,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **kw,
            }
        ),
        flush=True,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def direct_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url, headers={"User-Agent": "unified-mm-geometry-v5"}
    )
    with opener.open(request, timeout=60) as response:
        return json.load(response)


def allowed(name):
    # Exclude alternate weight formats, optimizer states, visuals, and other arms.
    return name in (
        "model.safetensors",
        "pytorch_model.bin",
        "README.md",
        "LICENSE",
    ) or ("/" not in name and name.endswith((".json", ".model", ".txt")))


def freeze_manifest(out):
    path = out / "asset-manifest.json"
    if path.exists():
        value = json.loads(path.read_text())
        assert value["schema"] == "cross_model_geometry_v5_assets_1"
        return value
    manifest = {
        "schema": "cross_model_geometry_v5_assets_1",
        "proxy_policy": "urllib ProxyHandler({}); curl --disable --proxy '' --noproxy '*'; proxy environment removed",
        "hashing": "disabled; byte counts and load-time tensor coverage only",
        "models": {},
        "sources": {},
        "files": [],
    }
    for repo, selection in REPOSITORIES.items():
        metadata = direct_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        revision = metadata["sha"]
        directory = MODELS / repo.replace("/", "--")
        chosen = [
            r
            for r in metadata["siblings"]
            if (r["rfilename"] in selection if selection else allowed(r["rfilename"]))
        ]
        if selection:
            assert set(selection) <= {r["rfilename"] for r in chosen}
        # Public .bin and safetensors are alternatives; choose safetensors if available.
        if any(r["rfilename"] == "model.safetensors" for r in chosen):
            chosen = [r for r in chosen if r["rfilename"] != "pytorch_model.bin"]
        manifest["models"][repo] = {
            "revision": revision,
            "directory": str(directory),
            "license": metadata.get("cardData", {}).get("license"),
            "parameter_metadata": metadata.get("safetensors", {}),
        }
        for entry in chosen:
            name = entry["rfilename"]
            manifest["files"].append(
                {
                    "repo": repo,
                    "revision": revision,
                    "name": name,
                    "bytes": entry["size"],
                    "path": str(directory / name),
                    "url": f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}",
                }
            )
        emit("asset_repository_frozen", repo=repo, files=len(chosen))
    for repo in SOURCES:
        env = {
            k: v
            for k, v in os.environ.items()
            if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }
        env.update(NO_PROXY="*", no_proxy="*", GIT_TERMINAL_PROMPT="0")
        remote = subprocess.check_output(
            [
                "git",
                "-c",
                "http.proxy=",
                "-c",
                "https.proxy=",
                "-c",
                "credential.helper=",
                "ls-remote",
                f"https://github.com/{repo}.git",
                "refs/heads/main",
            ],
            env=env,
            text=True,
            timeout=90,
        )
        revision, reference = remote.strip().split()
        assert reference == "refs/heads/main" and len(revision) == 40
        directory = MODELS / "_source_snapshots" / repo.replace("/", "--") / revision
        manifest["sources"][repo] = {"revision": revision, "directory": str(directory)}
        manifest["files"].append(
            {
                "repo": repo,
                "revision": revision,
                "name": "source.tar.gz",
                "bytes": None,
                "path": str(directory.parent / f"{revision}.tar.gz"),
                "url": f"https://codeload.github.com/{repo}/tar.gz/{revision}",
                "extract_to": str(directory),
            }
        )
    manifest["model_bytes"] = sum(row["bytes"] or 0 for row in manifest["files"])
    write_json(path, manifest)
    return manifest


def fetch(entry):
    path = Path(entry["path"])
    expected = entry["bytes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if path.exists():
        if expected is not None and path.stat().st_size != expected:
            raise RuntimeError(
                f"Refusing to overwrite existing incorrect-sized asset: {path}"
            )
    else:
        partial = path.with_suffix(path.suffix + ".part")
        env = {
            k: v
            for k, v in os.environ.items()
            if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }
        env.update(NO_PROXY="*", no_proxy="*")
        command = [
            "curl",
            "--disable",
            "--proxy",
            "",
            "--noproxy",
            "*",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--max-time",
            "1800",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            entry["url"],
        ]
        emit(
            "download_start",
            repo=entry["repo"],
            name=entry["name"],
            expected_bytes=expected,
        )
        subprocess.run(command, env=env, check=True)
        if expected is not None and partial.stat().st_size != expected:
            raise RuntimeError(f"Truncated download: {path}")
        partial.replace(path)
    if entry.get("extract_to"):
        directory = Path(entry["extract_to"])
        marker = directory / ".geometry-v5-extracted.json"
        if not marker.exists():
            directory.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path) as archive:
                members = archive.getmembers()
                prefixes = {m.name.split("/", 1)[0] for m in members}
                assert len(prefixes) == 1
                selected = []
                for member in members:
                    if "/" not in member.name:
                        continue
                    member.name = member.name.split("/", 1)[1]
                    if member.name:
                        selected.append(member)
                archive.extractall(directory, members=selected, filter="data")
            write_json(
                marker,
                {
                    "repo": entry["repo"],
                    "revision": entry["revision"],
                    "archive_bytes": path.stat().st_size,
                },
            )
    result = {
        "repo": entry["repo"],
        "name": entry["name"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "seconds": time.monotonic() - started,
        "proxy_disabled": True,
    }
    emit("download_complete", **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = freeze_manifest(args.output_dir)
    completed, failures = [], []
    # The model-specific contract is frozen before any weight bytes are requested.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, entry): entry for entry in manifest["files"]}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:  # noqa: BLE001 - persist every worker failure, then fail the whole run
                failures.append(
                    {"repo": entry["repo"], "name": entry["name"], "error": str(error)}
                )
                emit("download_failed", **failures[-1])
            write_json(
                args.output_dir / "asset-download-status.json",
                {
                    "complete": len(completed) == len(manifest["files"])
                    and not failures,
                    "pid": os.getpid(),
                    "completed": completed,
                    "failures": failures,
                    "expected_files": len(manifest["files"]),
                },
            )
    if failures:
        raise RuntimeError(f"{len(failures)} assets failed; resume this same manifest")
    emit(
        "all_assets_complete",
        files=len(completed),
        bytes=sum(r["bytes"] for r in completed),
    )


if __name__ == "__main__":
    main()
