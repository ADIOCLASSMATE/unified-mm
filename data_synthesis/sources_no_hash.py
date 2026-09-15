"""Counted image admission using original IDs/URLs and references, without hashing."""
import json
from pathlib import Path
import time
import uuid

from data_synthesis.clients import frozen_pixels
from data_synthesis.integrity import check_file_size, view_reference
from data_synthesis.io import atomic_json, dumps
from data_synthesis.state import State


def freeze_no_hash(root, config, *, prepared=(), releases=(), inbox=None,
                   exclude=None, exclude_prompts=None, pilot=False):
    from data_synthesis.sources import excluded_prompts, identity, prepared_rows, prompt_hash, release_rows

    if not pilot and (not exclude or not exclude_prompts):
        raise ValueError("production freeze requires image ID and test-prompt exclusions")
    inputs = [("prepared", Path(p).resolve()) for p in prepared]
    inputs += [("release", Path(p).resolve()) for p in releases]
    for directory in ([inbox] if isinstance(inbox, (str, Path)) else inbox) or []:
        directory = Path(directory).resolve()
        closed_path = directory / "closed.json"
        if not closed_path.is_file():
            raise ValueError("download/preparation inbox is not closed; bulk synthesis cannot start")
        closed = json.loads(closed_path.read_text())
        descriptors = sorted(p for p in directory.glob("*.json") if p.name != "closed.json")
        if len(descriptors) != closed["batches"]:
            raise ValueError("closed preparation inbox coverage differs")
        for descriptor in descriptors:
            value = json.loads(descriptor.read_text())
            source = Path(value["source_run"]).resolve()
            check_file_size(source / "state.sqlite3", value.get("state_bytes"))
            inputs.append(("prepared", source))
    if not inputs:
        raise ValueError("provide a closed prepared inbox, prepared run or accepted release")
    exclusions = set(Path(exclude).read_text().splitlines()) if exclude else set()
    prompts = excluded_prompts(exclude_prompts, compute_hashes=False)
    with State(root, config) as state:
        if state.meta("frozen"):
            raise ValueError("image pool is immutable after freezing; use another cohort/root")
        scope = {"image_ids": sorted(exclusions), "test_prompts": sorted(prompts)}
        previous = state.meta("exclusion_ids")
        if previous is not None and previous != scope:
            raise ValueError("evaluation exclusions changed during admission")
        state.set_meta("exclusion_ids", scope)
        state.set_meta("excluded_prompt_hashes", sorted(prompts))
        state.db.commit()
        for kind, source in inputs:
            admission = kind + ":" + str(source)
            if state.db.execute("SELECT 1 FROM admissions WHERE id=?", (admission,)).fetchone():
                continue
            expected = None
            if kind == "prepared" and not pilot:
                marker = json.loads((source / "batch.json").read_text())
                check_file_size(source / "state.sqlite3", marker.get("state_bytes"))
                expected = marker["records"]
            visited = 0
            for row, original_view in (prepared_rows(source) if kind == "prepared" else release_rows(source)):
                visited += 1
                ident = identity(row)
                aliases = set(row.get("identity_aliases", [])) | {ident, f"{row['source']}:{row['source_id']}"}
                if row.get("url"):
                    aliases.add("url:" + row["url"])
                aliases.add("image-reference:" + original_view["original_ref"])
                reason = None
                if row.get("split") != "train":
                    reason = "non_train"
                elif row["source"] == "imagenet" and not config.get("include_imagenet", False):
                    reason = "imagenet_deferred_not_part_of_non_imagenet_target"
                elif aliases & exclusions:
                    reason = "evaluation_image_id_exclusion"
                texts = [c.get(k, "") for c in row.get("caption_candidates", []) for k in ("text", "i2t", "t2i")]
                if any(isinstance(t, str) and t and prompt_hash(t, compute_hashes=False) in prompts for t in texts):
                    reason = "evaluation_test_prompt_overlap"
                view = {**original_view, "hashes_computed": False,
                        "source_sha256": None, "view_sha256": None,
                        "view_id": view_reference(original_view)}
                view.pop("perceptual_hashes", None)
                if not reason:
                    try:
                        frozen_pixels({"view": view}, compute_hashes=False)
                    except (ValueError, OSError) as exc:
                        reason = f"invalid_frozen_view:{type(exc).__name__}"
                if reason:
                    state.db.execute("INSERT OR REPLACE INTO exclusions VALUES (?,?,?)", (ident, reason, dumps(row)))
                    continue
                keys = {r[0] for alias in sorted(aliases) for r in
                        state.db.execute("SELECT item_key FROM image_aliases WHERE alias=?", (alias,))}
                if len(keys) > 1:
                    raise ValueError("image identity aliases refer to multiple admitted images")
                if keys:
                    key = keys.pop()
                    old = state.item(key)
                    merged = old["row"]
                    for field in ("caption_candidates", "verified_facts", "annotations", "capabilities", "identity_aliases"):
                        values = {dumps(v): v for v in merged.get(field, []) + row.get(field, [])}
                        merged[field] = list(values.values())
                    merged["identity_aliases"] = sorted(set(merged["identity_aliases"]) | aliases | {old["identity"]})
                    state.db.execute("UPDATE items SET row_json=? WHERE key=?", (dumps(merged), key))
                else:
                    key = uuid.uuid4().hex
                    row["identity_aliases"] = sorted(aliases)
                    state.db.execute("INSERT INTO items(key,identity,source_sha256,view_sha256,row_json,view_json,status,created_at) "
                                     "VALUES (?,?,NULL,NULL,?,?,'pending',?)", (key, ident, dumps(row), dumps(view), time.time()))
                state.db.executemany("INSERT OR IGNORE INTO image_aliases VALUES (?,?)", [(a, key) for a in sorted(aliases)])
                if visited % 4096 == 0:
                    state.db.commit()
            if expected is not None and visited != expected:
                raise ValueError("prepared batch row count changed")
            state.db.execute("INSERT INTO admissions VALUES (?,?,NULL,?)", (admission, str(source), visited))
            state.db.commit()
        total = state.db.execute("SELECT count(*) FROM items").fetchone()[0]
        non_imagenet = state.db.execute("SELECT count(*) FROM items WHERE json_extract(row_json,'$.source') != 'imagenet'").fetchone()[0]
        frozen = {"state": "frozen" if pilot or non_imagenet >= config["minimum_images"] else "needs_more_images",
                  "images": total, "non_imagenet_images": non_imagenet, "minimum_images": config["minimum_images"],
                  "pilot": pilot, "compute_hashes": False, "content_deduplication": False,
                  "identity_deduplication": "source_id_url_reference", "benchmark_perceptual_exclusion": False,
                  "excluded": state.db.execute("SELECT count(*) FROM exclusions").fetchone()[0],
                  "closed_at": time.time()}
        if frozen["state"] == "frozen":
            state.set_meta("frozen", frozen)
            state.db.commit()
        atomic_json(Path(root) / "freeze.json", frozen)
        return frozen
