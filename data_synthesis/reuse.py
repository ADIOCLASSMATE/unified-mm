"""Conservative reuse and explicit annotation rendering before model inference."""
import json
import re
from jsonschema import ValidationError

from data_synthesis.contract import parse_pair, text_pair
from data_synthesis.io import dumps, sha
from data_synthesis.integrity import hashing_enabled, same_view_binding, view_reference


def full_frame(view):
    width, height = view["original_size"]
    return list(view["crop"]) == [0, 0, width, height]


def normalize(text):
    # Formatting only: do not truncate facts, resolve pronouns or invent details.
    return text.replace("\r\n", "\n").strip()


def aligned(candidate, item, policy, *, compute_hashes=True):
    row, view = item["row"], item["view"]
    if candidate.get("image_identity") not in ({item["identity"]} | set(row.get("identity_aliases", []))):
        return False, "caption is not joined to this original image identity"
    if not compute_hashes and candidate.get("kind") == "accepted_pair":
        if candidate.get("view_id") == view_reference(view):
            return True, "previously_accepted_image_reference"
        return False, "accepted caption is not bound to this image reference"
    if compute_hashes and candidate.get("view_sha256"):
        if candidate["view_sha256"] != view["view_sha256"]:
            return False, "caption belongs to a different frozen view"
        if candidate.get("kind") == "accepted_pair":
            return True, "exact_previously_accepted_view"
    if compute_hashes and candidate.get("source_sha256") and candidate["source_sha256"] != view["source_sha256"]:
        return False, "caption original-image hash differs"
    if not full_frame(view):
        return False, "crop requires text realignment"
    if {"ocr", "text", "chart", "document"} & set(row.get("capabilities", [])):
        readable = (row.get("readability_view_sha256") == view["view_sha256"] if compute_hashes
                    else row.get("readability_view_id") == view_reference(view))
        if policy["require_readability_for_ocr"] and not readable:
            return False, "final-view text readability has not been checked"
    kind = candidate.get("kind")
    allowed = ((kind == "human_caption" and policy["allow_full_frame_human_captions"])
               or (kind == "curated_caption" and policy["allow_curated_captions"]))
    if not allowed:
        return False, "generation prompt or unqualified caption needs image correction"
    return True, "source_caption_full_frame_geometry_and_declared_region_checks"


def choose_reuse(item, tokenizer, config):
    compute_hashes = hashing_enabled(config)
    issues, candidates = [], item["row"].get("caption_candidates", [])
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("author") or not candidate.get("provenance"):
            issues.append("caption lacks original author or provenance")
            continue
        okay, reason = aligned(candidate, item, config["reuse"], compute_hashes=compute_hashes)
        if not okay:
            issues.append(reason)
            continue
        original = {task: candidate.get(task, candidate.get("text", "")) for task in ("i2t", "t2i")}
        if not all(isinstance(t, str) for t in original.values()):
            issues.append("invalid source text type")
            continue
        texts = {task: normalize(value) for task, value in original.items()}
        if any(re.match(r"(?i)^(?:draw|create|generate|imagine)\s+(?:an? |the )", t) for t in texts.values()):
            issues.append("instructional generation prompt is not a self-contained caption")
            continue
        if any(any(fact.casefold() not in text.casefold() for fact in item["row"].get("required_fact_text", []))
               for text in texts.values()):
            issues.append("required grounded fact is missing from source caption")
            continue
        pair = text_pair(item["key"], texts["i2t"], capabilities=item["row"].get("capabilities", []),
                         observations=candidate.get("observations"))
        pair["t2i"] = texts["t2i"]
        try:
            parse_pair({"status": "completed", "output_text": dumps(pair)}, item["key"], tokenizer)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            issues.append(f"source text validation: {type(exc).__name__}")
            continue
        changed = texts != original
        evidence = {"route": "normalize" if changed else "reuse", "alignment": reason,
                    "source_candidate": candidate,
                    "source_candidate_sha256": sha(dumps(candidate).encode()) if compute_hashes else None,
                    "original_text_sha256": {k: sha(v.encode()) if compute_hashes else None for k, v in original.items()},
                    "generator_models": {k: (candidate.get("generator_models") or {}).get(k, candidate["author"]) for k in texts},
                    "transforms": ["normalize_line_endings_and_outer_whitespace"] if changed else [],
                    "semantic_accuracy_independently_verified": False}
        return pair, evidence, issues
    rendered = render_verified_facts(item, compute_hashes=compute_hashes)
    if rendered:
        pair, used = rendered
        try:
            parse_pair({"status": "completed", "output_text": dumps(pair)}, item["key"], tokenizer)
        except (ValueError, TypeError, KeyError, ValidationError):
            issues.append("annotation rendering exceeds text contract")
        else:
            return pair, {"route": "annotation", "verified_facts": used,
                          "generator_models": {"i2t": "verified_annotations", "t2i": "verified_annotations"},
                          "renderer_version": "explicit-positive-facts-v1"}, issues
    return None, None, list(dict.fromkeys(issues)) or ["no reusable caption or verified final-view facts"]


def render_verified_facts(item, *, compute_hashes=True):
    observations = {"counts": [], "relations": [], "visible_text": []}
    sentences, used = [], []
    for fact in item["row"].get("verified_facts", []):
        if (fact.get("verified") is not True or not same_view_binding(fact, item["view"], compute_hashes)
                or not fact.get("provenance")):
            continue
        kind = fact.get("type")
        if kind == "count":
            count, entity = fact.get("count"), fact.get("entity")
            if type(count) is not int or count < 0 or not isinstance(entity, str) or not entity.strip():
                continue
            if fact.get("fully_visible") is not True or fact.get("exhaustive_for_referent") is not True:
                continue
            if count == 0:
                continue  # Negative supervision is outside this caption renderer.
            sentences.append(f"Visible {entity}: {count}.")
            observations["counts"].append({"entity": entity, "count": count})
        elif kind == "relation" and isinstance(fact.get("text"), str) and fact["text"].strip():
            sentences.append(fact["text"].strip())
            observations["relations"].append(fact["text"].strip())
        elif kind == "text" and fact.get("readable") is True and fact.get("carrier") and fact.get("text"):
            sentences.append(f"The {fact['carrier']} displays {json.dumps(fact['text'], ensure_ascii=False)}.")
            observations["visible_text"].append(fact["text"])
        else:
            continue
        used.append(fact)
    if not sentences:
        return None
    return text_pair(item["key"], " ".join(sentences), observations=observations,
                     capabilities=item["row"].get("capabilities", [])), used
