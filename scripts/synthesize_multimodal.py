"""
Synthesize multimodal training data using SII API (text-only generation).

Level 1: Caption enhancement — diverse captions, VQA pairs, instructions
Level 2: Interleaved documents — educational articles referencing images

Usage:
    uv run python scripts/synthesize_multimodal.py --max_images 10 --output_dir /tmp/test_synth/
    uv run python scripts/synthesize_multimodal.py --max_images 30000  # Full Phase 1
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm


LEVEL1_CAPTION_PROMPT = """Here are 5 captions describing the same image:
{joined_captions}

Generate 3 new one-sentence captions for this image, each with a different style:
(1) factual and precise
(2) conversational and casual
(3) detailed and descriptive

Output as JSON: {{"captions": ["caption1", "caption2", "caption3"]}}"""

LEVEL1_VQA_PROMPT = """Here are 5 captions describing the same image:
{joined_captions}

Based on these captions, generate 3 question-answer pairs about the image.
Include questions about: objects present, counting, colors, positions, actions depicted.
Output as JSON: {{"qa_pairs": [{{"question": "...", "answer": "..."}}, ...]}}"""

LEVEL1_INSTRUCTION_PROMPT = """Here are 5 captions describing the same image:
{joined_captions}

Generate a user instruction about this image and a helpful assistant response.
The instruction could be: describe the image, answer a question, or analyze something.
Output as JSON: {{"instruction": "...", "response": "..."}}"""

LEVEL2_INTERLEAVED_PROMPT = """Here are 5 captions describing the same image:
{joined_captions}

Write a 6-8 sentence educational article that naturally references this image
mid-article (not at the beginning or end). Use [IMAGE] as a placeholder where
the image appears. The article should be informative and engaging.

Output as JSON with a "segments" list where each item has "type" ("text" or "image")
and "content" (the text, or for image just use "[IMAGE]"):
{{"segments": [{{"type": "text", "content": "..."}}, {{"type": "image", "content": "[IMAGE]"}}, {{"type": "text", "content": "..."}}]}}"""


def load_captions(captions_path: str) -> List[Dict]:
    """Load COCO captions from JSONL file."""
    entries = []
    with open(captions_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def build_prompt_messages(prompt_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant for multimodal data synthesis. Output valid JSON only."},
        {"role": "user", "content": prompt_text},
    ]


async def synthesize_image(
    client,
    entry: Dict,
    do_level1: bool = True,
    do_level2: bool = True,
) -> List[Dict]:
    """Synthesize training samples for one image."""
    joined = "\n".join(f"{i+1}. {c}" for i, c in enumerate(entry["captions"]))
    img_id = entry["img_id"]
    samples = []

    if do_level1:
        # Caption variants -> text_to_image samples
        cap_prompt = build_prompt_messages(
            LEVEL1_CAPTION_PROMPT.format(joined_captions=joined)
        )
        result = await client.chat_json(cap_prompt)
        if result and "captions" in result:
            for caption in result["captions"]:
                samples.append({
                    "task_mode": "text_to_image",
                    "img_id": img_id,
                    "text": caption,
                })

        # VQA -> image_to_text samples
        vqa_prompt = build_prompt_messages(
            LEVEL1_VQA_PROMPT.format(joined_captions=joined)
        )
        result = await client.chat_json(vqa_prompt)
        if result and "qa_pairs" in result:
            for qa in result["qa_pairs"]:
                samples.append({
                    "task_mode": "image_to_text",
                    "img_id": img_id,
                    "text": f"Question: {qa['question']}\nAnswer: {qa['answer']}",
                })

        # Instruction -> image_to_text samples
        inst_prompt = build_prompt_messages(
            LEVEL1_INSTRUCTION_PROMPT.format(joined_captions=joined)
        )
        result = await client.chat_json(inst_prompt)
        if result and "instruction" in result and "response" in result:
            samples.append({
                "task_mode": "image_to_text",
                "img_id": img_id,
                "text": f"User: {result['instruction']}\nAssistant: {result['response']}",
            })

    if do_level2:
        # Interleaved document
        inter_prompt = build_prompt_messages(
            LEVEL2_INTERLEAVED_PROMPT.format(joined_captions=joined)
        )
        result = await client.chat_json(inter_prompt)
        if result and "segments" in result:
            # Replace [IMAGE] placeholders with image references
            segments = []
            img_idx = 0
            for seg in result["segments"]:
                if seg.get("type") == "image" or "[IMAGE]" in seg.get("content", ""):
                    segments.append({"type": "image", "img_idx": img_idx})
                    img_idx += 1
                else:
                    segments.append({"type": "text", "content": seg.get("content", "")})
            if segments:
                samples.append({
                    "task_mode": "interleaved",
                    "img_ids": [img_id],
                    "segments": segments,
                })

    return samples


async def main_async(args):
    from scripts.sii_client import SIIClient

    # Load data
    captions_path = Path(args.captions_jsonl)
    if not captions_path.exists():
        print(f"Captions file not found: {captions_path}")
        print("Run scripts/download_coco.py first to download COCO data.")
        return

    entries = load_captions(captions_path)
    if len(entries) > args.max_images:
        random.shuffle(entries)
        entries = entries[:args.max_images]

    print(f"Synthesizing data for {len(entries)} images...")
    print(f"  Level 1: {args.level1} (captions + VQA + instructions)")
    print(f"  Level 2: {args.level2} (interleaved documents)")

    client = SIIClient(max_concurrent=args.concurrency)

    # For small test batches, do sequentially for debugging
    if args.max_images <= 5:
        all_samples = []
        for entry in tqdm(entries, desc="Synthesizing"):
            samples = await synthesize_image(
                client, entry,
                do_level1=args.level1,
                do_level2=args.level2,
            )
            all_samples.extend(samples)
            for s in samples:
                print(f"  -> {s['task_mode']}: {str(s.get('text', s.get('segments', '')))[:100]}...")
    else:
        # Concurrent batch processing
        sem = asyncio.Semaphore(args.concurrency)

        async def process_one(entry):
            async with sem:
                return await synthesize_image(
                    client, entry,
                    do_level1=args.level1,
                    do_level2=args.level2,
                )

        tasks = [process_one(e) for e in entries]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Synthesizing"):
            samples = await coro
            results.extend(samples)

        all_samples = results

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"

    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Stats
    mode_counts = {}
    for s in all_samples:
        mode_counts[s["task_mode"]] = mode_counts.get(s["task_mode"], 0) + 1
    print(f"\nDone! {len(all_samples)} samples saved to {output_path}")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_jsonl", default="public/datasets/coco/captions.jsonl")
    parser.add_argument("--output_dir", default="public/datasets/coco/synthetic")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--level1", action="store_true", default=True)
    parser.add_argument("--level2", action="store_true", default=True)
    parser.add_argument("--no_level1", dest="level1", action="store_false")
    parser.add_argument("--no_level2", dest="level2", action="store_false")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
