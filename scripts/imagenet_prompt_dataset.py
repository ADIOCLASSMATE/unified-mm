"""
Build an OmniCorpus-compatible ImageNet prompt-to-image dataset.

Output docs use the same schema consumed by scripts/omnicorpus_build_arrow.py:
    {"doc_id": ..., "img_ids": [...], "segments": [{"type": "text"}, {"type": "image"}]}

Images are linked into a numeric image directory so modality encoders can load
them without any special ImageNet path logic.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


PROMPT_TEMPLATES = [
    "Generate a realistic ImageNet-style photograph of {article} {class_name}. The subject should be clearly recognizable and occupy the main visual focus of the image.",
    "Create a natural image that contains {article} {class_name}. Make the object easy to identify, with plausible lighting, background, and scale.",
    "Please draw {article} {class_name} in a scene that looks like a real photograph from a visual recognition dataset.",
    "I need an image of {article} {class_name}. The generated picture should make the class identity unambiguous.",
    "Produce a high-quality visual example of {article} {class_name}, centered enough that a classifier could recognize it.",
    "Render {article} {class_name} as the main subject. Keep the composition simple, realistic, and useful for testing image generation.",
    "This is a request for a picture of {article} {class_name}. Show the object with enough detail to distinguish it from similar categories.",
    "Now create an image whose primary category is {class_name}. The image should look like a normal ImageNet training sample.",
    "Imagine a dataset example labeled {class_name}. Generate the corresponding image with the labeled object visible.",
    "Given the class label {class_name}, produce a matching image where the visual evidence strongly supports that label.",
    "Create a scene containing {article} {class_name}. Avoid making the class too tiny, hidden, or ambiguous.",
    "Draw a recognizable {class_name} in a realistic setting. The image should be suitable for evaluating text-conditioned image generation.",
    "Generate an image for the category {class_name}. The result should emphasize the object's shape, texture, and typical appearance.",
    "Please synthesize a natural photograph-like image showing {article} {class_name} as the dominant subject.",
    "I want to see {article} {class_name} in the generated image. Make it visually clear and semantically aligned with the prompt.",
    "Create a clean image example of {article} {class_name}. The background may be simple, but the object should be identifiable.",
    "Render a plausible real-world scene where {article} {class_name} appears prominently.",
    "Generate a picture that could be captioned as: {article_cap} {class_name} is visible in the scene.",
    "Create an image matching the ImageNet class {synset}: {class_name}. The object should be the main thing a viewer notices.",
    "Make a visually grounded example of {article} {class_name}, with enough detail for both humans and models to recognize it.",
    "Please produce an image of {article} {class_name}. Use a natural composition instead of an abstract symbol or icon.",
    "Draw a realistic {class_name}. The generated content should correspond to the named ImageNet category.",
    "Generate {article} {class_name} in a simple scene. The object should not be cropped so heavily that its class becomes unclear.",
    "Create a sample image where the target object is {article} {class_name}. Keep the label-image alignment strong.",
    "I need to draw {article} {class_name} image for a class-conditional generation test.",
    "Now I have a prompt for {article} {class_name} image. Generate the corresponding visual sample.",
    "This image should depict {article} {class_name}. Make the category visually obvious.",
    "Please create a photographic example of {article} {class_name}, not just text or a diagram.",
    "Generate a realistic training image for the class {class_name}. The class should be represented by a visible object.",
    "Create an image where {article} {class_name} appears in a believable environment.",
    "Draw {article} {class_name} with a natural background and clear object boundaries.",
    "Produce an ImageNet-like image of {article} {class_name}; the object should be recognizable at a glance.",
    "Make an image that answers the instruction: show me {article} {class_name}.",
    "Generate a visual example of the class {class_name}, with the subject large enough to inspect.",
    "Create a realistic picture of {article} {class_name}. Preserve the typical visual traits of this category.",
    "I am testing whether the model can generate images from labels. The requested label is {class_name}.",
    "For this image generation example, the target category is {class_name}. Produce a matching natural image.",
    "Construct a scene that contains {article} {class_name}, with the target class clearly visible.",
    "Generate an image centered on {article} {class_name}. The output should resemble a real camera photograph.",
    "Please make {article} {class_name} the main subject of the image, with realistic colors and proportions.",
    "Create an image whose semantic content corresponds to {article} {class_name}.",
    "Draw a dataset-style example of {article} {class_name}. The picture should be useful for generation evaluation.",
    "Given only the class name {class_name}, create a plausible image that represents that category.",
    "Produce a visual sample for {class_name}. The subject should be specific enough to match the ImageNet label.",
    "Generate a photo-like image showing {article} {class_name}. It should not be a collage, logo, or text-only rendering.",
    "Create a single-image example of {article} {class_name}, suitable for multimodal pretraining verification.",
    "Please render {article} {class_name} in a way that makes the category easy to recover from the image.",
    "Make a natural image with {article} {class_name} as the key object. Use realistic lighting and perspective.",
    "Generate a class-conditioned image for {class_name}. The image should contain the named object rather than unrelated content.",
    "Create a visual scene for the label {class_name}. The label should be supported by the visible object.",
    "Draw {article} {class_name} in a normal environment, as if it were photographed for an object recognition benchmark.",
    "Produce an example image where the answer to 'what is the main object?' would be {class_name}.",
    "Generate a picture of {article} {class_name}; include enough context to make the object appear natural.",
    "Please create an image corresponding to this category description: {class_alias}.",
    "Render a realistic object from the category {class_name}. The image should be visually coherent and recognizable.",
    "Make a generation target where the prompt asks for {article} {class_name} and the image satisfies that request.",
    "Create a straightforward photograph-like depiction of {article} {class_name}.",
    "Generate {article} {class_name} with clear visual features, avoiding overly artistic abstraction.",
    "Please draw an image of {article} {class_name} that could plausibly appear in ImageNet.",
    "Create an image sample labeled {class_name}. The class should be the central visual concept.",
    "Generate a visual answer to the request: I need {article} {class_name} image.",
    "Make the main subject {article} {class_name}. The scene can be simple, but the category must be clear.",
    "Produce a realistic example of {article} {class_name} for an image generation benchmark.",
    "Create an image where a viewer would naturally describe the main object as {article} {class_name}.",
    "Generate a clean, high-signal image of {article} {class_name}, focusing on recognizability over decoration.",
    "Please synthesize a picture of {article} {class_name} using a natural visual style.",
    "Draw {article} {class_name} as a real object in the world, not as a written label.",
    "Create a plausible ImageNet validation-style image containing {article} {class_name}.",
    "Generate an image that demonstrates the concept {class_name} through visible pixels.",
    "Please produce a class-conditioned visual sample for {synset}, whose human-readable label is {class_name}.",
    "Make an image where the category {class_name} is visually dominant and not merely incidental.",
    "Generate a realistic scene with {article} {class_name}. The prompt-image pair should be useful for early image generation validation.",
    "Create an image for a unified multimodal model test: the text asks for {article} {class_name}, and the image should match.",
    "Please render the requested object category, {class_name}, with enough fidelity to evaluate generation quality.",
    "Draw {article} {class_name} in a typical pose, view, or context for that category.",
    "Generate a natural photo-like sample where {article} {class_name} is visible and identifiable.",
    "Create a generated image target for this instruction: show a clear example of {article} {class_name}.",
    "Produce a visual instance of {article} {class_name}. The image should be neither empty nor dominated by unrelated objects.",
    "Please create a realistic image that could be paired with the caption: this is {article} {class_name}.",
    "Generate a concrete visual depiction of {article} {class_name}, preserving the category's usual appearance.",
    "Make an image sample in which {class_name} is the intended label. The subject should be obvious to a human observer.",
    "Create a picture where the main object belongs to the ImageNet category {class_name}.",
    "Generate a scene suitable for training a text-to-image model: prompt asks for {article} {class_name}, image shows it.",
    "Please draw {article} {class_name} with realistic texture and shape cues.",
    "Produce a class-aligned image whose correct short description would include {class_name}.",
    "Create a simple but realistic image of {article} {class_name}, with minimal ambiguity.",
    "Generate a natural image that visually represents {class_alias}.",
    "Render {article} {class_name} as if captured by a camera in ordinary conditions.",
    "Please create an ImageNet-like photograph where {article} {class_name} is the target object.",
    "Generate a visual example for the prompt 'a photo of {article} {class_name}'.",
    "Draw {article} {class_name} in a believable setting. The class should remain the focus even if the background is detailed.",
    "Create a single-subject image of {article} {class_name} for testing conditional image generation.",
    "Produce a realistic picture that a classifier should recognize as {class_name}.",
    "Generate an image where the visual content is consistent with the label {class_name}.",
    "Please synthesize a clear and recognizable {class_name} in a natural scene.",
    "Create a prompt-following target image: the requested category is {class_name}.",
    "Draw {article} {class_name} with a composition that keeps the object visible and class-discriminative.",
    "Generate a normal photograph-like image of {article} {class_name}, with no need for extra text in the image.",
    "Make a realistic sample for the ImageNet synset {synset}. The visible subject should be {article} {class_name}.",
    "Create a visual instance of the class {class_name}. The image should look like data rather than a poster.",
    "Please generate a well-framed image of {article} {class_name}.",
    "Produce an image that could serve as a positive example for the category {class_name}.",
    "Draw a scene where {article} {class_name} appears clearly enough to support the class label.",
    "Generate an image for early multimodal generation verification: requested object is {article} {class_name}.",
    "Create a realistic depiction of {article} {class_name}, emphasizing the object instead of the background.",
    "Please make an image that corresponds to this instruction: I need a recognizable {class_name}.",
    "Render a class-conditioned sample of {article} {class_name}; the result should be visually grounded.",
    "Generate a picture that would be appropriate for the label {class_name} in an ImageNet-style dataset.",
    "Create a visual sample where the main semantic content is {article} {class_name}.",
    "Please draw a realistic image containing {article} {class_name}, with the object not hidden by clutter.",
    "Produce a plausible natural image of {article} {class_name} for a generative modeling experiment.",
    "Generate {article} {class_name} in an ordinary scene. Make the object clear enough for label supervision.",
    "Create a matching image for the text prompt: a clear photo of {article} {class_name}.",
    "Please synthesize an image whose target concept is {class_name}.",
    "Draw {article} {class_name} as the central subject of a realistic image.",
    "Generate a class prompt image pair where the text condition names {article} {class_name}.",
    "Create a high-signal example of {article} {class_name}; the category should be recoverable from pixels alone.",
    "Please render {article} {class_name} with typical colors, geometry, and context for that object.",
    "Produce a realistic image sample for class-conditioned generation, using {class_name} as the class label.",
    "Generate a visual instance matching the label {class_name}, avoiding unrelated dominant subjects.",
    "Create an image that can be used to verify whether a model understands the prompt {class_name}.",
    "Please make the generated picture show {article} {class_name} clearly and naturally.",
    "Draw {article} {class_name} in a way that would make sense as an ImageNet classification example.",
    "Generate a photo-like depiction of {article} {class_name}; the prompt and image should be tightly aligned.",
    "Create a natural image for the class {class_name}, suitable for a small-scale generation probe.",
    "Please produce a picture where the main object is {article} {class_name}.",
    "Render a realistic example of {article} {class_name} with recognizable visual details.",
    "Generate an image corresponding to the label text {class_alias}.",
    "Create a straightforward visual target: {article_cap} {class_name} should appear in the image.",
    "Please draw an ImageNet-style sample whose ground-truth category is {class_name}.",
    "Produce a scene where {article} {class_name} is present, prominent, and visually identifiable.",
    "Generate a realistic image where the intended answer to the prompt is {class_name}.",
    "Create a class-conditional image for {class_name}; keep the output natural and object-focused.",
    "Please synthesize a picture of {article} {class_name} for testing whether image tokens can be generated from text.",
    "Draw {article} {class_name} as a visible object in a coherent scene.",
    "Generate a plausible dataset image whose label would be {class_name}.",
    "Create an image that follows this request: show me a clear, realistic {class_name}.",
    "Please make a natural visual example of {article} {class_name}, with the object easy to inspect.",
    "Render the class {class_name} as an image. The generated sample should be useful for evaluating class-conditioned generation.",
    "Generate a photo-like image of {article} {class_name}, not an explanation or written description.",
    "Create a visual sample for {class_name}. The target object should be the central reason for the image.",
    "Please draw {article} {class_name} with enough resolution and detail to make the label meaningful.",
]


def article_for(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def load_synset_mapping(path: Path) -> Dict[str, Dict[str, object]]:
    mapping = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            synset, names = line.split(" ", 1)
            aliases = [name.strip() for name in names.split(",") if name.strip()]
            mapping[synset] = {
                "primary": aliases[0],
                "aliases": aliases,
                "alias_text": names,
            }
    return mapping


def load_val_synsets(path: Path) -> Dict[str, str]:
    image_to_synset = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"]
            prediction = row["PredictionString"].split()
            if prediction:
                image_to_synset[image_id] = prediction[0]
    return image_to_synset


def iter_train_images(data_root: Path) -> Iterable[Tuple[Path, str]]:
    train_root = data_root / "train"
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                yield path, class_dir.name


def select_class_dirs(
    train_root: Path,
    max_classes: int,
    class_selection: str,
    seed: int,
) -> List[Path]:
    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if max_classes <= 0 or max_classes >= len(class_dirs):
        return class_dirs
    if class_selection == "first":
        return class_dirs[:max_classes]
    if class_selection == "random":
        selected = random.Random(seed).sample(class_dirs, max_classes)
        return sorted(selected)
    if class_selection == "spread":
        if max_classes == 1:
            return [class_dirs[len(class_dirs) // 2]]
        last = len(class_dirs) - 1
        indices = [round(i * last / (max_classes - 1)) for i in range(max_classes)]
        return [class_dirs[index] for index in indices]
    raise ValueError(f"unknown class selection mode: {class_selection}")


def iter_train_images_balanced(
    data_root: Path,
    max_images_per_class: int,
    seed: int,
    shuffle_within_class: bool,
    max_classes: int,
    class_selection: str,
) -> Iterable[Tuple[Path, str]]:
    train_root = data_root / "train"
    rng = random.Random(seed)
    for class_dir in select_class_dirs(train_root, max_classes, class_selection, seed):
        paths = [path for path in class_dir.iterdir() if path.is_file()]
        if shuffle_within_class:
            rng.shuffle(paths)
        else:
            paths.sort()
        selected_paths = paths if max_images_per_class <= 0 else paths[:max_images_per_class]
        for path in selected_paths:
            yield path, class_dir.name


def iter_val_images(data_root: Path, val_synsets: Dict[str, str]) -> Iterable[Tuple[Path, str]]:
    val_root = data_root / "val"
    for path in sorted(val_root.iterdir()):
        if not path.is_file():
            continue
        synset = val_synsets.get(path.stem)
        if synset is not None:
            yield path, synset


def choose_class_name(info: Dict[str, object], mode: str, rng: random.Random) -> str:
    aliases = info["aliases"]
    if mode == "primary":
        return str(info["primary"])
    if mode == "random_alias":
        return str(rng.choice(aliases))
    if mode == "full":
        return str(info["alias_text"])
    raise ValueError(f"unknown class name mode: {mode}")


def render_prompt(template: str, synset: str, info: Dict[str, object], mode: str, rng: random.Random) -> str:
    class_name = choose_class_name(info, mode, rng)
    values = {
        "synset": synset,
        "class_name": class_name,
        "class_alias": str(info["alias_text"]),
        "article": article_for(class_name),
        "article_cap": article_for(class_name).capitalize(),
    }
    return template.format(**values).strip() + "\n"


def link_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    elif mode == "none":
        return
    else:
        raise ValueError(f"unknown image link mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", default="/inspire/dataset/imagenet/v1")
    parser.add_argument("--data_root", default=None,
                        help="Defaults to IMAGENET_ROOT/ILSVRC/Data/CLS-LOC.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output_jsonl", default="public/datasets/imagenet_prompt/docs/train.jsonl")
    parser.add_argument("--image_output_dir", default="public/datasets/imagenet_prompt/images")
    parser.add_argument("--image_link_mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--prompts_per_image", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--max_images_per_class", type=int, default=-1,
                        help="For train split, take at most this many images per class.")
    parser.add_argument("--max_classes", type=int, default=-1,
                        help="For train split, take at most this many classes.")
    parser.add_argument("--class_selection", choices=["spread", "random", "first"], default="spread",
                        help="How to choose classes when --max_classes is set.")
    parser.add_argument("--shuffle_within_class", action="store_true",
                        help="Shuffle images within each class before applying --max_images_per_class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_name_mode", choices=["primary", "random_alias", "full"], default="primary")
    parser.add_argument("--start_img_id", type=int, default=None,
                        help="Default: 1 for train, 200000000 for val.")
    parser.add_argument("--source_name", default="ILSVRC2012-CLS-LOC")
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompts_per_image < 1:
        raise ValueError("--prompts_per_image must be >= 1")

    rng = random.Random(args.seed)
    imagenet_root = Path(args.imagenet_root)
    data_root = Path(args.data_root) if args.data_root else imagenet_root / "ILSVRC" / "Data" / "CLS-LOC"
    synset_mapping = load_synset_mapping(imagenet_root / "LOC_synset_mapping.txt")

    if args.split == "train":
        if args.max_images_per_class > 0 or args.max_classes > 0 or args.shuffle_within_class:
            sample_iter = iter_train_images_balanced(
                data_root,
                max_images_per_class=args.max_images_per_class,
                seed=args.seed,
                shuffle_within_class=args.shuffle_within_class,
                max_classes=args.max_classes,
                class_selection=args.class_selection,
            )
        else:
            sample_iter = iter_train_images(data_root)
        default_start_img_id = 1
    else:
        val_synsets = load_val_synsets(imagenet_root / "LOC_val_solution.csv")
        sample_iter = iter_val_images(data_root, val_synsets)
        default_start_img_id = 200_000_000

    if args.shuffle:
        samples = list(sample_iter)
        rng.shuffle(samples)
        sample_iter = iter(samples)

    start_img_id = args.start_img_id if args.start_img_id is not None else default_start_img_id
    output_jsonl = Path(args.output_jsonl)
    image_output_dir = Path(args.image_output_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    records = 0
    linked_images = 0
    skipped = 0
    used_images = 0
    total_expected = None
    if args.max_images > 0:
        total_expected = args.max_images
    elif args.split == "train" and args.max_classes > 0 and args.max_images_per_class > 0:
        total_expected = args.max_classes * args.max_images_per_class

    with output_jsonl.open("w") as out:
        progress = tqdm(
            sample_iter,
            total=total_expected,
            desc="Building ImageNet prompts",
            unit="img",
            mininterval=1.0,
        )
        for source_index, (image_path, synset) in enumerate(progress):
            if args.max_images > 0 and used_images >= args.max_images:
                break
            info = synset_mapping.get(synset)
            if info is None:
                skipped += 1
                continue
            img_id = start_img_id + used_images
            linked_path = image_output_dir / f"{img_id:012d}.jpg"
            before_exists = linked_path.exists() or linked_path.is_symlink()
            link_image(image_path.resolve(), linked_path, args.image_link_mode)
            if not before_exists and (linked_path.exists() or linked_path.is_symlink()):
                linked_images += 1

            template_indices = list(range(len(PROMPT_TEMPLATES)))
            rng.shuffle(template_indices)
            for prompt_i in range(args.prompts_per_image):
                template = PROMPT_TEMPLATES[template_indices[prompt_i % len(template_indices)]]
                prompt = render_prompt(template, synset, info, args.class_name_mode, rng)
                record = {
                    "doc_id": f"imagenet-{args.split}-{img_id:012d}-p{prompt_i:02d}",
                    "source": args.source_name,
                    "img_ids": [img_id],
                    "segments": [
                        {"type": "text", "content": prompt},
                        {"type": "image", "img_idx": 0},
                    ],
                    "metadata": {
                        "split": args.split,
                        "synset": synset,
                        "class_name": str(info["primary"]),
                        "class_alias": str(info["alias_text"]),
                        "image_path": str(image_path),
                        "source_index": source_index,
                        "prompt_template_index": template_indices[prompt_i % len(template_indices)],
                    },
                }
                out.write(json.dumps(record, ensure_ascii=True) + "\n")
                records += 1
            used_images += 1

    print(
        f"Done: wrote {records} records for {used_images} images to {output_jsonl}; "
        f"linked {linked_images} new images in {image_output_dir}; skipped={skipped}"
    )


if __name__ == "__main__":
    main()
