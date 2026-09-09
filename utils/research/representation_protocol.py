"""Original frozen-representation sample protocol and report I/O."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FLICKR = ROOT / "public/benchmarks/flickr30k_karpathy_retrieval_v1"
IMAGENET = ROOT / "public/datasets/imagenet_full"
TEMPLATES = (
    "a photo of a {name}.",
    "an image of a {name}.",
    "a picture of a {name}.",
    "this is a {name}.",
    "the subject is a {name}.",
    "a close-up of a {name}.",
    "a photograph showing a {name}.",
    "there is a {name} in the picture.",
)
PREFIX = "Describe this image in one detailed caption:"
SUFFIX = "\nThe main subject is"
DATASETS = ("flickr_images", "flickr_texts", "imagenet_images", "imagenet_texts")


def emit(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
