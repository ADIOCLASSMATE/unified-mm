import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_flow_latent import ImageLatentFlowDataset
from utils.utils import get_config


def main():
    config = get_config()
    params = config.dataset.params
    cache_path = params.get("cache_path", None)
    if not cache_path:
        raise ValueError("dataset.params.cache_path is required")

    ImageLatentFlowDataset(
        docs_jsonl=params.get("docs_jsonl", None),
        latent_dir=params.latent_dir,
        image_tokens_per_img=params.get("image_tokens_per_img", config.model.image_tokens_per_img),
        image_latent_dim=params.get("image_latent_dim", config.model.image_latent_dim),
        latent_key=params.get("latent_key", "latent"),
        max_samples=params.get("max_samples", -1),
        deduplicate_image_ids=params.get("deduplicate_image_ids", True),
        cache_path=cache_path,
        cache_mode="rebuild" if params.get("cache_rebuild", False) else "auto",
        cache_dtype=params.get("cache_dtype", "float16"),
        cache_mmap=params.get("cache_mmap", True),
        return_dtype=params.get("return_dtype", "float16"),
    )
    print(f"Flow latent cache is ready: {cache_path}")


if __name__ == "__main__":
    main()
