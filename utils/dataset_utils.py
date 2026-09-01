def get_dataloaders(config, tokenizer):
    dataset_class = config.dataset.class_name
    if dataset_class == "UnifiedMixedDataset":
        from .combined_dataloaders import build_unified_mixed_dataloaders

        return build_unified_mixed_dataloaders(
            config=config,
            tokenizer=tokenizer,
        )
    if dataset_class != "ImageNetFlowCacheDataset":
        raise ValueError(
            "Selfless-Flow supports only ImageNetFlowCacheDataset or "
            "UnifiedMixedDataset; "
            f"got dataset.class_name={dataset_class!r}."
        )

    from .imagenet_flow_dataloaders import (
        build_imagenet_flow_cache_dataloaders,
    )

    return build_imagenet_flow_cache_dataloaders(
        config=config,
        tokenizer=tokenizer,
    )
