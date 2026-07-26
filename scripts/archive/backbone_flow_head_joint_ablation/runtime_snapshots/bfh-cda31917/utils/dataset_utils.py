def get_dataloaders(config, tokenizer):
    dataset_class = config.dataset.class_name
    if dataset_class == "CombinedImageNetTextFlowDataset":
        from .dataset_combined_flow import build_combined_flow_dataloaders

        train_dataloader, val_dataloader = build_combined_flow_dataloaders(
            config=config,
            tokenizer=tokenizer,
        )
    elif dataset_class == "TextArrowDataset":
        from .dataset_combined_flow import build_text_arrow_dataloaders

        train_dataloader, val_dataloader = build_text_arrow_dataloaders(
            config=config,
            tokenizer=tokenizer,
        )
    elif dataset_class == "ImageNetFlowCacheDataset":
        from .dataset_imagenet_flow_cache import build_imagenet_flow_cache_dataloaders

        train_dataloader, val_dataloader = build_imagenet_flow_cache_dataloaders(
            config=config,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class}")

    return train_dataloader, val_dataloader
