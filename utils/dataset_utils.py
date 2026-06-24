
def get_dataloaders(config, tokenizer):
    dataset_class = config.dataset.class_name
    if dataset_class == "ImageNetFlowCacheDataset":
        from .dataset_imagenet_flow_cache import build_imagenet_flow_cache_dataloaders
        train_dataloader, val_dataloader = build_imagenet_flow_cache_dataloaders(
                config=config,
                tokenizer=tokenizer,
            )
    elif "imagenet" in dataset_class.lower() and "latent" in dataset_class.lower():
        from .dataset_imagenet_latent import build_imagenet_latent_dataloaders
        train_dataloader, val_dataloader = build_imagenet_latent_dataloaders(
                config=config,
                tokenizer=tokenizer,
            )
    elif "omnicorpus" in dataset_class.lower():
        from .dataset_omnicorpus import build_omnicorpus_dataloaders
        train_dataloader, val_dataloader = build_omnicorpus_dataloaders(
                config=config,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class}")

    return train_dataloader, val_dataloader
