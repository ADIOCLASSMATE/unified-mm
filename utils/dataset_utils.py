


def get_dataloaders(config, tokenizer):
    dataset_class = config.dataset.class_name
    if "arrow" in dataset_class.lower():
        from .dataset_arrow import build_dataloaders
        train_dataloader, val_dataloader = build_dataloaders(
                config=config,
                tokenizer=tokenizer,
            )
    elif "huggingface" in dataset_class.lower():
        from .dataset_hf import get_dataloaders
        train_dataloader, val_dataloader = get_dataloaders(
                config=config,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class}")
    
    
    return train_dataloader, val_dataloader