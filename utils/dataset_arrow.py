import os
import glob
import torch
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader

def collate_fn_with_labels(batch):
    # 从 batch 中提取 input_ids
    input_ids = torch.stack([item["input_ids"] for item in batch])
    
    # 创建 labels (右移一位)
    # labels[i] = input_ids[i+1]
    labels = input_ids.clone()
    
    return {
        "input_ids": input_ids,
        "labels": labels[:, 1:] # 去掉第一个 token
    }
    
def build_dataloaders(config, tokenizer):
    """
    读取预处理好的 Arrow 分片并构建 DataLoader
    """
    data_dir = config.dataset.params.data_path
    batch_size = config.training.batch_size
    num_workers = config.training.dataloader_workers # 建议设为 4 或 8
    
    # 1. 扫描所有分片
    print(f"🔍 Scanning shards in {data_dir}...")
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard-*")))
    
    if not shard_paths:
        raise ValueError(f"No shards found in {data_dir}")
    
    print(f"✅ Found {len(shard_paths)} shards. Loading (lazy mapping)...")

    # 2. 懒加载所有分片 (不会读取到内存，只是建立内存映射)
    # 注意：如果文件数成千上万，可能需要考虑打开文件句柄限制的问题
    datasets_list = []
    for path in tqdm(shard_paths, desc="Loading Shards"):
        try:
            ds = load_from_disk(path, keep_in_memory=False)
            datasets_list.append(ds)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load {path}: {e}")

    # 3. 合并为一个逻辑数据集
    full_dataset = concatenate_datasets(datasets_list)
    print(f"📊 Total samples: {len(full_dataset)}")
    

    # 4. 设定 PyTorch 格式
    # 这一步至关重要，它告诉 dataset 在被访问时返回 PyTorch Tensor 而不是 Python List
    full_dataset.set_format(type='torch', columns=['input_ids'])


    split_dataset = full_dataset.train_test_split(test_size=0.0001, seed=config.training.seed)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"🚆 Train size: {len(train_dataset)}, 🧪 Val size: {len(val_dataset)}")

    # 6. 构建 DataLoader
    # pin_memory=True 能加速 CPU 到 GPU 的传输
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_with_labels
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_with_labels
    )

    return train_dataloader, val_dataloader