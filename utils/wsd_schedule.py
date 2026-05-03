import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def get_wsd_schedule(
    optimizer, 
    num_warmup_steps: int, 
    num_decay_steps: int, 
    num_training_steps: int, 
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    创建一个 WSD (Warmup-Stable-Decay) 调度器:
    1. Warmup: 线性上升
    2. Stable: 保持 Constant (最高 LR)
    3. Decay: 余弦下降 (Cosine Decay) 到 min_lr
    """
    
    def lr_lambda(current_step):
        # 1. Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 计算开始 Decay 的步数
        start_decay_step = num_training_steps - num_decay_steps
        
        # 2. Stable (Constant) 阶段
        if current_step < start_decay_step:
            return 1.0
        
        # 3. Decay 阶段
        # 计算在 Decay 阶段内的进度 (0.0 -> 1.0)
        progress = float(current_step - start_decay_step) / float(max(1, num_decay_steps))
        
        # Cosine 计算: 从 1.0 降到 0.0
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # 考虑最小学习率 (min_lr_ratio)
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch)