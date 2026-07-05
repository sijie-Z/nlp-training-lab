"""
随机种子管理

作用：固定所有随机种子，保证实验结果可复现

为什么重要：
模型训练有大量随机性（参数初始化、shuffle、dropout），
不固定种子的话，同样代码跑两次结果不同，没法判断改动是否有效。
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    固定所有随机种子

    参数：
        seed: 随机种子（推荐 42）

    用法：
        set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 使用确定性算法（牺牲一点速度，保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
