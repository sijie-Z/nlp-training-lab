"""
ReCoLoRA: Spectrum-Aware Recursive Consolidation for Continual LLM Fine-Tuning

arXiv: 2607.07719 (July 4, 2026)

核心思想:
    LoRA 做持续微调时, 每学一个新任务就要加新的 adapter。
    累积的 adapter 越来越多, 推理越来越慢。
    ReCoLoRA 在任务之间递归地"合并" adapter 权重到主干,
    用 SVD 频谱分析确定哪些 rank 方向真正重要,
    然后"回收"不重要的 rank 给下一个任务用。

标准 LoRA 持续微调的问题:
    Task1 → LoRA_A1, LoRA_B1 (各 rank=8)
    Task2 → LoRA_A2, LoRA_B2 (再各 rank=8)
    ...
    推理时: W' = W + Σ A_i·B_i  → 计算量随任务数线性增长

ReCoLoRA 的解决方案:
    Task1 → LoRA → 训完 → SVD 频谱分析 → 合并到 W
    → "回收" 未使用的 rank 容量
    → Task2 → 用回收的 rank 继续训
    → 推理时: W' = W + single_A·B  → 恒定开销


关键创新:
    1. Randomized SVD Initialization:
       不放任 A·B=0（标准 LoRA 初始化），而是用随机 SVD 初始化
       → LoRA adapter 的初始状态覆盖更多频谱方向

    2. Recursive Weight Re-decomposition:
       任务完成后, 对 ΔW = A·B 做 SVD
       → 保留 top-k 奇异值对应的 rank（真正有用的方向）
       → 回收 bottom-(r-k) 个 rank 用于下一个任务

    3. Spectrum-Aware Allocation:
       不同任务可能需要不同数量的 rank
       → 根据累积 SVD 频谱动态分配 rank

简化版实现 (用于小模型演示):
    - 在 TinyGPT 上模拟多任务持续微调
    - Task1: 用 LoRA 训练 → 频谱分析 → 合并
    - Task2: 用回收的 rank 训练
    - 对比: 标准 LoRA (累积) vs ReCoLoRA (合并)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """标准 LoRA layer: W' = W + B @ A"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 原始权重（冻结）
        self.register_buffer('weight', torch.zeros(out_features, in_features))

        # LoRA：A 用 kaiming 初始化, B 用零初始化 → 初始 W' = W
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def set_weight(self, w):
        self.weight.data.copy_(w)

    def forward(self, x):
        # x: (..., in_features)
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    def get_delta(self):
        """获取 LoRA 的权重增量 ΔW = B @ A"""
        return (self.lora_B @ self.lora_A) * self.scaling


def spectrum_analysis(delta_W, keep_ratio=0.7):
    """
    对 ΔW 做 SVD 频谱分析, 返回 top-k 方向

    Args:
        delta_W: (out, in) LoRA 学的权重增量
        keep_ratio: 保留多少 rank

    Returns:
        kept_delta: 只包含 top-k 方向的 ΔW
        singular_values: 所有奇异值
        kept_rank: 保留的 rank 数
    """
    U, S, V = torch.svd(delta_W.float())

    # 保留 top-k% 的 rank
    total_singular_mass = S.sum()
    cumulative = S.cumsum(dim=0)

    # 找累积 90% 能量的 rank 数
    k = (cumulative / total_singular_mass > keep_ratio).nonzero(as_tuple=True)
    k = k[0][0].item() + 1 if len(k[0]) > 0 else len(S)

    # 重建只保留 top-k 的 delta
    kept_delta = (U[:, :k] * S[:k].unsqueeze(0)) @ V[:, :k].T
    dropped_rank = len(S) - k

    return kept_delta.to(delta_W.dtype), S, k, dropped_rank


def rescale_lora_rank(lora_A, lora_B, new_rank, alpha=16):
    """
    回收 LoRA rank → 创建新的 LoRA layer 用剩余的 rank

    把 ΔW = B @ A 的低频谱部分丢弃, 重新分解为 r' 个 rank (r' <= 原 rank)
    """
    delta = lora_B @ lora_A
    U, S, Vt = torch.svd(delta.float())

    # 取 top-new_rank 个分量
    new_A = (Vt[:new_rank, :].T * torch.sqrt(S[:new_rank]).unsqueeze(0)).T
    new_B = U[:, :new_rank] * torch.sqrt(S[:new_rank]).unsqueeze(0)
    new_A = new_A.to(lora_A.dtype)
    new_B = new_B.to(lora_B.dtype)

    return new_A, new_B


class ReCoLoRADemo:
    """
    ReCoLoRA 演示: 在 TinyGPT 上模拟多任务持续微调

    Task1: 训练 → 频谱分析 → 合并到 FC 权重 → 回收 rank
    Task2: 用回收的 rank 训练 → 合并
    → 推理时只有 1 个 adapter, 不是 2 个
    """

    def __init__(self, linear_layer_weight, rank=8, alpha=16, keep_ratio=0.7):
        """
        Args:
            linear_layer_weight: (out_features, in_features) 原始权重
            rank: LoRA rank
            alpha: LoRA 缩放因子
            keep_ratio: 频谱分析保留比例
        """
        self.weight = linear_layer_weight.clone()
        self.out_features, self.in_features = linear_layer_weight.shape
        self.rank = rank
        self.alpha = alpha
        self.keep_ratio = keep_ratio

        self.frozen_weight = linear_layer_weight.clone()
        # 当前活跃的 LoRA
        self.current_lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.02)
        self.current_lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank

        self.task_count = 0
        self.history = []

    def get_current_weight(self):
        """获取当前有效权重: W_frozen + LoRA"""
        delta = self.current_lora_B @ self.current_lora_A * self.scaling
        return self.frozen_weight + delta

    def consolidate_task(self):
        """
        完成一个任务 → 频谱分析 → 合并重要方向 → 回收 rank

        返回: kept_rank, dropped_rank
        """
        delta = (self.current_lora_B @ self.current_lora_A) * self.scaling

        # 频谱分析
        kept_delta, S, kept_rank, dropped_rank = spectrum_analysis(
            delta, keep_ratio=self.keep_ratio
        )

        # 合并到 frozen weight
        self.frozen_weight = self.frozen_weight + kept_delta

        # 回收 rank: 用剩余的 rank 重建 LoRA
        if dropped_rank > 0:
            new_rank = dropped_rank
            self.current_lora_A = nn.Parameter(torch.randn(new_rank, self.in_features) * 0.02)
            self.current_lora_B = nn.Parameter(torch.zeros(self.out_features, new_rank))
            self.rank = new_rank
            self.scaling = self.alpha / self.rank

        self.task_count += 1
        self.history.append({
            'task': self.task_count,
            'dropped_rank': dropped_rank,
            'kept_rank': kept_rank,
            'singular_values': S.tolist(),
        })

        return kept_rank, dropped_rank

    def forward(self, x):
        """带当前 LoRA 的 forward"""
        base = F.linear(x, self.frozen_weight)
        delta = self.current_lora_B @ self.current_lora_A * self.scaling
        lora = F.linear(x, delta)
        return base + lora
