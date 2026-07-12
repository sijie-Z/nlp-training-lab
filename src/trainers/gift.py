"""
GIFT: Geometry-Informed Low-Precision Gradient Communication

arXiv: 2607.07494 (July 8, 2026)

核心思想:
    分布式训练中把梯度压到 FP8/NVFP4 来减少通信量。
    但直接量化会破坏梯度的几何方向——某些方向的量化误差比其他方向大得多。
    GIFT 在量化前先把梯度变换到"近等距空间"(near-isotropic space),
    再做量化 → 所有方向的量化误差相同 → 训练质量不下降。

为什么直接量化效果不好:
    梯度的分布不是均匀的——有些参数方向的梯度总是大,
    有些总是小。FP8 的量化间隔是固定的,
    小梯度方向直接变 0 → 优化器"看不到"这些方向。

GIFT 的解决:
    1. 计算梯度的协方差估计 → 确定哪些方向被"挤压"
    2. 用等距变换 (whitening) 把梯度映射到等距空间
    3. 在等距空间中量化 → 所有方向同等对待
    4. 通信后用逆变换恢复

简化版 (单卡演示):
    单卡不需要通信优化, 但"梯度等距化"本身有价值:
    - 模拟 FP8 量化环境
    - 对比直接量化 vs GIFT 等距化后量化
    - 证明 GIFT 能在相同 bit-width 下保留更多方向信息
"""
import math
import torch
import torch.nn as nn


def simulate_fp8_quantize(tensor, exp_bits=4, mantissa_bits=3):
    """
    模拟 FP8 E4M3 量化

    FP8 E4M3: 4 位指数, 3 位尾数 → 范围: ~10^4.5, 精度: 1/32

    Args:
        tensor: (任意形状) 要量化的梯度
    Returns:
        quantized: 量化后的 tensor (FP32 表示, 但值只取 FP8 能表示的那些)
    """
    # FP8 E4M3 的 max
    max_val = 448.0  # 2^(2^4-1-7) * (1 + 7/8) = 2^8 * 1.875 ≈ 480

    # 存储符号
    sign = torch.sign(tensor)
    abs_t = tensor.abs()

    # Clip to FP8 range
    abs_t = abs_t.clamp(max=max_val)

    # 量化到 FP8 的离散值
    # 简化: 用均匀量化模拟 FP8
    q_levels = 2 ** (mantissa_bits + 1)  # 有效量化级别（含隐藏位）
    scale = max_val / q_levels

    # 量化
    abs_q = (abs_t / scale).round() * scale
    abs_q = abs_q.clamp(max=max_val)

    return sign * abs_q


def compute_gradient_covariance(grad_list):
    """
    估计梯度的协方差 — 用于等距化

    简化: 用参数的 L2 norm 作为"方向重要性"的代理
    真实 GIFT 用 running EMA of outer product
    """
    total = 0
    for g in grad_list:
        total += g.norm().item() ** 2
    return math.sqrt(total / len(grad_list))


def gift_whiten_gradient(grad, running_std=None):
    """
    GIFT 等距化: 将梯度归一化到近似均匀的幅度

    核心: 对每个参数组, 按 running_std 做 rescale
    → 所有方向的梯度幅度接近 → 量化误差均匀

    Args:
        grad: 当前梯度
        running_std: 该参数组的历史 std 估计
    """
    if running_std is None or running_std < 1e-8:
        return grad, grad.std().item()

    # Whiten: divide by running std
    whitened = grad / running_std

    # 记录当前 std (用于通信端恢复)
    current_std = grad.std().item()

    return whitened, current_std


def gift_unwhiten_gradient(whitened_grad, original_std):
    """GIFT 逆变换: 恢复原始幅度"""
    return whitened_grad * original_std


class GIFTQuantizer:
    """
    GIFT 量化器: 等距化 → FP8 量化 → 逆等距化
    """

    def __init__(self, use_gift=True):
        self.use_gift = use_gift
        self.running_stds = {}  # param_id → running std estimate

    def quantize(self, param_id, grad):
        """
        对单个参数的梯度做 GIFT 量化

        Args:
            param_id: 参数的唯一 id
            grad: 梯度 tensor

        Returns:
            quantized_grad: 量化后的梯度
            info: 量化信息
        """
        original_norm = grad.norm().item()

        if self.use_gift and param_id in self.running_stds:
            # Step 1: 等距化
            whitened, current_std = gift_whiten_gradient(
                grad, self.running_stds[param_id]
            )

            # Step 2: FP8 量化
            quantized = simulate_fp8_quantize(whitened)

            # Step 3: 逆变换
            quantized = gift_unwhiten_gradient(quantized, current_std)
        else:
            # 直接量化
            quantized = simulate_fp8_quantize(grad)
            current_std = grad.std().item()

        # 更新 running std
        if param_id not in self.running_stds:
            self.running_stds[param_id] = current_std
        else:
            self.running_stds[param_id] = (
                0.99 * self.running_stds[param_id] + 0.01 * current_std
            )

        quantized_norm = quantized.norm().item()
        info = {
            'original_norm': original_norm,
            'quantized_norm': quantized_norm,
            'direction_cosine': (grad * quantized).sum().item() / max(1e-8, original_norm * quantized_norm),
            'running_std': self.running_stds.get(param_id, 0.0),
        }

        return quantized, info
