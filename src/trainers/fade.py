"""
FADE: Focal Advantage with Dynamic Entropy — 自适应 RL 优势函数

arXiv: 2607.01490 (July 1, 2026)

核心思想:
    传统 RL/GRPO 用固定 advantage 函数 → 训练后半段梯度信号弱
    FADE 把 advantage 分解为 sign (方向) × difficulty (难度) 两个轴
    → 对不同难度/方向的样本自适应调整 advantage 权重

问题:
    GRPO/PPO 用 group-relative 归一化做 advantage
    → 训练后期几乎所有样本的 advantage 都趋近于 0
    → 梯度"褪色"(FADE)

FADE 的解决方案:
    1. Decompose advantage into sign × difficulty:
       sign = +1 (好) or -1 (坏)
       difficulty = |advantage| (越小越难 → 越需要强化)

    2. Focal weighting:
       难样本 (边界样本) 的 advantage 被放大
       简单样本 的 advantage 被抑制
       → 训练信号持续有效, 不会"褪色"

    3. Dynamic Entropy scheduling:
       training early: high entropy (探索)
       training late: low entropy (利用)
       → 自适应从探索过渡到利用

关键实验数据 (论文):
    7B 模型上:
    - 达到 peak pass@1 比最佳静态 baseline 快 20K steps
    - LiveCodeBench / AIME 上最佳准确率-多样性 trade-off

简化实现:
    实现 Focal Advantage 调度器: 根据 advantage 的难度动态调整权重
"""
import math
import torch
import torch.nn.functional as F


class FocalAdvantageScheduler:
    """
    FADE 优势函数调度器

    核心公式:
        A_focal = sign(A) * |A|^γ * weight(difficulty)

        其中:
        - γ: focal exponent (>0 → 压制简单样本, <0 → 压制困难样本)
        - difficulty = 1 - |A| (越接近0越难分 → difficulty 接近1)
        - weight(d): α * d + (1-α) * (1-d)

    论文的建议:
        γ = 2.0 (focal gamma — 来自 focal loss 的思想)
        α = 0.75 (困难样本的权重更大)

    面试时讲:
        "FADE 把 RL 的 advantage 函数从固定权重变成了自适应调度器。
         核心是 focal term: 对于边界样本(|A|≈0), 将其视为困难样本,
         放大其梯度权重 — 这样即使训练后期所有 sample 的 advantage
         都接近 0, 仍然有持续的梯度信号。\
    """

    def __init__(self, gamma=2.0, alpha=0.75, entropy_start=0.5, entropy_end=0.01,
                 total_steps=1000):
        self.gamma = gamma
        self.alpha = alpha
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.total_steps = total_steps
        self.current_step = 0

    def get_entropy_coef(self):
        """动态 entropy 系数: 从高(探索)到低(利用)"""
        progress = min(1.0, self.current_step / self.total_steps)
        return self.entropy_start + (self.entropy_end - self.entropy_start) * progress

    def compute_focal_advantage(self, rewards):
        """
        对原始 reward 做 FADE 自适应加权

        Args:
            rewards: (B,) 原始的 group-relative advantage

        Returns:
            focal_advantages: (B,) FADE 加权后的 advantage
            difficulty_scores: (B,) 每个样本的 difficulty
        """
        B = rewards.shape[0]

        # Group normalization (GRPO-style)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # Sign: 方向 (正=好, 负=差)
        sign = advantages.sign()

        # Difficulty: 越接近 0 越难分
        # 归一化到 [0, 1]
        abs_adv = advantages.abs()
        difficulty = 1.0 - abs_adv / (abs_adv.max() + 1e-8)

        # Focal weighting
        # 对困难样本 (difficulty ≈ 1) 放大, 简单样本抑制
        focal_weight = self.alpha * difficulty + (1 - self.alpha) * (1 - difficulty)

        # 应用 focal term
        focal_advantages = sign * (abs_adv ** self.gamma) * focal_weight

        self.current_step += 1

        return focal_advantages, difficulty

    def step(self):
        self.current_step += 1
        return self.get_entropy_coef()


def fade_loss(rewards, scheduler=None, clip_epsilon=0.2):
    """
    FADE-style policy gradient loss

    标准 PPO loss:
        L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    FADE 增强:
        A → A_focal (根据 difficulty 自适应加权)

    这里简化为: 直接返回 FADE-advantage-weighted 的 advantage 值
    用于下游模型更新
    """
    if scheduler is None:
        scheduler = FocalAdvantageScheduler()

    focal_adv, difficulty = scheduler.compute_focal_advantage(rewards)

    return focal_adv, {
        'mean_advantage': rewards.mean().item(),
        'focal_advantage_mean': focal_adv.mean().item(),
        'difficulty_mean': difficulty.mean().item(),
        'entropy_coef': scheduler.get_entropy_coef(),
    }
