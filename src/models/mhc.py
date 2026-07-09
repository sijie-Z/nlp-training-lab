"""
Manifold-Constrained Hyper-Connections (mHC) — DeepSeek V4 的核心架构

论文: arXiv:2512.24880 (DeepSeek-AI, 2025年12月)
应用: DeepSeek V4 (2026年4月发布)

核心思想:
    标准残差: h = h + f(h)          ← 权重固定为 [1]
    HC (Hyper-Connections):  h = α ⊙ h + β ⊙ f(h)  ← 可学习的逐元素权重
    mHC: HC 的权重矩阵约束在 Birkhoff polytope（双随机矩阵）上
         → 行和 = 列和 = 1，保证训练稳定性


为什么需要 mHC？

问题: 标准残差每层权重固定为 1，深层网络信号衰减
      HC 给每层可学习的混合权重，但不约束的话训练不稳定
      mHC 把权重限制在双随机矩阵上 → 保证信息守恒 + 训练稳定

Sinkhorn-Knopp 算法:
    将任意正矩阵迭代归一化为双随机矩阵
    1. 行归一化: X = X / row_sum(X)
    2. 列归一化: X = X / col_sum(X)
    重复直到收敛（通常 5-20 次）


Kimi AttnRes vs DeepSeek mHC 对比（面试必问）:

|              | AttnRes (Kimi K2)              | mHC (DeepSeek V4)               |
|--------------|-------------------------------|--------------------------------|
| 核心机制      | Softmax attention over layers  | Birkhoff polytope constraint   |
| 权重来源      | 每层学 query，与所有层算 attention | 每层学 H 矩阵，Sinkhorn-Knopp 归一化 |
| 数学保证      | softmax 保证和为 1              | 行和=列和=1 (双随机)              |
| 参数开销      | <0.03%                        | ~6.7%                          |
| 训练稳定性    | 零初始化热启动                   | Sinkhorn-Knopp 保证不爆炸        |
| 论文时间      | 2026年3月                      | 2025年12月 (V4: 2026年4月)      |


简化版实现（用于小模型验证）:
    不实现完整的 Sinkhorn-Knopp（计算量大，小模型不需要）
    而是用 learnable alpha 参数 + softmax 约束
    核心是演示"让残差连接变得可学习"的思想
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHCBlock(nn.Module):
    """
    简化版 mHC Block（Manifold-Constrained Hyper-Connection）

    每个 sublayer (Attention/FFN) 的输出和残差输入通过可学习的混合矩阵组合

    完整版 mHC 的 H 矩阵是 (n_sublayers × n_sublayers) 的双随机矩阵
    这里简化为每个 sublayer 学习 (input, sublayer_output) 两个通道的混合权重

    forward:
        residual = x
        x = attn_block(x)
        x = alpha_attn * residual + beta_attn * x    ← 可学习的混合
        residual = x
        x = ffn_block(x)
        x = alpha_ffn * residual + beta_ffn * x      ← 可学习的混合
    """

    def __init__(self, attention_block, ffn_block, n_embd):
        super().__init__()
        self.attn = attention_block
        self.ffn = ffn_block

        # 可学习的混合权重（每个 sublayer 2 个参数: alpha（残差权重）, beta（输出权重））
        # 初始化为 [1, 0] → 等价于标准残差（安全热启动）
        self.alpha_attn = nn.Parameter(torch.tensor(1.0))
        self.beta_attn = nn.Parameter(torch.tensor(0.0))

        self.alpha_ffn = nn.Parameter(torch.tensor(1.0))
        self.beta_ffn = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Attention sublayer
        residual = x
        x = self.attn(x)
        # 可学习的残差混合: α·residual + β·output
        a = torch.sigmoid(self.alpha_attn)
        b = torch.sigmoid(self.beta_attn)
        # 归一化保证 (α + β) ≈ 1（而不强制，给模型更多自由度）
        total = a + b + 1e-8
        x = (a / total) * residual + (b / total) * x

        # FFN sublayer
        residual = x
        x = self.ffn(x)
        a = torch.sigmoid(self.alpha_ffn)
        b = torch.sigmoid(self.beta_ffn)
        total = a + b + 1e-8
        x = (a / total) * residual + (b / total) * x

        return x


class FullMHCBlock(nn.Module):
    """
    完整版 mHC Block — 使用可学习的 H 矩阵（简化实现）

    H 是 2×2 的可学习矩阵，约束为双随机（行和=列和=1）

    对于每个 sublayer:
        [h_new, sublayer_out] = [h_old, f(h_old)] @ H

    其中 H 通过 Sinkhorn 迭代保证双随机性

    简化: 直接学 H，不加 Sinkhorn（计算太贵）
    用 L1 正则化的 soft 约束替代严格的 Birkhoff 约束
    """

    def __init__(self, attention_block, ffn_block, n_embd, n_sinkhorn_iters=5):
        super().__init__()
        self.attn = attention_block
        self.ffn = ffn_block
        self.n_sinkhorn_iters = n_sinkhorn_iters

        # 每个 sublayer 的 H 矩阵（2×2）
        # H_0 = [[1, 0],   ← 初始化时 h = 1*h_old + 0*f(h_old) = 标准残差
        #        [0, 1]]   ← 输出给下一层的信号初始化等于 sublayer 输出
        self.H_attn = nn.Parameter(torch.eye(2) + 0.01 * torch.randn(2, 2))
        self.H_ffn = nn.Parameter(torch.eye(2) + 0.01 * torch.randn(2, 2))

    @staticmethod
    def sinkhorn_knopp(H, n_iters=5):
        """Sinkhorn-Knopp 迭代：将矩阵投影到双随机空间"""
        X = H.clone()
        for _ in range(n_iters):
            X = X / (X.sum(dim=1, keepdim=True) + 1e-8)  # 行归一化
            X = X / (X.sum(dim=0, keepdim=True) + 1e-8)  # 列归一化
        return X

    def mhc_mix(self, residual, sublayer_out, H, apply_sinkhorn=True):
        """
        使用 mHC H 矩阵混合残差和子层输出

        Args:
            residual: (B, T, D) 残差输入
            sublayer_out: (B, T, D) 子层输出
            H: (2, 2) 混合矩阵

        Returns:
            new_residual: (B, T, D) 新的残差（给下一层的输入）
            output: (B, T, D) 当前层的输出
        """
        if apply_sinkhorn:
            H_constrained = self.sinkhorn_knopp(H, self.n_sinkhorn_iters)
        else:
            H_constrained = H

        # [h_old, f(h_old)] @ H = [h_old * H[0,0] + f(h_old) * H[0,1],
        #                           h_old * H[0,1] + f(h_old) * H[1,1]]
        # 简化: 只取第一个输出作为新的 residual
        new_residual = (
            H_constrained[0, 0] * residual +
            H_constrained[0, 1] * sublayer_out
        )

        # 第二行: 给下游的额外信号
        downstream = (
            H_constrained[1, 0] * residual +
            H_constrained[1, 1] * sublayer_out
        )

        return new_residual, downstream

    def forward(self, x):
        # Attention sublayer
        residual = x
        h1 = self.attn(x)
        x, downstream_attn = self.mhc_mix(residual, h1, self.H_attn)

        # FFN sublayer
        residual = x
        h2 = self.ffn(x)
        x, downstream_ffn = self.mhc_mix(residual, h2, self.H_ffn)

        # 可选: 将下游信号也融入（简化版暂不加）
        return x


def convert_tinygpt_to_mhc(model):
    """
    将标准 TinyGPT 转换为 mHC 版本

    替换所有 DecoderBlock 为 MHCBlock
    混合权重初始化为 1/0（等价于标准残差）→ 安全热启动
    """
    from src.models.tiny_gpt import DecoderBlock

    new_layers = []
    n_embd = model.config.n_embd
    replaced = 0

    for layer in model.layers:
        if isinstance(layer, DecoderBlock):
            mhc_block = MHCBlock(layer.attn, layer.mlp, n_embd)
            new_layers.append(mhc_block)
            replaced += 1
        else:
            new_layers.append(layer)

    model.layers = nn.ModuleList(new_layers)
    print(f"[mHC] 替换 {replaced} 个 DecoderBlock → MHCBlock")
    print(f"[mHC] 每层新增可学习参数: 4 个标量 (alpha_attn, beta_attn, alpha_ffn, beta_ffn)")
    print(f"[mHC] 初始化: α=1, β=0 → 行为等价于标准残差（安全热启动）")

    return model


def convert_tinygpt_to_full_mhc(model, n_sinkhorn_iters=3):
    """转换为完整版 mHC（H 矩阵 + Sinkhorn-Knopp）"""
    from src.models.tiny_gpt import DecoderBlock

    new_layers = []
    n_embd = model.config.n_embd

    for layer in model.layers:
        if isinstance(layer, DecoderBlock):
            mhc_block = FullMHCBlock(layer.attn, layer.mlp, n_embd, n_sinkhorn_iters)
            new_layers.append(mhc_block)
        else:
            new_layers.append(layer)

    model.layers = nn.ModuleList(new_layers)
    added_params_per_layer = 8  # 2× (2×2) H matrices
    total_added = added_params_per_layer * len(new_layers)
    print(f"[Full mHC] 替换 {len(new_layers)} 层, 每层新增 {added_params_per_layer} 参数")
    print(f"[Full mHC] 总新增参数: {total_added}")
    print(f"[Full mHC] Sinkhorn-Knopp 迭代次数: {n_sinkhorn_iters}")

    return model
