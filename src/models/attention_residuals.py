"""
Attention Residuals (AttnRes) — Kimi K2 团队 2026年3月 提出的架构创新

论文: arXiv:2603.15031
核心思想: 用 learned softmax attention 替代固定的加法残差连接

标准残差（用了 10 年）：
    h_l = h_{l-1} + f(h_{l-1})   ← 每层权重都是 1，不能选择性关注

Attention Residuals:
    h_l = Σ softmax(q_l^T · k_i) · v_i   ← 每层学习自己的 query，
                                             选择性地关注前面哪些层


时间-深度对偶性（Kimi 团队的洞见）：
    时间轴:  RNN(固定时序聚合) → Transformer Attention(选择性时序聚合)
    深度轴:  残差(固定深度聚合) → Attention Residuals(选择性深度聚合)

    相当于把注意力"旋转90度"——从时间维度转到深度维度。


Block AttnRes（论文推荐的高效版本）：
    - 每 6 层分成一个 block
    - block 内部用标准残差（常规通信）
    - block 之间用 Attention Residuals（选择性通信）
    - 内存/通信从 O(L) 降到 O(N)，N=block 数


工程特性（面试友好）：
    - 参数开销: <0.03%（只加几个 query vector 和 norm 层）
    - 训练开销: <4%
    - 推理延迟: <2%
    - Query 零初始化 → 训练初期行为 ≈ 标准残差（安全）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionResidualsBlock(nn.Module):
    """
    Block Attention Residuals — 论文推荐的实现

    架构:
        原始 DecoderBlock 的 forward 改为:
        1. 每个 block 内部: 标准 Pre-norm + Attention + Residual + Pre-norm + FFN + Residual
        2. Block 之间: 用 softmax attention 加权聚合之前的 block 输出

    Args:
        num_blocks: block 数量
        n_embd: 隐藏维度
        recency_bias_init: recency bias 初始值（控制初始化时对最近 block 的偏好）
    """

    def __init__(self, num_blocks, n_embd, recency_bias_init=0.0):
        super().__init__()
        self.num_blocks = num_blocks

        # 每个 block 学习一个 query vector（用于决定"我该关注哪些前面的 block"）
        # 论文说 pseudo-query — 不依赖当前 token，是一个全局的可学习参数
        self.query = nn.Parameter(torch.zeros(num_blocks, n_embd))

        # Key/Value normalization（防止不同 block 输出的量级偏差导致 attention 被 dominate）
        self.kv_norm = nn.LayerNorm(n_embd)

        # Recency bias：给最近的 block 一个额外的偏置
        # 类似于 Transformer 位置编码在深度维度的应用
        self.recency_bias = nn.Parameter(torch.full((num_blocks,), recency_bias_init))

        # 初始化：query 零初始化 → 初始 softmax 是均匀分布 → ≈ 所有 block 的平均
        # 这保证了训练初期行为和标准残差差不多（安全的热启动）

        print(f"[AttnRes] Block 模式: {num_blocks} 个 block, "
              f"参数量: {sum(p.numel() for p in self.parameters()):,} "
              f"(n_embd={n_embd})")

    def forward(self, block_outputs):
        """
        Args:
            block_outputs: list of tensors, 每个形状 (B, T, n_embd)
                           block_outputs[0] 是最早的 block

        Returns:
            output: (B, T, n_embd) 加权聚合后的表示
        """
        n = len(block_outputs)
        assert n <= self.num_blocks, f"传入 {n} 个 block 但只配置了 {self.num_blocks}"

        B, T, D = block_outputs[0].shape
        device = block_outputs[0].device

        # 堆叠所有 block 输出
        # stack: (n_blocks, B, T, D)
        stacked = torch.stack(block_outputs, dim=0)

        # Key = 归一化后的 block 输出
        keys = self.kv_norm(stacked)  # (n_blocks, B, T, D)

        # Query: (n_blocks, D) — 每个 block 学一个 query
        # broadcast 到 (n_blocks, B, T, D)
        queries = self.query[:n].unsqueeze(1).unsqueeze(2)  # (n_blocks, 1, 1, D)

        # 计算注意力分数: Q·K
        # (n_blocks, 1, 1, D) @ (n_blocks, B, D, T)? 不对
        # 改一下：把 K 的 n_blocks 维度当作"key 序列"
        # 对每个 block 输出位置，计算其 query 和所有 block 的 key 的相似度

        # queries: (n_blocks, B, T, D) - 复制到 batch 和 seq 维度
        q = queries.expand(n, B, T, D)

        # 点积: q·k → (n_blocks, B, T)
        # 每个 block 位置的分数 = 该 block 的 query · 该 block 的 key
        scores = (q * keys).sum(dim=-1)  # (n_blocks, B, T)

        # Recency bias（对最近的 block 有额外偏置）
        bias = self.recency_bias[:n].unsqueeze(1).unsqueeze(2)  # (n_blocks, 1, 1)
        bias = bias.expand(n, B, T)

        # 设计 attention mask：每个 block 只能关注自己和之前的 block（因果性）
        # 1. 每个位置的得分是对应 block 作为 key 的得分
        # 2. 对 n 个 block 做 softmax

        # 转换：scores (n_blocks, B, T) 需要在 dim=0（block 维度）上做 softmax
        # 这代表"对于每个 (B,T) 位置，各个历史 block 的贡献权重"

        # 加权: 加上 recency bias 再 softmax
        weights = F.softmax(scores + bias, dim=0)  # (n_blocks, B, T)

        # 加权聚合: 用 softmax 权重聚合各个 block 的输出
        # weights: (n_blocks, B, T) → unsqueeze to (n_blocks, B, T, 1)
        # stacked: (n_blocks, B, T, D)
        output = (weights.unsqueeze(-1) * stacked).sum(dim=0)  # (B, T, D)

        return output


class BlockAttnResWrapper(nn.Module):
    """
    将 TinyGPT 的 DecoderBlock 列表包装成 Block Attention Residuals

    用法:
        original_layers = model.layers  # list of DecoderBlock (e.g., 24 layers)
        block_size = 6  # 每 6 层一个 block
        wrapper = BlockAttnResWrapper(original_layers, n_embd=384, block_size=6)

    forward:
        1. 把 layers 分成 N 个 block
        2. 每个 block 内部正常 forward
        3. 收集所有 block 的输出
        4. 用 Attention Residuals 聚合
    """

    def __init__(self, layers, n_embd, block_size=6):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.block_size = block_size
        num_blocks = (len(layers) + block_size - 1) // block_size
        self.attn_res = AttentionResidualsBlock(num_blocks, n_embd)

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) 初始的 hidden state

        Returns:
            x: (B, T, n_embd) 聚合后的 hidden state
        """
        block_outputs = []
        current_x = x

        for i, layer in enumerate(self.layers):
            # 正常的层内 forward
            current_x = layer(current_x)

            # 每 block_size 层收集一次输出
            if (i + 1) % self.block_size == 0 or i == len(self.layers) - 1:
                block_outputs.append(current_x)

        # 如果 block_outputs 数量 > 1，用 AttnRes 聚合
        if len(block_outputs) > 1:
            x = self.attn_res(block_outputs)
        else:
            x = block_outputs[0]

        return x


def convert_tinygpt_to_attnres(model, block_size=6):
    """
    将 TinyGPT 模型转换为 Attention Residuals 版本

    替换 model.layers 的 forward 逻辑，
    使得层间通信变为选择性深度注意力。

    Args:
        model: TinyGPT 模型实例
        block_size: 每个 block 包含的层数

    Returns:
        修改后的 model（原地修改）
    """
    n_layers = len(model.layers)
    n_embd = model.config.n_embd

    print(f"[AttnRes] 转换 TinyGPT: {n_layers} 层 → {n_layers//block_size} 个 block (block_size={block_size})")

    # 替换 layers 为带 AttnRes 的版本
    wrapper = BlockAttnResWrapper(model.layers, n_embd, block_size)

    # 把 wrapper 当作新的 layers processor
    model._attnres_wrapper = wrapper
    model._use_attnres = True

    # 添加参数
    added_params = sum(p.numel() for p in wrapper.attn_res.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[AttnRes] 新增参数: {added_params:,} / {total_params:,} ({100*added_params/total_params:.3f}%)")

    return model


def attnres_forward(model, input_ids, labels=None):
    """
    使用 Attention Residuals 的 forward 替代原 TinyGPT.forward

    核心变化:
        原始: for layer in layers: x = layer(x)
        新:   x = attn_res_wrapper(x)  ← 块间注意力聚合

    其余保持不变（embedding、final LN、LM head）。
    """
    config = model.config
    B, T = input_ids.shape

    positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    tok_emb = model.wte(input_ids)
    pos_emb = model.wpe(positions)
    x = model.drop(tok_emb + pos_emb)

    # 核心变化：用 AttnRes wrapper 替代逐层 forward
    if hasattr(model, '_attnres_wrapper'):
        x = model._attnres_wrapper(x)
    else:
        # Fallback: 标准逐层 forward
        for layer in model.layers:
            x = layer(x)

    x = model.ln_f(x)
    logits = model.lm_head(x)

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=config.pad_token_id,
        )

    return logits, loss
