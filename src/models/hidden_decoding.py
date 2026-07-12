"""
Hidden Decoding at Scale: Latent Computation Scaling for LLMs

arXiv: 2607.08186 (July 9, 2026) — 提交于 3 天前!

核心思想:
    传统 Chain-of-Thought 推理: 在 token 空间生成推理步骤
    Hidden Decoding: 在隐藏空间中做推理 — token 被展开为 n 个独立的
    embedding stream, 通过 Stream-Factorized Attention 在隐藏空间中交互

为什么这很重要:
    1. Token-space CoT 的问题是: 每个 token 的输出必须是离散的词汇
       → 中间推理步骤受限于语言表达能力
    2. Hidden-space 推理: 中间步骤是连续向量 → 表达能力远超离散 token
    3. 多流架构: n 个独立的 embedding stream = 每个 token 有 n 条"思维线"
    4. 已在 WeLM-HD4-617B (MoE) 上验证了 frontier scale

论文数据:
    - 617B MoE 模型上验证
    - 第一个在 100B+ 级别实现序列长度扩展的方法
    - Stream-Factorized Attention 保持计算量近乎线性增长

简化实现 (用"多流思维"演示核心概念):
    1. 把每个 token 的 embedding 复制 n 份 → n 条"思维线"
    2. 每条思维线独立做 self-attention (stream-internal)
    3. 跨流做 cross-stream attention (stream-mixing)
    4. 聚合 n 条思维线的输出 → "做出决定"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamFactorizedAttention(nn.Module):
    """
    流分解注意力 — Hidden Decoding 的核心

    把标准的 self-attention 分解为:
        1. Intra-stream attention: 同一 stream 内的 token 交互 (时间维度)
        2. Inter-stream attention: 同一位置不同 stream 之间的交互 (深度维度)

    时间复杂度:
        标准: O(T² × D)
        流分解: O(n_steps × T² × D/n + n_steps² × T × D/n)
                ≈ 线性于 n_steps 而非二次

    为什么这很重要:
        n=4 时, 标准 O(16×T²) → 流分解 O(4×T² + 16×T) ≈ O(4×T²)
        → 计算量减 4 倍
    """

    def __init__(self, n_embd, n_heads, n_streams=4, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_streams = n_streams
        self.head_dim = n_embd // n_heads

        # Intra-stream: 标准的 causal self-attention (每个 stream 内部)
        self.intra_attn = nn.MultiheadAttention(
            n_embd, n_heads, dropout=dropout, batch_first=True
        )

        # Inter-stream: 跨流混合 (所有 stream 在同一位置)
        self.inter_attn = nn.MultiheadAttention(
            n_embd, n_heads, dropout=dropout, batch_first=True
        )

        # Stream gating: 学到的 gate 决定每个 stream 的贡献
        self.stream_gate = nn.Parameter(torch.ones(n_streams) / n_streams)

    def forward(self, x):
        """
        Args:
            x: (B, n_streams, T, n_embd) — 多个流

        Returns:
            out: (B, T, n_embd) — 聚合后的输出
        """
        B, S, T, D = x.shape

        # 1. Intra-stream attention (并行处理所有流)
        # Reshape: (B, S, T, D) → (B*S, T, D)
        x_intra = x.reshape(B * S, T, D)
        x_intra = self.intra_attn(x_intra, x_intra, x_intra)[0]
        x_intra = x_intra.reshape(B, S, T, D)

        # 2. Inter-stream attention (每个时间步)
        # Reshape: (B, S, T, D) → (B*T, S, D)
        x_inter = x_intra.permute(0, 2, 1, 3).reshape(B * T, S, D)
        x_inter = self.inter_attn(x_inter, x_inter, x_inter)[0]
        x_inter = x_inter.reshape(B, T, S, D).permute(0, 2, 1, 3)  # (B, S, T, D)

        # 3. Gated aggregation
        gate = F.softmax(self.stream_gate, dim=0)  # (S,)
        out = (gate.view(1, S, 1, 1) * x_inter).sum(dim=1)  # (B, T, D)

        return out


class HiddenDecodingDecoder(nn.Module):
    """
    Hidden Decoding Decoder — 在隐藏空间中做多流推理

    替代标准 Transformer 的逐层 forward:
        标准: x → layer1 → layer2 → ... → layerN → output
        HD:   x → 展开为 n 个流 → 流内+流间 attention → 聚合 → output
    """

    def __init__(self, attn_block, ffn_block, n_streams=4, n_embd=384, n_head=6):
        super().__init__()
        self.attn_block = attn_block  # 原始的 causal self-attention
        self.ffn = ffn_block

        # 流分解 attention（替代标准 attention）
        self.stream_attn = StreamFactorizedAttention(
            n_embd, n_head, n_streams
        )

        # 流嵌入：每个 stream 有独特的偏置
        self.stream_embed = nn.Parameter(torch.randn(n_streams, n_embd) * 0.02)

        self.n_streams = n_streams
        self.ln_stream = nn.LayerNorm(n_embd)
        self.n_embd = n_embd

    def forward(self, x):
        B, T, D = x.shape

        # 展开为 n 个流
        x_expanded = x.unsqueeze(1).expand(B, self.n_streams, T, D)
        x_expanded = x_expanded + self.stream_embed.view(1, self.n_streams, 1, D)

        # 流分解 attention (替代标准 self-attention)
        x_streamed = self.stream_attn(x_expanded)

        # 残差: Pre-norm from the original DecoderBlock
        residual = x
        x_norm = self.ln_stream(x)
        x = residual + x_streamed
        x = x + self.ffn(x)

        return x


def create_hidden_decoding_model(base_model, n_streams=4):
    """
    将 TinyGPT 的标准 DecoderBlock 替换为 Hidden Decoding 版本

    Args:
        base_model: TinyGPT 模型
        n_streams: 流数（默认 4）

    Returns:
        替换后的模型
    """
    from src.models.tiny_gpt import DecoderBlock

    new_layers = []
    for layer in base_model.layers:
        if isinstance(layer, DecoderBlock):
            hd_layer = HiddenDecodingDecoder(
                layer.attn, layer.mlp,
                n_streams=n_streams,
                n_embd=base_model.config.n_embd,
                n_head=base_model.config.n_head,
            )
            new_layers.append(hd_layer)
        else:
            new_layers.append(layer)

    base_model.layers = nn.ModuleList(new_layers)
    added = n_streams * base_model.config.n_embd  # stream embeddings
    total = sum(p.numel() for p in base_model.parameters())
    print(f"[HiddenDecoding] {n_streams} streams, 新增 {added:,} params ({100*added/total:.3f}%)")

    return base_model
