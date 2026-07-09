"""
Recurrent Depth（深度循环推理）— ICLR/NeurIPS 2026 主流方向

核心思想:
    标准Transformer: 堆叠N层不同的参数 → O(N)参数，固定深度
    Recurrent Depth: 同一组参数循环N次 → O(1)参数，动态深度

论文来源:
    - "Scaling up Test-Time Compute with Latent Reasoning" (NeurIPS 2025)
    - "Universal YOCO for Efficient Depth Scaling" (arXiv 2026.04)
    - "Depth-Recurrent Attention Mixtures (Dreamer)" (arXiv 2026)

为什么重要:
    1. 参数效率极高: 1层参数 = N层效果（共享权重循环N次）
    2. 测试时可动态调整深度: 简单问题浅层，难问题深层
    3. 天然支持KV-cache共享: 循环层复用同一套cache
    4. 和 speculative decoding 天然兼容

PyTorch 实现核心:
    标准:  for layer in layers: x = layer(x)           # 6 层不同参数
    循环:  for _ in range(depth): x = shared_block(x)  # 1 层参数, 跑 6 次

增强: 同时支持不同深度的输出做加权平均（depth attention）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentDepthBlock(nn.Module):
    """
    可循环的 Decoder Block — 同一组参数跑多次

    这是 Recurrent Depth 的核心：把"堆叠"变成"循环"

    标准 GPT:  layers = [Block1, Block2, Block3, Block4, Block5, Block6]
    Recurrent: block = Block()  # 只有 1 组参数
               for _ in range(6): x = block(x)  # 循环 6 次
    """

    def __init__(self, attention_block, ffn_block):
        """
        Args:
            attention_block: CausalSelfAttention 实例
            ffn_block: MLP 实例
        """
        super().__init__()
        self.ln_1 = attention_block.ln_1  # 复用原 attention 的 LayerNorm
        self.attn = attention_block
        self.ln_2 = ffn_block.c_fc  # 用一个独立的 LN for FFN
        self.mlp = ffn_block

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AdaptiveRecurrentDepth(nn.Module):
    """
    自适应深度循环 — 每一步的输出可以提前退出（adaptive depth）

    思路:
        不是固定循环 N 次，而是每一步检查一个 "halt" 信号
        如果当前步的输出和上一步的输出足够接近 → 提前退出
        这样可以动态节省计算

    简化实现:
        不使用 halt 信号，而是收集每一步的输出
        用 learned weight 做加权平均（depth attention）
        类似于 "depth ensembling"
    """

    def __init__(self, block, max_depth=6):
        super().__init__()
        self.block = block
        self.max_depth = max_depth

        # 每一步的权重（可学习），初始化为均匀分布
        self.depth_weights = nn.Parameter(torch.ones(max_depth) / max_depth)

    def forward(self, x):
        """
        Returns:
            output: (B, T, D) 各深度的加权平均
            all_depths: list of (B, T, D) 每个深度的输出
        """
        all_outputs = []
        for d in range(self.max_depth):
            x = self.block(x)
            all_outputs.append(x)

        # 加权平均
        weights = F.softmax(self.depth_weights, dim=0)  # (max_depth,)
        stacked = torch.stack(all_outputs, dim=0)        # (max_depth, B, T, D)

        # 每个 (B, T) 位置用相同的深度权重
        output = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)  # (B, T, D)

        return output, all_outputs


class RecurrentGPT(nn.Module):
    """
    Recurrent Depth GPT 模型

    不同于堆叠 N 层不同的 DecoderBlock,
    这里只创建 1 个 DecoderBlock, 循环 N 次

    配置:
        n_embd, n_head, vocab_size, block_size: 和标准 TinyGPT 一样
        recurrence_depth: 循环次数（相当于标准模型的层数）
    """

    def __init__(self, config, recurrence_depth=6):
        super().__init__()
        self.config = config
        self.recurrence_depth = recurrence_depth

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # 只有 1 个 DecoderBlock，循环 N 次
        from src.models.tiny_gpt import CausalSelfAttention, MLP

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        # Final
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        print(f"[RecurrentGPT] depth={recurrence_depth}, params={self.get_num_params():,}")
        print(f"[RecurrentGPT] vs 标准GPT: 参数少 {(1 - self.get_num_params() / model_params_standard(config)) * 100:.0f}%")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.wte(input_ids) + self.wpe(positions))

        # 循环 N 次（替代 N 层堆叠）
        for _ in range(self.recurrence_depth):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.config.block_size:]
            logits, _ = self.forward(input_cond)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


def model_params_standard(config):
    """标准 TinyGPT 的参数量"""
    d = config.n_embd
    n_layers = getattr(config, 'n_layer', 6)
    emb = config.vocab_size * d + config.block_size * d
    attn_per = 4 * d * d
    ffn_per = 2 * d * d * config.ffn_ratio
    ln_per = 4 * d
    per_layer = attn_per + ffn_per + ln_per
    final = 2 * d + d * config.vocab_size
    return emb + n_layers * per_layer + final


def create_recurrent_model(config, depth=6):
    """创建 RecurrentGPT"""
    return RecurrentGPT(config, recurrence_depth=depth)
