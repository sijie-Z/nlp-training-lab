"""
TinyGPT —— 从零实现的 GPT-2 风格微型 Transformer
用于理解预训练的核心：Decoder-only 因果语言模型

架构：
  Token Embedding + Position Embedding
  → N 层 Decoder Block (Causal Self-Attention + FFN)
  → LayerNorm → LM Head → 输出下一个 token 的概率分布

参考：nanoGPT / GPT-2 / TinyLlama
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TinyGPTConfig:
    """TinyGPT 超参数配置"""
    # 词表
    vocab_size: int = 8000
    # 模型维度
    n_embd: int = 384          # embedding 维度
    n_head: int = 6            # 注意力头数
    n_layer: int = 6           # Transformer 层数
    # 序列长度
    block_size: int = 256      # 最大上下文长度
    # FFN 扩展比例
    ffn_ratio: int = 4         # FFN 隐藏层 = n_embd * ffn_ratio
    # 正则化
    dropout: float = 0.1
    # 特殊 token
    pad_token_id: int = 0
    # RoPE (等会再说，先不加)
    use_rotary: bool = False

    @property
    def n_params(self):
        """估算参数量"""
        # Embedding
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        # Per layer: attention (QKV + output proj) + FFN (2 layers) + 2 LayerNorms
        d = self.n_embd
        attn = 4 * d * d                    # Q, K, V, O projections
        ffn = 2 * d * d * self.ffn_ratio    # FFN up + down
        ln = 4 * d                          # 2 LayerNorms × 2 params each (weight + bias)
        per_layer = attn + ffn + ln
        # Final LayerNorm + LM Head
        final = 2 * d + d * self.vocab_size
        total = emb + self.n_layer * per_layer + final
        return total

    @classmethod
    def from_preset(cls, preset: str):
        """预设配置"""
        presets = {
            "tiny": dict(n_embd=192, n_head=4, n_layer=4, vocab_size=8000),
            "small": dict(n_embd=384, n_head=6, n_layer=6, vocab_size=8000),
            "medium": dict(n_embd=512, n_head=8, n_layer=8, vocab_size=8000),
        }
        if preset in presets:
            return cls(**presets[preset])
        raise ValueError(f"Unknown preset: {preset}")


class CausalSelfAttention(nn.Module):
    """因果自注意力（Decoder-only）"""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V 一起做
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # 因果 mask（上三角为 -inf）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd

        # QKV 投影
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 拆成多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Causal mask
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 加权求和
        attn_output = attn_weights @ v  # (B, nh, T, hd)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        output = self.c_proj(attn_output)
        output = F.dropout(output, p=self.dropout, training=self.training)

        return output


class MLP(nn.Module):
    """前馈网络（GPT-2 风格的 GELU）"""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffn_ratio, bias=False)
        self.c_proj = nn.Linear(config.n_embd * config.ffn_ratio, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")  # GPT-2 用 tanh 近似的 GELU
        x = self.c_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DecoderBlock(nn.Module):
    """一个 Transformer Decoder Block"""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm: LayerNorm → Attention → Residual
        x = x + self.attn(self.ln_1(x))
        # Pre-norm: LayerNorm → MLP → Residual
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """
    微型 GPT 模型，用于理解预训练过程。

    用法:
        config = TinyGPTConfig.from_preset("small")
        model = TinyGPT(config)
        logits, loss = model(input_ids, labels=input_ids)
    """

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config

        # Token + Position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layer)
        ])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM Head（权重与 wte 共享，节省参数）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重绑定
        self.lm_head.weight = self.wte.weight

        # 参数初始化
        self.apply(self._init_weights)

        print(f"[TinyGPT] 模型参数: {self.get_num_params():,}")
        print(f"[TinyGPT] 配置: n_embd={config.n_embd}, "
              f"n_head={config.n_head}, n_layer={config.n_layer}, "
              f"block_size={config.block_size}, vocab={config.vocab_size}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=False):
        """参数统计"""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.wpe.weight.numel()
            total -= self.wte.weight.numel()
        return total

    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        Args:
            input_ids: (B, T) token ids
            labels: (B, T) 用于计算 loss，通常 = input_ids（自回归）
            attention_mask: (B, T) padding mask（可选）

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar（如果提供了 labels）
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"

        # 位置编码
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Token + Position embedding
        tok_emb = self.wte(input_ids)  # (B, T, C)
        pos_emb = self.wpe(positions)  # (1, T, C)
        x = self.drop(tok_emb + pos_emb)

        # 通过所有 Decoder Block
        for layer in self.layers:
            x = layer(x)

        # 最终 LayerNorm
        x = self.ln_f(x)

        # LM Head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss 计算
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
        """
        自回归生成
        Args:
            input_ids: (1, T) 初始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度（>1 更随机，<1 更确定）
            top_k: top-k 采样（None = 贪婪）
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 block_size
            input_cond = input_ids[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self.forward(input_cond)  # (1, T, vocab)
            logits = logits[:, -1, :]  # 取最后一个位置的 logit

            # 温度缩放
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax → 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def estimate_loss(self, data_loader, eval_iters=10):
        """在 data_loader 上估算 loss（不更新梯度）"""
        self.eval()
        losses = []
        for i, (x, y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            x = x.to(next(self.parameters()).device)
            y = y.to(next(self.parameters()).device)
            with torch.no_grad():
                _, loss = self.forward(x, labels=y)
            losses.append(loss.item())
        self.train()
        return sum(losses) / len(losses)


def create_model(preset="small", vocab_size=None, block_size=None):
    """工厂函数：创建 TinyGPT 模型"""
    config = TinyGPTConfig.from_preset(preset)

    if vocab_size is not None:
        config.vocab_size = vocab_size
    if block_size is not None:
        config.block_size = block_size

    model = TinyGPT(config)
    return model, config


if __name__ == "__main__":
    # 快速测试
    print("TinyGPT 预设参数量:")
    for preset in ["tiny", "small", "medium"]:
        config = TinyGPTConfig.from_preset(preset)
        print(f"  {preset}: {config.n_params:,} params "
              f"(n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer})")

    # 创建模型
    model, config = create_model("small")
    print(f"\n实际参数量: {model.get_num_params():,}")

    # 前向测试
    B, T = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(input_ids, labels=input_ids)
    print(f"输入: {input_ids.shape} → 输出: {logits.shape}, Loss: {loss.item():.4f}")

    # 生成测试
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"生成: {prompt.shape[1]} tokens → {generated.shape[1]} tokens")
