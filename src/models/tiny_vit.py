"""
轻量级 Vision Transformer (ViT) — 用于多模态对比学习的图像编码器

设计原则：
  - 极小参数量（~2M），在 4GB 显存上能跑
  - Patch embedding + Position embedding + Transformer Encoder
  - 输出全局图像表示（CLS token）

架构：
  Image (3, 64, 64)
    → Patch Embedding: 8x8 patches → 64 patches × 192 dims
    → CLS token + Position Embedding
    → N 层 Transformer Encoder (Pre-norm, Self-Attention, FFN)
    → CLS token output → Linear → projection_dim
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TinyViTConfig:
    """TinyViT 配置"""
    image_size: int = 64         # 输入图像尺寸
    patch_size: int = 8          # Patch 大小
    in_channels: int = 3         # RGB
    n_embd: int = 192            # 隐藏维度
    n_head: int = 4              # 注意力头数
    n_layer: int = 4             # Transformer 层数
    ffn_ratio: int = 4           # FFN 扩展比例
    dropout: float = 0.1
    projection_dim: int = 384    # 最终投影维度（对齐文本编码器的 n_embd）

    @property
    def num_patches(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def n_params(self):
        # Embedding: patch_proj + cls_token + pos_embed
        emb = (self.patch_size ** 2 * self.in_channels * self.n_embd
               + self.n_embd
               + (self.num_patches + 1) * self.n_embd)
        # Per layer: attn + ffn + 2LN
        d = self.n_embd
        attn = 4 * d * d
        ffn = 2 * d * d * self.ffn_ratio
        ln = 4 * d
        per_layer = attn + ffn + ln
        # Final LN + projection
        final = 2 * d + d * self.projection_dim
        total = emb + self.n_layer * per_layer + final
        return total


class PatchEmbedding(nn.Module):
    """将图像切分为 patches 并做线性投影"""

    def __init__(self, config: TinyViTConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.in_channels, config.n_embd,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        # patches: (B, n_embd, H/P, W/P)

    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = self.proj(x)                         # (B, n_embd, 8, 8)
        x = x.flatten(2).transpose(1, 2)         # (B, 64, n_embd)
        return x


class ViTEncoderBlock(nn.Module):
    """Transformer Encoder Block (Pre-norm)"""

    def __init__(self, config: TinyViTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            config.n_embd, config.n_head,
            dropout=config.dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * config.ffn_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd * config.ffn_ratio, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # Pre-norm + Attention + Residual
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]
        # Pre-norm + MLP + Residual
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyViT(nn.Module):
    """
    轻量级 ViT 图像编码器

    用法:
        config = TinyViTConfig()
        model = TinyViT(config)
        features = model.encode_image(images)  # (B, projection_dim)
    """

    def __init__(self, config: TinyViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # CLS token + Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.n_embd)
        )

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_post = nn.LayerNorm(config.n_embd)

        # 投影头：使图像特征对齐到文本空间
        self.projection = nn.Sequential(
            nn.Linear(config.n_embd, config.projection_dim, bias=False),
            nn.LayerNorm(config.projection_dim),
        )

        self._init_weights()
        print(f"[TinyViT] 参数量: {self.get_num_params():,}")
        print(f"[TinyViT] 输入: ({config.in_channels}, {config.image_size}, {config.image_size})")
        print(f"[TinyViT] Patches: {config.num_patches}, n_embd: {config.n_embd}, 层数: {config.n_layer}")

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 图像 batch

        Returns:
            image_features: (B, projection_dim)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, n_embd)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, n_embd)

        # Position embedding
        x = x + self.pos_embed
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # CLS token output
        x = self.ln_post(x[:, 0, :])  # (B, n_embd)

        # Project to shared space
        x = self.projection(x)  # (B, projection_dim)

        return x


class ImageTextCLIP(nn.Module):
    """
    CLIP 风格的双塔模型

    图像编码器 (TinyViT) + 文本编码器 (TinyGPT)

    训练目标：让匹配的图文对在共享空间中有更高的余弦相似度。

    Loss = (image2text_contrastive + text2image_contrastive) / 2
    即 InfoNCE loss 的对称版本
    """

    def __init__(self, image_encoder, text_encoder, logit_scale=2.6592):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 可学习的温度参数（CLIP 用 exp(logit_scale) 作为温度）
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

        print(f"[CLIP] 图像编码器: {image_encoder.get_num_params():,} params")
        print(f"[CLIP] 文本编码器: ~13.8M params (TinyGPT)")
        print(f"[CLIP] logit_scale: {logit_scale:.4f}")

    def encode_image(self, images):
        """返回归一化的图像特征"""
        features = self.image_encoder(images)  # (B, dim)
        return F.normalize(features, dim=-1)

    def encode_text(self, text_tokens):
        """
        用 TinyGPT 编码文本

        Args:
            text_tokens: (B, T) token ids

        Returns:
            normalized_text_features: (B, projection_dim)
        """
        B, T = text_tokens.shape

        # 通过 TinyGPT 取最后一个 token 的 hidden state
        model = self.text_encoder
        device = text_tokens.device

        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = model.wte(text_tokens)
        pos_emb = model.wpe(positions)
        x = model.drop(tok_emb + pos_emb)

        for layer in model.layers:
            x = layer(x)

        x = model.ln_f(x)  # (B, T, n_embd)

        # 取最后一个 token 的输出作为文本表示
        text_features = x[:, -1, :]  # (B, n_embd)

        # TinyGPT 的 n_embd(384) 和 ViT 的 projection_dim(384) 一致
        # 直接 normalize
        return F.normalize(text_features, dim=-1)

    def forward(self, images, text_tokens):
        """
        Args:
            images: (B, 3, 64, 64)
            text_tokens: (B, T)

        Returns:
            logits_per_image: (B, B) 图像到文本的相似度矩阵
            logits_per_text: (B, B) 文本到图像的相似度矩阵
        """
        image_features = self.encode_image(images)       # (B, dim)
        text_features = self.encode_text(text_tokens)     # (B, dim)

        # 缩放余弦相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


def clip_loss(logits_per_image, logits_per_text):
    """
    CLIP 对比学习 loss (对称 InfoNCE)

    每张图片的正样本是对应的文本（在对角线上），所有其他文本都是负样本。反之亦然。
    """
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)  # 图像 → 文本
    loss_t = F.cross_entropy(logits_per_text, labels)   # 文本 → 图像

    loss = (loss_i + loss_t) / 2

    with torch.no_grad():
        acc_i = (logits_per_image.argmax(dim=-1) == labels).float().mean()
        acc_t = (logits_per_text.argmax(dim=-1) == labels).float().mean()

    return loss, acc_i, acc_t


def create_clip_model(vit_config=None, tiny_gpt_checkpoint=None, device="cpu"):
    """工厂函数：创建 CLIP 双塔模型"""
    if vit_config is None:
        vit_config = TinyViTConfig()

    image_encoder = TinyViT(vit_config)

    # 加载 TinyGPT 文本编码器
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.tiny_gpt import TinyGPT, TinyGPTConfig

    if tiny_gpt_checkpoint is None:
        tiny_gpt_checkpoint = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch020.pt")

    ckpt = torch.load(tiny_gpt_checkpoint, map_location="cpu", weights_only=False)
    gpt_config = ckpt["config"]
    text_encoder = TinyGPT(gpt_config)
    text_encoder.load_state_dict(ckpt["model"])

    # 冻结文本编码器
    for p in text_encoder.parameters():
        p.requires_grad = False

    model = ImageTextCLIP(image_encoder, text_encoder)
    return model, vit_config
