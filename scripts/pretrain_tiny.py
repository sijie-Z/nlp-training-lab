"""
TinyGPT 预训练脚本
从随机初始化开始，用领域语料训练一个微型 GPT 模型
观察 loss 如何从 ~9（随机猜测）下降到 ~2-3（学到领域知识）

用法:
    # 默认训练
    python scripts/pretrain_tiny.py

    # 自定义参数
    python scripts/pretrain_tiny.py --preset small --epochs 50 --lr 1e-3

    # 从 checkpoint 恢复
    python scripts/pretrain_tiny.py --resume outputs/checkpoints/pretrain_tiny/checkpoint.pt
"""
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tiny_gpt import TinyGPT, TinyGPTConfig, create_model
from tokenizers import Tokenizer


# ============================================================
# 数据准备
# ============================================================
class PretrainDataset(Dataset):
    """预训练数据集：滑动窗口生成固定长度的序列"""

    def __init__(self, text_path, tokenizer_path, block_size=256, stride=None):
        self.block_size = block_size
        self.stride = stride or block_size

        # 加载 tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]") or 0

        # 加载并 tokenize 文本
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize 全文
        print(f"[Dataset] Tokenizing {len(text):,} chars...")
        t0 = time.time()
        encoding = self.tokenizer.encode(text)
        self.ids = encoding.ids
        elapsed = time.time() - t0
        print(f"[Dataset] {len(self.ids):,} tokens, 耗时 {elapsed:.1f}s, "
              f"压缩比: {len(self.ids)/len(text):.1f}x")

        # 计算样本数
        self.num_samples = max(0, (len(self.ids) - block_size) // self.stride + 1)
        print(f"[Dataset] 生成 {self.num_samples} 个训练样本 (block_size={block_size})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size

        x = torch.tensor(self.ids[start:end], dtype=torch.long)
        y = torch.tensor(self.ids[start:end], dtype=torch.long)  # 自回归：labels = inputs

        return x, y


class DataCollator:
    """Batch 填充到相同长度（用于变长数据）"""

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch = list of (x, y) tuples
        max_len = max(x.size(0) for x, y in batch)

        padded_x = []
        padded_y = []
        for x, y in batch:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.cat([x, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                y = torch.cat([y, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            padded_x.append(x)
            padded_y.append(y)

        return torch.stack(padded_x), torch.stack(padded_y)


# ============================================================
# 训练
# ============================================================
def get_lr_schedule(optimizer, warmup_steps, total_steps, base_lr):
    """带 warmup 的余弦退火学习率调度"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval=10):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # 前向
        logits, loss = model(x, labels=y)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止 loss spike）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 统计
        batch_tokens = (y != model.config.pad_token_id).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if step % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            ppl = math.exp(loss.item())
            print(f"  Epoch {epoch:3d} | Step {step:4d}/{len(dataloader):4d} | "
                  f"Loss {loss.item():.4f} | PPL {ppl:.1f} | LR {lr:.2e} | "
                  f"Time {elapsed:.0f}s")

    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


def train(config_dict):
    """主训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 设备: {device}")

    # 路径
    corpus_path = config_dict.get("corpus_path",
        str(PROJECT_ROOT / "data/raw/domain_corpus_cleaned.txt"))
    tokenizer_path = config_dict.get("tokenizer_path",
        str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json"))
    output_dir = config_dict.get("output_dir",
        str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny"))
    os.makedirs(output_dir, exist_ok=True)

    # 模型
    model_preset = config_dict.get("model_preset", "small")
    block_size = config_dict.get("block_size", 256)
    vocab_size = config_dict.get("vocab_size", 8000)
    model, model_config = create_model(preset=model_preset, vocab_size=vocab_size, block_size=block_size)
    model = model.to(device)
    print(f"[Train] 模型参数量: {model.get_num_params():,}")

    # 数据
    batch_size = config_dict.get("batch_size", 8)
    stride = config_dict.get("stride", 128)
    dataset = PretrainDataset(
        corpus_path, tokenizer_path,
        block_size=block_size, stride=stride,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=DataCollator(pad_token_id=model_config.pad_token_id),
        pin_memory=(device.type == "cuda"),
    )

    # 优化器
    learning_rate = config_dict.get("learning_rate", 5e-4)
    weight_decay = config_dict.get("weight_decay", 0.1)
    betas = config_dict.get("betas", (0.9, 0.95))
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=weight_decay, betas=betas,
    )

    # 学习率调度
    epochs = config_dict.get("epochs", 30)
    warmup_ratio = config_dict.get("warmup_ratio", 0.1)
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps, learning_rate)

    # 混合精度（GPU 上用）
    use_amp = config_dict.get("use_amp", device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 恢复训练
    start_epoch = 0
    resume_path = config_dict.get("resume")
    if resume_path and os.path.exists(resume_path):
        print(f"[Train] 从 checkpoint 恢复: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Train] 恢复 epoch {start_epoch}")

    # ============================================================
    # 训练循环
    # ============================================================
    print(f"\n[Train] 开始训练: {epochs} epochs, {len(dataloader)} steps/epoch")
    print(f"[Train] 总训练 tokens: {dataset.num_samples * block_size:,}")
    print(f"[Train] LR: {learning_rate}, Batch: {batch_size}, Warmup: {warmup_steps} steps")
    print("=" * 70)

    history = []
    best_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        avg_loss, avg_ppl = train_epoch(
            model, dataloader, optimizer, scheduler, device, epoch,
            log_interval=config_dict.get("log_interval", 20),
        )

        epoch_time = time.time() - epoch_start
        print(f"--- Epoch {epoch:3d} 完成 | Avg Loss: {avg_loss:.4f} | "
              f"PPL: {avg_ppl:.1f} | 耗时: {epoch_time:.0f}s ---")

        history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "ppl": round(avg_ppl, 1),
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(epoch_time, 1),
        })

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save({"model": model.state_dict(), "config": model_config, "epoch": epoch}, ckpt_path)
            print(f"  → 最佳模型: {ckpt_path} (loss={avg_loss:.4f})")

        # 定期保存
        if epoch % config_dict.get("save_every", 10) == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_config,
                "epoch": epoch,
            }, ckpt_path)

    # ============================================================
    # 训练后总结
    # ============================================================
    print("\n" + "=" * 70)
    print("[Train] 训练完成！")
    print(f"[Train] 初始 PPL: {history[0]['ppl']:.0f}" if history else "N/A")
    print(f"[Train] 最终 PPL: {history[-1]['ppl']:.1f}" if history else "N/A")
    print(f"[Train] 最佳 Loss: {best_loss:.4f}")

    # 保存训练历史
    history_path = os.path.join(output_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[Train] 训练历史: {history_path}")

    return model, history


# ============================================================
# 生成测试
# ============================================================
def generate_sample(model, tokenizer, prompt="GIS是", max_tokens=30, device="cpu"):
    """用训练好的模型生成文本"""
    model.eval()
    model = model.to(device)

    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=40)

    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text


def run_generation_demo(checkpoint_path, tokenizer_path, device="cpu"):
    """加载 checkpoint 并跑几个生成示例"""
    print("\n" + "=" * 70)
    print("[Generate] 生成演示")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = TinyGPT(config)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    prompts = [
        "GIS",
        "遥感技术",
        "坐标系",
        "QGIS中",
        "NDVI",
    ]

    for prompt in prompts:
        output = generate_sample(model, tokenizer, prompt=prompt, max_tokens=40, device=device)
        print(f"\n  Prompt: {prompt}")
        print(f"  生成:   {output}")


# ============================================================
# 对比：随机初始化 vs 训练后
# ============================================================
def compare_before_after(checkpoint_path, tokenizer_path, device="cpu"):
    """对比随机初始化模型和训练后模型的输出"""
    print("\n" + "=" * 70)
    print("[Compare] 预训练前后对比")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    # 随机初始化模型
    random_model = TinyGPT(config).to(device)

    # 训练后模型
    trained_model = TinyGPT(config).to(device)
    trained_model.load_state_dict(ckpt["model"])

    prompts = ["什么是", "地理信息", "卫星遥感"]

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")
        r_out = generate_sample(random_model, tokenizer, prompt=prompt, max_tokens=20, device=device)
        t_out = generate_sample(trained_model, tokenizer, prompt=prompt, max_tokens=20, device=device)
        print(f"  随机模型: {r_out}")
        print(f"  训练模型: {t_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyGPT 预训练")
    parser.add_argument("--preset", type=str, default="small",
                       choices=["tiny", "small", "medium"],
                       help="模型大小预设")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--block_size", type=int, default=256, help="最大序列长度")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练 checkpoint")
    parser.add_argument("--generate", action="store_true", help="训练后生成演示")
    parser.add_argument("--compare", action="store_true", help="对比随机 vs 训练后")
    args = parser.parse_args()

    config = {
        "model_preset": args.preset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "block_size": args.block_size,
        "resume": args.resume,
        "log_interval": 10,
        "save_every": 10,
    }

    model, history = train(config)

    # 训练后演示
    if args.generate or args.compare:
        tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")
        ckpt_path = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/best_model.pt")

        if os.path.exists(ckpt_path):
            if args.generate:
                run_generation_demo(ckpt_path, tokenizer_path, device="cpu")
            if args.compare:
                compare_before_after(ckpt_path, tokenizer_path, device="cpu")
        else:
            print("[Warn] 没有找到最佳模型 checkpoint")
