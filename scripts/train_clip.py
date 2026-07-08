"""
CLIP 风格多模态对比学习训练脚本

训练图-文对齐模型：
  - 图像编码器：TinyViT (~1.6M params)
  - 文本编码器：TinyGPT (~13.8M params, 冻结)
  - 数据集：EuroSAT (自动下载) — 10 类遥感场景

用法:
    python scripts/train_clip.py --epochs 10 --batch_size 32
"""
import os
import sys
import json
import time
import argparse
import zipfile
import io
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image

from src.models.tiny_vit import (
    TinyViTConfig, TinyViT, ImageTextCLIP, clip_loss, create_clip_model
)
from tokenizers import Tokenizer


# ============================================================
# EuroSAT 数据集
# ============================================================
EUROSAT_CLASSES = [
    "AnnualCrop",       # 一年生作物
    "Forest",            # 森林
    "HerbaceousVegetation",  # 草本植被
    "Highway",           # 高速公路
    "Industrial",        # 工业区
    "Pasture",           # 牧场
    "PermanentCrop",     # 多年生作物
    "Residential",       # 居民区
    "River",             # 河流
    "SeaLake",           # 海洋/湖泊
]

# 每个类别的中文描述（作为训练时的文本 prompt）
EUROSAT_CAPTIONS = {
    "AnnualCrop": "一年生作物的卫星遥感影像展示了农田的生长周期和耕作模式",
    "Forest": "森林区域的卫星遥感影像呈现了密集的树木覆盖和自然生态系统",
    "HerbaceousVegetation": "草本植被的卫星遥感影像显示了草地和低矮植物的分布",
    "Highway": "高速公路的卫星遥感影像展示了交通基础设施和道路网络",
    "Industrial": "工业区的卫星遥感影像显示了工厂建筑和工业设施",
    "Pasture": "牧场的卫星遥感影像展示了用于放牧的草地和开阔区域",
    "PermanentCrop": "多年生作物的卫星遥感影像显示了果园或葡萄园等长期种植区域",
    "Residential": "居民区的卫星遥感影像展示了住宅建筑和城市社区布局",
    "River": "河流的卫星遥感影像显示了自然水道的形态和分布",
    "SeaLake": "海洋和湖泊的卫星遥感影像展示了大型水体的特征和边界",
}

EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"


def download_eurosat(data_dir):
    """下载 EuroSAT 数据集（~90MB）"""
    data_dir = Path(data_dir)
    zip_path = data_dir / "EuroSAT_RGB.zip"
    extract_path = data_dir / "EuroSAT_RGB"

    if extract_path.exists():
        n_images = len(list(extract_path.glob("*/*.jpg")))
        print(f"[EuroSAT] 数据集已存在: {extract_path} ({n_images} 张图片)")
        return str(extract_path)

    print(f"[EuroSAT] 下载数据集 (~90MB)...")
    print(f"[EuroSAT] URL: {EUROSAT_URL}")
    print(f"[EuroSAT] 存放路径: {zip_path}")

    try:
        urllib.request.urlretrieve(EUROSAT_URL, zip_path)
    except Exception as e:
        print(f"[EuroSAT] 下载失败: {e}")
        print(f"[EuroSAT] 改用 synthetic 数据集（随机图像 + 标签）用于演示链路")

        # 创建 synthetic 数据集用于验证代码链路
        os.makedirs(extract_path, exist_ok=True)
        for cls_name in EUROSAT_CLASSES:
            cls_dir = extract_path / cls_name
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(100):  # 每类 100 张
                img = Image.fromarray(
                    (torch.rand(3, 64, 64) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                )
                img.save(cls_dir / f"synthetic_{i:05d}.jpg")
        n_total = len(list(extract_path.glob("*/*.jpg")))
        print(f"[EuroSAT] 生成 synthetic 数据集: {n_total} 张图片 (仅用于链路验证)")
        return str(extract_path)

    # 解压
    print(f"[EuroSAT] 解压中...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)

    n_images = len(list(extract_path.glob("*/*.jpg")))
    print(f"[EuroSAT] 下载完成: {n_images} 张图片")
    return str(extract_path)


class EuroSATDataset(Dataset):
    """EuroSAT 遥感图像数据集 + 中文文本描述"""

    def __init__(self, root_dir, tokenizer, split="train", train_ratio=0.8, max_length=128):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 收集所有图片路径和对应的类别
        self.samples = []
        for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            images = list(cls_dir.glob("*.jpg"))
            # 训练/验证分割
            n_train = int(len(images) * train_ratio)
            if split == "train":
                images = images[:n_train]
            else:
                images = images[n_train:]

            for img_path in images:
                self.samples.append((str(img_path), cls_idx, cls_name))

        # 图像预处理
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

        self.class_names = EUROSAT_CLASSES
        self.captions = EUROSAT_CAPTIONS

        print(f"[EuroSAT] {split}: {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_idx, cls_name = self.samples[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 构造文本描述（增强：训练时随机选择不同的描述方式）
        caption = self.captions[cls_name]

        # Tokenize
        enc = self.tokenizer.encode(caption)
        tokens = enc.ids[:self.max_length]

        return {
            "image": image,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "label": cls_idx,
            "class_name": cls_name,
        }


def collate_clip(batch, pad_token_id=0):
    """Collate CLIP batch"""
    images = torch.stack([b["image"] for b in batch])

    max_len = max(len(b["tokens"]) for b in batch)
    padded_tokens = []
    for b in batch:
        pad_len = max_len - len(b["tokens"])
        padded = torch.cat([b["tokens"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_tokens.append(padded)

    return {
        "images": images,
        "tokens": torch.stack(padded_tokens),
        "labels": torch.tensor([b["label"] for b in batch]),
    }


# ============================================================
# 训练
# ============================================================
def train_clip(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CLIP Train] 设备: {device}")

    # 路径
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")
    data_dir = str(PROJECT_ROOT / "data/eurosat")
    output_dir = str(PROJECT_ROOT / "outputs/checkpoints/clip")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]") or 0

    # 下载数据集
    eurosat_path = download_eurosat(data_dir)

    # 创建模型
    print("[CLIP Train] 创建 CLIP 双塔模型...")
    model, vit_config = create_clip_model(device=device)
    model = model.to(device)

    # 只训练图像编码器 + logit_scale，文本编码器冻结
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[CLIP Train] 可训练参数: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # 数据加载
    batch_size = config.get("batch_size", 32)
    train_dataset = EuroSATDataset(eurosat_path, tokenizer, split="train")
    val_dataset = EuroSATDataset(eurosat_path, tokenizer, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_clip(b, pad_token_id),
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_clip(b, pad_token_id),
                            pin_memory=(device.type == "cuda"))

    # 优化器
    lr = config.get("lr", 3e-4)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.1,
    )

    # 学习率调度
    warmup = config.get("warmup_epochs", 2)
    total_epochs = config.get("epochs", 10)
    total_steps = total_epochs * len(train_loader)
    warmup_steps = warmup * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(math.pi * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    history = []
    best_val_acc = 0

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc_i = 0
        epoch_acc_t = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)

            logits_i, logits_t = model(images, tokens)
            loss, acc_i, acc_t = clip_loss(logits_i, logits_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc_i += acc_i.item()
            epoch_acc_t += acc_t.item()

            if step % 20 == 0:
                print(f"  Epoch {epoch:3d} Step {step:4d} | Loss {loss.item():.4f} | "
                      f"ImgAcc {acc_i.item():.2%} | TxtAcc {acc_t.item():.2%}")

        n_steps = len(train_loader)
        avg_loss = epoch_loss / n_steps
        avg_acc_i = epoch_acc_i / n_steps
        avg_acc_t = epoch_acc_t / n_steps
        elapsed = time.time() - t0

        # 验证
        model.eval()
        val_loss = 0
        val_acc_i = 0
        val_acc_t = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                tokens = batch["tokens"].to(device)
                logits_i, logits_t = model(images, tokens)
                loss, acc_i, acc_t = clip_loss(logits_i, logits_t)
                val_loss += loss.item()
                val_acc_i += acc_i.item()
                val_acc_t += acc_t.item()
        n_val = len(val_loader)
        avg_val_loss = val_loss / n_val
        avg_val_acc_i = val_acc_i / n_val
        avg_val_acc_t = val_acc_t / n_val

        print(f"--- Epoch {epoch:3d} | Loss {avg_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc {avg_acc_i:.2%}/{avg_acc_t:.2%} (train) "
              f"{avg_val_acc_i:.2%}/{avg_val_acc_t:.2%} (val) | "
              f"{elapsed:.0f}s ---")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "train_acc": round((avg_acc_i + avg_acc_t) / 2, 4),
            "val_acc": round((avg_val_acc_i + avg_val_acc_t) / 2, 4),
        })

        # 保存最佳
        val_avg_acc = (avg_val_acc_i + avg_val_acc_t) / 2
        if val_avg_acc > best_val_acc:
            best_val_acc = val_avg_acc
            torch.save({"model": model.state_dict(), "config": vit_config, "history": history},
                       os.path.join(output_dir, "best_model.pt"))
            print(f"  → 最佳模型 (val_acc={val_avg_acc:.4f})")

    # 最终保存
    torch.save({"model": model.state_dict(), "config": vit_config, "history": history},
               os.path.join(output_dir, "final_model.pt"))

    history_path = os.path.join(output_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[CLIP] 训练完成！最佳 val_acc: {best_val_acc:.4f}")

    # ============================================================
    # 零样本分类演示
    # ============================================================
    print("\n" + "=" * 60)
    print("[CLIP] 零样本分类演示")
    print("=" * 60)
    demo_zero_shot(model, val_dataset, tokenizer, device)


@torch.no_grad()
def demo_zero_shot(model, dataset, tokenizer, device, n_samples=10):
    """用 CLIP 做零样本分类：图像和所有类别的文本描述比相似度"""
    model.eval()

    # 编码所有类别的文本
    captions = [EUROSAT_CAPTIONS[cls] for cls in EUROSAT_CLASSES]
    text_features_list = []
    for caption in captions:
        enc = tokenizer.encode(caption)
        tokens = torch.tensor([enc.ids[:128]], dtype=torch.long, device=device)
        text_feat = model.encode_text(tokens)
        text_features_list.append(text_feat)
    text_features = torch.cat(text_features_list, dim=0)  # (10, dim)

    correct = 0
    total = 0
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        true_cls = sample["class_name"]

        # 图像特征
        img_feat = model.encode_image(image)  # (1, dim)

        # 与所有类别文本的相似度
        logit_scale = model.logit_scale.exp()
        similarities = logit_scale * (img_feat @ text_features.T)  # (1, 10)
        pred_idx = similarities.argmax(dim=-1).item()
        pred_cls = EUROSAT_CLASSES[pred_idx]

        correct += int(pred_cls == true_cls)
        total += 1

        if i < 5:
            top3 = similarities[0].topk(3)
            top3_classes = [EUROSAT_CLASSES[idx] for idx in top3.indices]
            mark = "✓" if pred_cls == true_cls else "✗"
            print(f"  {mark} True: {true_cls:<20s} Pred: {pred_cls:<20s} Top3: {top3_classes}")

    if total > 0:
        print(f"\n  零样本分类准确率: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 多模态对比学习训练")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    import math
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_epochs": args.warmup_epochs,
    }
    train_clip(config)
