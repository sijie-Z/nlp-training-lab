"""
DPO 训练脚本 — 在预训练的 TinyGPT 上做偏好对齐

用法:
    python scripts/train_dpo.py --epochs 10 --beta 0.1
    python scripts/train_dpo.py --checkpoint outputs/checkpoints/pretrain_tiny/best_model.pt
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tiny_gpt import TinyGPT
from src.trainers.dpo_trainer import DPOTrainer


# ============================================================
# DPO 数据集
# ============================================================
class DPODataset(Dataset):
    """加载 (prompt, chosen, rejected) 三元组"""

    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"[DPO Dataset] 加载 {len(self.data)} 对偏好数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen_enc = self.tokenizer.encode(item["chosen"])
        rejected_enc = self.tokenizer.encode(item["rejected"])

        chosen_ids = chosen_enc.ids[:self.max_length]
        rejected_ids = rejected_enc.ids[:self.max_length]

        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "prompt": item["prompt"],
        }


def collate_fn(batch, pad_token_id=0):
    """填充 batch 到等长"""
    max_chosen = max(len(b["chosen_ids"]) for b in batch)
    max_rejected = max(len(b["rejected_ids"]) for b in batch)

    chosen_inputs = []
    rejected_inputs = []
    for b in batch:
        c = b["chosen_ids"] + [pad_token_id] * (max_chosen - len(b["chosen_ids"]))
        r = b["rejected_ids"] + [pad_token_id] * (max_rejected - len(b["rejected_ids"]))
        chosen_inputs.append(torch.tensor(c, dtype=torch.long))
        rejected_inputs.append(torch.tensor(r, dtype=torch.long))

    return {
        "chosen_input_ids": torch.stack(chosen_inputs),
        "rejected_input_ids": torch.stack(rejected_inputs),
    }


# ============================================================
# 训练
# ============================================================
def train_dpo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DPO Train] 设备: {device}")

    # 路径
    checkpoint_path = config.get("checkpoint_path",
        str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/best_model.pt"))
    tokenizer_path = config.get("tokenizer_path",
        str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json"))
    dpo_data_path = config.get("dpo_data_path",
        str(PROJECT_ROOT / "data/dpo/dpo_pairs.json"))
    output_dir = config.get("output_dir",
        str(PROJECT_ROOT / "outputs/checkpoints/dpo_aligned"))
    os.makedirs(output_dir, exist_ok=True)

    # 加载 tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tokenizer.token_to_id("[PAD]") or 0

    # 加载预训练的 policy model
    print(f"[DPO Train] 加载预训练模型: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt["config"]

    policy_model = TinyGPT(model_config).to(device)
    policy_model.load_state_dict(ckpt["model"])

    # 克隆参考模型（冻结）
    ref_model = TinyGPT(model_config).to(device)
    ref_model.load_state_dict(ckpt["model"])

    # 数据
    dataset = DPODataset(dpo_data_path, tokenizer,
                         max_length=config.get("max_length", 256))
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
    )

    # 优化器
    learning_rate = config.get("learning_rate", 1e-5)
    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # DPO Trainer
    beta = config.get("beta", 0.1)
    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=beta,
        device=device,
    )

    # 训练
    epochs = config.get("epochs", 10)
    trainer.train(dataloader, optimizer, epochs=epochs,
                  log_interval=config.get("log_interval", 5))

    # ============================================================
    # 保存模型
    # ============================================================
    save_path = os.path.join(output_dir, "dpo_model.pt")
    torch.save({
        "model": policy_model.state_dict(),
        "config": model_config,
        "beta": beta,
        "history": trainer.history,
    }, save_path)
    print(f"[DPO Train] 模型已保存: {save_path}")

    # 保存训练历史
    history_path = os.path.join(output_dir, "dpo_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.history, f, ensure_ascii=False, indent=2)
    print(f"[DPO Train] 训练历史: {history_path}")

    # ============================================================
    # Before/After 对比
    # ============================================================
    print("\n" + "=" * 70)
    print("[DPO] Before vs After 对比")
    print("=" * 70)

    test_questions = [
        "请解释什么是GIS",
        "什么是遥感技术",
        "坐标系是什么",
        "NDVI怎么计算",
    ]

    trainer.policy_model = policy_model.to(device)
    trainer.ref_model = ref_model.to(device)

    comparison = trainer.compare_with_ref(test_questions, max_new_tokens=30)
    for item in comparison:
        print(f"\n  ❓ {item['prompt']}")
        print(f"  Before (SFT): {item['ref'][:120]}")
        print(f"  After  (DPO):  {item['policy'][:120]}")

    return policy_model, trainer.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 偏好对齐训练")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 温度")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--checkpoint", type=str, default=None, help="预训练模型路径")
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "beta": args.beta,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }
    if args.checkpoint:
        config["checkpoint_path"] = args.checkpoint

    train_dpo(config)
