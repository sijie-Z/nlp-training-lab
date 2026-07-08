"""
Reward Model（奖励模型）— RLHF 的第一步

RLHF 三步：
  1. SFT: 指令微调 → 模型学会"回答问题"
  2. Reward Model: 训一个模型给回答打分 → 量化"好不好"
  3. PPO: 用 RM 的信号优化 SFT 模型 → 对齐人类偏好

DPO 跳过了第 2、3 步，直接把偏好数据用 Bradley-Terry 模型做分类。
现在补上第 2 步：从头训一个 Reward Model，用于后续 PPO。

Reward Model 架构：
  基座是预训练的 TinyGPT，把最后的 LM Head 替换成一个标量头：
  Base Model → 取最后一个 token 的 hidden state → Linear → 1 个分数

  训练目标（Bradley-Terry）：
  P(chosen > rejected) = σ(r_chosen - r_rejected)
  L = -log σ(r_chosen - r_rejected)

  这和 DPO 的内在逻辑一致，只是 DPO 用策略的 log-ratio 隐式表示 reward，
  而 Reward Model 显式输出一个标量分数。
"""
import os
import sys
import json
import time
import math
import copy
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tokenizers import Tokenizer
from src.models.tiny_gpt import TinyGPT


class RewardModel(nn.Module):
    """
    奖励模型：基座 Transformer + 标量输出头

    取序列最后一个非 padding token 的 hidden state，
    过一个 Linear 层输出一个标量（分数）。

    GPT 已经使用了 causal mask，所以最后一个 token
    的 hidden state 会包含整个序列的信息。
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.config = base_model.config

        # 冻结 base 的大部分层（可选，先用全冻结然后再决定）
        # 只训练 reward head 作为第一阶段
        for p in self.base.parameters():
            p.requires_grad = False

        # Reward head: hidden_dim → 1
        self.reward_head = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_embd),
            nn.ReLU(),
            nn.Linear(self.config.n_embd, 1),
        )

        print(f"[RewardModel] 基座参数: {sum(p.numel() for p in self.base.parameters()):,} (冻结)")
        print(f"[RewardModel] Reward Head 参数: {sum(p.numel() for p in self.reward_head.parameters()):,} (可训练)")

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, T)
            attention_mask: (B, T) padding mask, 1 = 有效, 0 = 填充

        Returns:
            reward: (B,) 标量分数
        """
        B, T = input_ids.shape

        # 通过基座
        logits, _ = self.base(input_ids)  # logits 不需要, 需要 hidden states
        # 重新获取 hidden states（base model 的 forward 不返回它们）
        # 所以需要手动取

        # 方案：直接用 base 的 encode 部分
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.base.wte(input_ids)
        pos_emb = self.base.wpe(positions)
        x = self.base.drop(tok_emb + pos_emb)

        for layer in self.base.layers:
            x = layer(x)

        x = self.base.ln_f(x)  # (B, T, n_embd)

        # 取最后一个有效 token 的 hidden state
        if attention_mask is not None:
            # 找每个样本最后一个有效位置
            seq_lens = attention_mask.sum(dim=1).long() - 1  # (B,)
            last_hidden = x[torch.arange(B, device=x.device), seq_lens]  # (B, n_embd)
        else:
            last_hidden = x[:, -1, :]  # (B, n_embd)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward

    def get_score(self, text, tokenizer, max_length=256, device="cpu"):
        """给定文本，输出分数（用于推理）"""
        enc = tokenizer.encode(text)
        ids = enc.ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            score = self.forward(input_ids)
        return score.item()


def reward_model_loss(rewards_chosen, rewards_rejected):
    """
    Bradley-Terry 偏好损失

    P(chosen > rejected) = σ(r_chosen - r_rejected)
    L = -log σ(r_chosen - r_rejected)
      = -log(1 / (1 + exp(-(r_chosen - r_rejected))))
      = logsigmoid(r_chosen - r_rejected) 取反

    额外技巧：loss 应该鼓励 r_chosen 和 r_rejected 的差距拉开
    但不能无限大（否则梯度消失）
    """
    # 核心 loss
    margin = rewards_chosen - rewards_rejected  # (B,)
    loss = -F.logsigmoid(margin).mean()

    # 统计
    with torch.no_grad():
        accuracy = (margin > 0).float().mean()
        avg_margin = margin.mean()

    return loss, accuracy, avg_margin


class RewardModelTrainer:
    """Reward Model 训练器"""

    def __init__(self, reward_model, tokenizer, device="cpu"):
        self.model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0
        self.history = []

    def encode_batch(self, chosen_texts, rejected_texts, max_length=256):
        """编码一对文本"""
        chosen_ids = []
        rejected_ids = []
        masks_chosen = []
        masks_rejected = []

        for ct, rt in zip(chosen_texts, rejected_texts):
            c_enc = self.tokenizer.encode(ct)
            r_enc = self.tokenizer.encode(rt)

            c_ids = c_enc.ids[:max_length]
            r_ids = r_enc.ids[:max_length]

            chosen_ids.append(c_ids)
            rejected_ids.append(r_ids)

        # Padding
        max_cl = max(len(ids) for ids in chosen_ids)
        max_rl = max(len(ids) for ids in rejected_ids)

        chosen_padded = []
        rejected_padded = []
        for i in range(len(chosen_ids)):
            cl = len(chosen_ids[i])
            rl = len(rejected_ids[i])
            chosen_padded.append(chosen_ids[i] + [self.pad_token_id] * (max_cl - cl))
            rejected_padded.append(rejected_ids[i] + [self.pad_token_id] * (max_rl - rl))
            masks_chosen.append([1] * cl + [0] * (max_cl - cl))
            masks_rejected.append([1] * rl + [0] * (max_rl - rl))

        return (
            torch.tensor(chosen_padded, dtype=torch.long, device=self.device),
            torch.tensor(rejected_padded, dtype=torch.long, device=self.device),
            torch.tensor(masks_chosen, dtype=torch.float, device=self.device),
            torch.tensor(masks_rejected, dtype=torch.float, device=self.device),
        )

    def train_step(self, batch):
        """一个训练步骤"""
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        chosen_ids, rejected_ids, chosen_mask, rejected_mask = self.encode_batch(
            chosen_texts, rejected_texts
        )

        # 前向
        r_chosen = self.model(chosen_ids, attention_mask=chosen_mask)
        r_rejected = self.model(rejected_ids, attention_mask=rejected_mask)

        # Loss
        loss, accuracy, avg_margin = reward_model_loss(r_chosen, r_rejected)

        return loss, {
            "accuracy": accuracy.item(),
            "avg_margin": avg_margin.item(),
            "r_chosen_mean": r_chosen.mean().item(),
            "r_rejected_mean": r_rejected.mean().item(),
        }

    def train(self, dataloader, optimizer, epochs=5, log_interval=5):
        print(f"[RM Train] 开始训练: {epochs} epochs")
        print("=" * 60)

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            n_steps = 0

            for step, batch in enumerate(dataloader):
                loss, metrics = self.train_step(batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += metrics["accuracy"]
                n_steps += 1

                if step % log_interval == 0:
                    print(f"  Epoch {epoch:3d} Step {step:3d} | Loss {loss.item():.4f} | "
                          f"Acc {metrics['accuracy']:.2%} | "
                          f"Margin {metrics['avg_margin']:.4f} | "
                          f"R+={metrics['r_chosen_mean']:.3f} R-={metrics['r_rejected_mean']:.3f}")

            avg_loss = epoch_loss / max(n_steps, 1)
            avg_acc = epoch_acc / max(n_steps, 1)
            print(f"--- Epoch {epoch:3d} | Avg Loss {avg_loss:.4f} | Avg Acc {avg_acc:.2%} ---")

            self.history.append({"epoch": epoch, "loss": round(avg_loss, 4), "accuracy": round(avg_acc, 4)})

        print(f"[RM Train] 训练完成！最终 accuracy: {self.history[-1]['accuracy']:.2%}")


class RewardDataLoader:
    """加载 DPO 偏好数据，适配 Reward Model 训练"""

    def __init__(self, data_path, batch_size=4):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.batch_size = batch_size
        self.indices = list(range(len(self.data)))
        print(f"[RM Data] 加载 {len(self.data)} 对偏好数据")

    def __len__(self):
        return max(1, len(self.data) // self.batch_size)

    def __iter__(self):
        import random
        random.shuffle(self.indices)
        batch_indices = [self.indices[i:i+self.batch_size]
                        for i in range(0, len(self.indices), self.batch_size)]

        for bi in batch_indices:
            chosen = [self.data[i]["chosen"] for i in bi]
            rejected = [self.data[i]["rejected"] for i in bi]
            yield {"chosen": chosen, "rejected": rejected}


def train_reward_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RM] 设备: {device}")

    # 加载预训练基座
    checkpoint_path = config.get("checkpoint",
        str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch020.pt"))
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")
    dpo_data_path = str(PROJECT_ROOT / "data/dpo/dpo_pairs.json")
    output_dir = config.get("output_dir",
        str(PROJECT_ROOT / "outputs/checkpoints/reward_model"))

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt["config"]

    base_model = TinyGPT(model_config).to(device)
    base_model.load_state_dict(ckpt["model"])

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 构建 Reward Model
    rm = RewardModel(base_model).to(device)

    # 数据
    dataloader = RewardDataLoader(dpo_data_path, batch_size=config.get("batch_size", 4))

    # 优化器（只优化 reward head）
    optimizer = AdamW(
        [p for p in rm.parameters() if p.requires_grad],
        lr=config.get("lr", 1e-4),
    )

    trainer = RewardModelTrainer(rm, tokenizer, device=device)
    trainer.train(dataloader, optimizer, epochs=config.get("epochs", 5))

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "reward_model.pt")
    torch.save({"model": rm.state_dict(), "history": trainer.history}, save_path)
    print(f"[RM] 模型已保存: {save_path}")

    # 测试打分
    print("\n[RM] 打分测试:")
    rm.eval()
    test_pairs = [
        ("什么是GIS？地理信息系统是一种用于采集、存储、管理、分析和展示地理空间数据的计算机系统。",
         "什么是GIS？GIS就是一个做地图的软件。"),
        ("遥感技术是指通过卫星或航空器远距离感知地物信息的技术。",
         "遥感就是拍照。"),
    ]

    for chosen, rejected in test_pairs:
        score_c = rm.get_score(chosen, tokenizer, device=device)
        score_r = rm.get_score(rejected, tokenizer, device=device)
        correct = "✓" if score_c > score_r else "✗"
        print(f"  {correct} chosen={score_c:.3f}, rejected={score_r:.3f} | {chosen[:50]}...")

    return rm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Reward Model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    train_reward_model(config)
