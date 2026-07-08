"""
PPO (Proximal Policy Optimization) — RLHF 的第三步

有了 SFT 模型 + Reward Model 后，用 PPO 优化策略。

为什么叫 "Proximal"（近端）？
  PPO 限制每次更新的幅度，防止策略变得太远：
    L = min(r × A, clip(r, 1-ε, 1+ε) × A)
  其中 r = π_new / π_old（新旧策略的比值）

在 RLHF 中：
  - π_new: 正在训练的模型（actor）
  - π_old: 上一步的模型（冻结）
  - A: Advantage（由 Reward Model 给出）
  - ε: clip 范围（通常是 0.2）

额外技巧：KL 惩罚
  防止策略偏离 SFT 模型太远
  L_total = L_ppo - β × KL(π_new || π_sft)

RLHF 完整流程（现在补全了）：
  SFT (LoRA阶段) → Reward Model (本次) → PPO (本次)
  对比 DPO 捷径：直接拿偏好数据做分类
"""
import os
import sys
import json
import time
import math
import copy
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer
from src.models.tiny_gpt import TinyGPT
from src.trainers.reward_model import RewardModel


def compute_log_probs_for_ppo(model, input_ids, pad_token_id=0):
    """
    计算序列上每个 token 的 log-prob（用于 PPO 的 π_new/π_old 比值）

    返回: (B, T-1) 每个位置的 log-prob, (B, T-1) mask
    """
    B, T = input_ids.shape
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab)
    shift_labels = input_ids[:, 1:].contiguous()   # (B, T-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != pad_token_id).float()
    return token_logps, mask


@torch.no_grad()
def compute_rewards(reward_model, tokenizer, prompts, responses, device="cpu"):
    """
    用 Reward Model 给生成的回答打分

    Args:
        prompts: list of str
        responses: list of str
    Returns:
        rewards: (B,) 每个回答的奖励
    """
    texts = [p + " " + r for p, r in zip(prompts, responses)]
    scores = []
    for text in texts:
        score = reward_model.get_score(text, tokenizer, device=device)
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float, device=device)


def ppo_step(
    actor_model,         # π_new: 要训练的模型
    ref_actor_model,     # π_old: 冻结的旧策略
    sft_model,           # π_sft: 初始 SFT 模型（用于 KL 惩罚）
    input_ids,           # (B, T) prompt + response
    advantages,          # (B,) 由 Reward Model 给出的优势
    pad_token_id=0,
    clip_epsilon=0.2,
    kl_beta=0.02,
):
    """
    一次 PPO 更新

    核心公式：
      ratio = exp(log π_new - log π_old)
      L_policy = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
      L_kl = KL(π_new || π_sft) ≈ mean(log π_new - log π_sft)
      L = -(L_policy - β × L_kl)   ← 负号因为要最大化
    """
    B, T = input_ids.shape

    # 计算 log probs
    new_logps, mask = compute_log_probs_for_ppo(actor_model, input_ids, pad_token_id)
    old_logps, _ = compute_log_probs_for_ppo(ref_actor_model, input_ids, pad_token_id)
    sft_logps, _ = compute_log_probs_for_ppo(sft_model, input_ids, pad_token_id)

    # 序列级别的平均 log-prob（排除 padding）
    seq_new_logps = (new_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (B,)
    seq_old_logps = (old_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    seq_sft_logps = (sft_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    # 重要性采样比
    log_ratio = seq_new_logps - seq_old_logps  # (B,)
    ratio = torch.exp(log_ratio)

    # PPO clip loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()  # 负号：最小化负奖励 = 最大化奖励

    # KL 惩罚：防止偏离 SFT 太远
    kl_div = (seq_new_logps - seq_sft_logps).mean()
    kl_loss = kl_beta * kl_div

    total_loss = policy_loss + kl_loss

    # 统计
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()  # 近似 KL
        clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean()

    return total_loss, {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "total_loss": total_loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_frac": clip_frac.item(),
        "ratio_mean": ratio.mean().item(),
    }


def generate_responses(model, tokenizer, prompts, max_new_tokens=50, device="cpu"):
    """用 actor 模型批量生成回答"""
    model.eval()
    responses = []
    for prompt in prompts:
        enc = tokenizer.encode(prompt)
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                       temperature=0.8, top_k=40)
        full_text = tokenizer.decode(output_ids[0].tolist())
        # 去掉 prompt 部分
        if prompt in full_text:
            response = full_text[full_text.index(prompt) + len(prompt):]
        else:
            response = full_text
        responses.append(response)
    return responses


class PPOTrainer:
    """PPO 训练器 — 简化的 RLHF PPO"""

    def __init__(self, actor_model, sft_model, reward_model, tokenizer,
                 clip_epsilon=0.2, kl_beta=0.02, device="cpu"):
        self.actor = actor_model
        self.sft = sft_model
        self.rm = reward_model
        self.tokenizer = tokenizer
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.device = device
        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0

        # 冻结 SFT 和 RM
        for p in self.sft.parameters():
            p.requires_grad = False
        for p in self.rm.parameters():
            p.requires_grad = False

        self.history = []

    def train_epoch(self, prompts, optimizer, ppo_epochs=4, max_new_tokens=50):
        """一个 PPO epoch: 生成 → 打分 → 多次 PPO 更新"""

        # 1. 生成回答
        responses = generate_responses(self.actor, self.tokenizer, prompts,
                                       max_new_tokens=max_new_tokens, device=self.device)
        self.actor.train()

        # 2. 用 Reward Model 打分
        rewards = compute_rewards(self.rm, self.tokenizer, prompts, responses, device=self.device)

        # 3. 标准化 advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 4. 编码 prompt + response
        texts = [p + " " + r for p, r in zip(prompts, responses)]
        max_len = min(256, max(len(self.tokenizer.encode(t).ids) for t in texts))
        input_ids_list = [self.tokenizer.encode(t).ids[:max_len] for t in texts]
        max_batch_len = max(len(ids) for ids in input_ids_list)
        padded = torch.tensor([ids + [self.pad_token_id] * (max_batch_len - len(ids))
                              for ids in input_ids_list], dtype=torch.long, device=self.device)

        # 5. 冻结旧策略的副本
        ref_actor = copy.deepcopy(self.actor)
        for p in ref_actor.parameters():
            p.requires_grad = False

        # 6. 多次 PPO 更新（同一批数据）
        epoch_metrics = []
        for ppo_step_i in range(ppo_epochs):
            loss, metrics = ppo_step(
                self.actor, ref_actor, self.sft,
                padded, advantages,
                pad_token_id=self.pad_token_id,
                clip_epsilon=self.clip_epsilon,
                kl_beta=self.kl_beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_metrics.append(metrics)

        # 汇总
        avg_policy_loss = sum(m["policy_loss"] for m in epoch_metrics) / len(epoch_metrics)
        avg_kl = sum(m["kl_div"] for m in epoch_metrics) / len(epoch_metrics)
        avg_reward = rewards.mean().item()

        print(f"  Reward: {avg_reward:.3f} | Policy Loss: {avg_policy_loss:.4f} | "
              f"KL: {avg_kl:.4f} | Clip Frac: {epoch_metrics[-1]['clip_frac']:.2%}")

        return {
            "avg_reward": avg_reward,
            "avg_policy_loss": avg_policy_loss,
            "avg_kl": avg_kl,
            "clip_frac": epoch_metrics[-1]["clip_frac"],
        }


def run_ppo_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PPO] 设备: {device}")

    # 加载模型
    sft_ckpt_path = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch020.pt")
    rm_ckpt_path = str(PROJECT_ROOT / "outputs/checkpoints/reward_model/reward_model.pt")
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # SFT 模型
    sft_ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
    model_config = sft_ckpt["config"]
    sft_model = TinyGPT(model_config).to(device)
    sft_model.load_state_dict(sft_ckpt["model"])

    # Actor 模型（从 SFT 初始化）
    actor_model = TinyGPT(model_config).to(device)
    actor_model.load_state_dict(sft_ckpt["model"])

    # Reward Model
    base_model = TinyGPT(model_config).to(device)
    base_model.load_state_dict(sft_ckpt["model"])
    rm = RewardModel(base_model).to(device)
    if os.path.exists(rm_ckpt_path):
        rm_ckpt = torch.load(rm_ckpt_path, map_location=device, weights_only=False)
        rm.load_state_dict(rm_ckpt["model"])
        print("[PPO] Reward Model 加载成功")
    else:
        print("[PPO] 警告: Reward Model 不存在，使用未经训练的 RM（效果会很差）")

    # 训练数据（用一部分 benchmark 问题作为 prompt）
    with open(str(PROJECT_ROOT / "data/benchmark/gis_benchmark.json"), "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    prompts = [q["question"] for q in benchmark["questions"][::5]]  # 每5题取1题 → 10题
    print(f"[PPO] 训练 prompt 数: {len(prompts)}")

    # 优化器
    lr = config.get("lr", 1e-6)
    optimizer = AdamW([p for p in actor_model.parameters() if p.requires_grad], lr=lr)

    trainer = PPOTrainer(
        actor_model, sft_model, rm, tokenizer,
        clip_epsilon=config.get("clip_epsilon", 0.2),
        kl_beta=config.get("kl_beta", 0.02),
        device=device,
    )

    # PPO 训练
    ppo_epochs_per = config.get("ppo_epochs_per_step", 4)
    total_steps = config.get("total_steps", 5)

    print(f"\n[PPO] 开始训练: {total_steps} 步, 每步 {ppo_epochs_per} 次 PPO 更新")
    print("=" * 60)

    for step in range(total_steps):
        print(f"\n[PPO] Step {step+1}/{total_steps}")
        metrics = trainer.train_epoch(prompts, optimizer, ppo_epochs=ppo_epochs_per)
        trainer.history.append(metrics)

    # 保存
    output_dir = str(PROJECT_ROOT / "outputs/checkpoints/ppo_aligned")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ppo_model.pt")
    torch.save({"model": actor_model.state_dict(), "history": trainer.history}, save_path)
    print(f"\n[PPO] 模型已保存: {save_path}")

    # 对比：SFT vs PPO
    print("\n" + "=" * 60)
    print("[PPO] SFT vs PPO 生成对比")
    print("=" * 60)

    test_prompts = [
        "什么是GIS",
        "遥感技术是什么",
        "NDVI怎么计算",
        "坐标系转换方法",
    ]

    sft_model.eval()
    actor_model.eval()

    for prompt in test_prompts:
        sft_resp = generate_responses(sft_model, tokenizer, [prompt], device=device)[0]
        ppo_resp = generate_responses(actor_model, tokenizer, [prompt], device=device)[0]

        print(f"\n  Prompt: {prompt}")
        print(f"  SFT:    {sft_resp[:120]}")
        print(f"  PPO:    {ppo_resp[:120]}")

    return actor_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO RLHF 训练")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    args = parser.parse_args()

    run_ppo_training({"lr": args.lr, "total_steps": args.steps, "kl_beta": args.kl_beta})
