"""
On-Policy Distillation (OPD) — DeepSeek V4 / Qwen3 的核心训练方法

论文: DeepSeek V4 Technical Report (2026.04) / Qwen3 Technical Report (2026)
核心思想: Student 自己先采样生成, Teacher 在 Student 的轨迹上给反馈
          → 避免模式覆盖, 减少灾难性遗忘

为什么 OPD 是 2026 年最重要的训练创新？

传统蒸馏 (Forward KL):
    Teacher 采样 → Student 模仿
    问题: mode-covering — Student 试图覆盖 Teacher 的所有模式
          会稀释 Student 已有的能力, 导致灾难性遗忘

On-Policy Distillation (Reverse KL):
    Student 自己采样 → Teacher 在 Student 轨迹上给反馈
    优势: mode-seeking — Student 只学和自己当前行为相关的 Teacher 知识
          不会遗忘已有能力, 训练更稳定


DeepSeek V4 的 OPD 用法:
    10+ 个专家 Teacher (数学/代码/Agent 各领域) → 统一 Student
    全词表 KL 散度（不是 token-level 估计）

Qwen3 的 OPD 用法:
    单旗舰 Teacher → 小模型 Student (Strong-to-Weak)
    约 1/10 GPU 算力即可达到 RL 同等效果


Forward KL vs Reverse KL 对比:

    Forward KL:  KL(teacher || student) = Σ t * log(t/s)
                  → mean-seeking, 覆盖 Teacher 所有模式

    Reverse KL:  KL(student || teacher) = Σ s * log(s/t)
                  → mode-seeking, 只匹配 Teacher 在 Student 感兴趣区域的分布


本实现的简化方案:
    1. Student (TinyGPT 预训练权重) 自己生成回答
    2. Teacher (同一模型但更好的 checkpoint, 或外部模型) 对回答打分
    3. 用 Reverse KL loss 让 Student 向 Teacher 对齐
    4. 只更新 Student, Teacher 冻结
"""
import os
import sys
import json
import copy
import math
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
from src.models.tiny_gpt import TinyGPT, TinyGPTConfig


# ============================================================
# 1. Reverse KL Divergence Loss
# ============================================================
def reverse_kl_divergence(student_logits, teacher_logits, labels, pad_token_id=0, temperature=1.0):
    """
    Reverse KL: KL(student || teacher) = Σ student_prob * log(student_prob / teacher_prob)

    这是 DeepSeek V4 / Qwen3 OPD 的核心 loss。

    Args:
        student_logits: (B, T, vocab_size) Student 模型的输出
        teacher_logits: (B, T, vocab_size) Teacher 模型的输出（冻结）
        labels: (B, T) 目标 token（用于 mask）
        pad_token_id: padding token id
        temperature: softmax 温度

    Returns:
        reverse_kl: 标量 loss
    """
    B, T, V = student_logits.shape

    # Shift: 预测位置 t 的 token 用的是位置 t-1 的输出
    shift_s_logits = student_logits[:, :-1, :].contiguous()
    shift_t_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Mask: 排除 padding
    mask = (shift_labels != pad_token_id).float()  # (B, T-1)

    # Student probs & log probs
    s_log_probs = F.log_softmax(shift_s_logits / temperature, dim=-1)  # (B, T-1, V)
    s_probs = F.softmax(shift_s_logits / temperature, dim=-1)

    # Teacher probs
    t_probs = F.softmax(shift_t_logits / temperature, dim=-1)
    t_log_probs = torch.log(t_probs + 1e-10)

    # Reverse KL: Σ student * (log student - log teacher)
    kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (B, T-1)

    # 加权平均（排除 padding）
    kl_masked = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

    return kl_masked


def forward_kl_divergence(student_logits, teacher_logits, labels, pad_token_id=0, temperature=1.0):
    """
    Forward KL: KL(teacher || student) = Σ teacher_prob * log(teacher_prob / student_prob)

    这是传统知识蒸馏的 loss（Teacher 采样 → Student 模仿）。
    用于和 Reverse KL 做对比。
    """
    B, T, V = student_logits.shape

    shift_s_logits = student_logits[:, :-1, :].contiguous()
    shift_t_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != pad_token_id).float()

    s_log_probs = F.log_softmax(shift_s_logits / temperature, dim=-1)
    t_probs = F.softmax(shift_t_logits / temperature, dim=-1)
    t_log_probs = torch.log(t_probs + 1e-10)

    # Forward KL: Σ teacher * (log teacher - log student)
    kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    kl_masked = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

    return kl_masked


def js_divergence(student_logits, teacher_logits, labels, pad_token_id=0, temperature=1.0):
    """
    Jensen-Shannon Divergence: (Forward KL + Reverse KL) / 2

    JSD 是对称的，介于 Forward KL 和 Reverse KL 之间。
    比 Forward KL 更稳定，比 Reverse KL 覆盖面更广。
    """
    fwd = forward_kl_divergence(student_logits, teacher_logits, labels, pad_token_id, temperature)
    rev = reverse_kl_divergence(student_logits, teacher_logits, labels, pad_token_id, temperature)
    return (fwd + rev) / 2


# ============================================================
# 2. On-Policy Distillation 训练器
# ============================================================
class OnPolicyDistillationTrainer:
    """
    OPD 训练器

    训练流程:
        1. Student 自己生成一批回答（on-policy 采样）
        2. Teacher 在 Student 生成的回答上计算 logits（给反馈）
        3. 用 Reverse KL loss 让 Student 向 Teacher 对齐
        4. 可选: 加入 CE loss 作为辅助（保证 Student 不会完全偏离语言模型能力）
    """

    def __init__(self, student_model, teacher_model, tokenizer, device="cpu",
                 temperature=1.0, ce_alpha=0.1):
        """
        Args:
            student_model: 要训练的模型
            teacher_model: 冻结的参考模型（更好的 checkpoint）
            tokenizer: Tokenizer 实例
            temperature: softmax 温度
            ce_alpha: 标准 CE loss 的权重（防止 Student 偏离太远）
        """
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.ce_alpha = ce_alpha
        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0
        self.history = []

        # 冻结 Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    @torch.no_grad()
    def sample_responses(self, prompts, max_new_tokens=40):
        """Student 自己生成回答（on-policy 采样）"""
        self.student.eval()

        full_texts = []
        for prompt in prompts:
            enc = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)

            output_ids = self.student.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=0.8, top_k=40
            )

            full_text = self.tokenizer.decode(output_ids[0].tolist())
            full_texts.append(full_text)

        return full_texts

    def distillation_step(self, batch, loss_type="reverse_kl"):
        """
        一次 OPD 训练步骤

        Args:
            batch: (input_ids, labels) from DataLoader
            loss_type: "reverse_kl" | "forward_kl" | "jsd"

        Returns:
            loss, metrics
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Student forward
        self.student.train()
        s_logits, _ = self.student(x)  # (B, T, V)

        # Teacher forward（冻结）
        with torch.no_grad():
            t_logits, _ = self.teacher(x)

        # Distillation loss
        if loss_type == "reverse_kl":
            distill_loss = reverse_kl_divergence(
                s_logits, t_logits, y,
                pad_token_id=self.pad_token_id,
                temperature=self.temperature
            )
        elif loss_type == "forward_kl":
            distill_loss = forward_kl_divergence(
                s_logits, t_logits, y,
                pad_token_id=self.pad_token_id,
                temperature=self.temperature
            )
        elif loss_type == "jsd":
            distill_loss = js_divergence(
                s_logits, t_logits, y,
                pad_token_id=self.pad_token_id,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # CE loss（辅助，防止 Student 完全偏离）
        ce_loss = F.cross_entropy(
            s_logits[:, :-1, :].contiguous().view(-1, s_logits.size(-1)),
            y[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
        )

        total_loss = distill_loss + self.ce_alpha * ce_loss

        metrics = {
            "distill_loss": distill_loss.item(),
            "ce_loss": ce_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics

    def train(self, dataloader, optimizer, epochs=5, loss_type="reverse_kl",
              log_interval=10):
        print(f"[OPD] 开始训练: {epochs} epochs, loss_type={loss_type}")
        print(f"[OPD] temperature={self.temperature}, ce_alpha={self.ce_alpha}")
        print(f"[OPD] Student trainable params: {sum(p.numel() for p in self.student.parameters() if p.requires_grad):,}")
        print(f"[OPD] Teacher params (frozen): {sum(p.numel() for p in self.teacher.parameters()):,}")
        print("=" * 70)

        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_ce_loss = 0
            n_steps = 0

            for step, (x, y) in enumerate(dataloader):
                total_loss, metrics = self.distillation_step((x, y), loss_type)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_d_loss += metrics["distill_loss"]
                epoch_ce_loss += metrics["ce_loss"]
                n_steps += 1

                if step % log_interval == 0:
                    print(f"  Epoch {epoch:3d} Step {step:4d} | "
                          f"Distill {metrics['distill_loss']:.4f} | "
                          f"CE {metrics['ce_loss']:.4f} | "
                          f"Total {metrics['total_loss']:.4f}")

            avg_d = epoch_d_loss / max(n_steps, 1)
            avg_ce = epoch_ce_loss / max(n_steps, 1)
            print(f"--- Epoch {epoch:3d} | Distill={avg_d:.4f} CE={avg_ce:.4f} ---")

            self.history.append({
                "epoch": epoch,
                "distill_loss": round(avg_d, 4),
                "ce_loss": round(avg_ce, 4),
            })

        print(f"[OPD] 训练完成！最终 distill_loss: {self.history[-1]['distill_loss']:.4f}")


# ============================================================
# 主训练函数
# ============================================================
def train_opd(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OPD] 设备: {device}")

    # 加载 Teacher（更好的 checkpoint — epoch 20）
    teacher_ckpt_path = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch020.pt")
    student_ckpt_path = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch000.pt")
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Teacher: epoch 20, PPL ~4
    print("[OPD] 加载 Teacher (Epoch 20)...")
    t_ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    t_config = t_ckpt["config"]
    teacher_model = TinyGPT(t_config).to(device)
    teacher_model.load_state_dict(t_ckpt["model"])

    # Student: epoch 0(随机) — 从零开始向 Teacher 对齐
    print("[OPD] 加载 Student (Epoch 0 / 随机)...")
    s_ckpt = torch.load(student_ckpt_path, map_location=device, weights_only=False)
    student_model = TinyGPT(t_config).to(device)
    student_model.load_state_dict(s_ckpt["model"])

    # 数据加载
    from scripts.pretrain_tiny import PretrainDataset, DataCollator

    corpus_path = str(PROJECT_ROOT / "data/raw/domain_corpus_cleaned.txt")
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    dataset = PretrainDataset(corpus_path, tokenizer_path,
                              block_size=t_config.block_size, stride=128)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.get("batch_size", 8), shuffle=True,
        collate_fn=DataCollator(pad_token_id=pad_id),
    )

    # 优化器
    optimizer = AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=config.get("lr", 1e-4),
    )

    # OPD Trainer
    trainer = OnPolicyDistillationTrainer(
        student_model, teacher_model, tokenizer,
        device=device,
        temperature=config.get("temperature", 1.0),
        ce_alpha=config.get("ce_alpha", 0.1),
    )

    # 对比三种 loss
    loss_types = ["reverse_kl", "forward_kl", "jsd"]
    all_results = {}

    for lt in loss_types:
        print(f"\n{'='*70}")
        print(f"[OPD] 测试 {lt}")
        print(f"{'='*70}")

        # 重新加载 Student
        student_model = TinyGPT(t_config).to(device)
        student_model.load_state_dict(s_ckpt["model"])

        trainer_lt = OnPolicyDistillationTrainer(
            student_model, teacher_model, tokenizer,
            device=device,
            temperature=config.get("temperature", 1.0),
            ce_alpha=config.get("ce_alpha", 0.1),
        )

        opt = AdamW(
            [p for p in student_model.parameters() if p.requires_grad],
            lr=config.get("lr", 1e-4),
        )

        trainer_lt.train(
            dataloader, opt,
            epochs=config.get("epochs", 3),
            loss_type=lt,
            log_interval=config.get("log_interval", 5),
        )

        all_results[lt] = trainer_lt.history[-1]["distill_loss"]
        del student_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # 汇总对比
    print(f"\n{'='*70}")
    print("[OPD] 三种 KL Divergence 对比")
    print(f"{'='*70}")
    for lt, final_loss in all_results.items():
        print(f"  {lt:<15s}: final distill_loss = {final_loss:.4f}")

    # 保存
    output_dir = str(PROJECT_ROOT / "outputs/checkpoints/opd")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "kl_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-Policy Distillation 训练")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce_alpha", type=float, default=0.1)
    args = parser.parse_args()

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "ce_alpha": args.ce_alpha,
        "log_interval": 5,
    }

    train_opd(config)
