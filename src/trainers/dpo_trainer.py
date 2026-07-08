"""
DPO (Direct Preference Optimization) — 从零实现

参考论文: Direct Preference Optimization (Rafailov et al., 2023)
https://arxiv.org/abs/2305.18290

DPO 的核心思想：
  不用先训 Reward Model 再做 PPO，而是直接在偏好数据上用 Bradley-Terry 模型
  优化策略，等价于 RLHF 的最优解。

公式:
  L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中:
  - π_θ: 当前策略（正在训练的模型）
  - π_ref: 参考策略（冻结的初始模型）
  - y_w: 偏好回答 (chosen)
  - y_l: 非偏好回答 (rejected)
  - β: 温度参数，控制偏离参考策略的程度
  - σ: sigmoid 函数
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps,    # 当前模型对 chosen 的 log-prob (B,)
    policy_rejected_logps,  # 当前模型对 rejected 的 log-prob (B,)
    ref_chosen_logps,       # 参考模型对 chosen 的 log-prob (B,)
    ref_rejected_logps,     # 参考模型对 rejected 的 log-prob (B,)
    beta=0.1,               # 温度参数
    reference_free=False,   # 是否无参考（等同于 SFT）
):
    """
    DPO Loss 实现

    如果 reference_free=True，退化为简单的对比损失：
      L = -log σ(β * (policy_chosen_logps - policy_rejected_logps))

    返回:
      loss: 标量
      chosen_rewards: chosen 的隐式奖励 (B,)  — 越大越好
      rejected_rewards: rejected 的隐式奖励 (B,) — 越小越好
      accuracy: chosen 奖励 > rejected 奖励的比例
    """
    if reference_free:
        log_ratio = beta * (policy_chosen_logps - policy_rejected_logps)
    else:
        pi_log_ratio = policy_chosen_logps - policy_rejected_logps
        ref_log_ratio = ref_chosen_logps - ref_rejected_logps
        log_ratio = beta * (pi_log_ratio - ref_log_ratio)

    # DPO loss: -log σ(log_ratio)
    loss = -F.logsigmoid(log_ratio).mean()

    # 隐式奖励: r = β * log(π_θ/π_ref)
    with torch.no_grad():
        if reference_free:
            chosen_rewards = beta * policy_chosen_logps
            rejected_rewards = beta * policy_rejected_logps
        else:
            chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), accuracy


def compute_log_probs(model, input_ids, labels, pad_token_id=0):
    """
    计算模型在 labels 上的 token-level log-probabilities

    返回: (B,) — 每个样本的平均 log-prob（排除 padding）

    为什么用平均 log-prob？
      - DPO 公式要求将序列长度归一化，避免长序列有优势
      - 求和会偏向长回答，除以长度才公平
    """
    B, T = input_ids.shape
    logits, _ = model(input_ids)  # (B, T, vocab_size)

    # Shift: 预测位置 t 的 token 用的是位置 t-1 的 logit
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 逐 token 的 log-prob
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, vocab)

    # 收集每个 label token 的 log-prob
    token_logps = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask: 排除 padding
    mask = (shift_labels != pad_token_id).float()

    # 平均 log-prob（每个 token 权重相同）
    # 注意：如果某个样本全是 pad，要避免除零
    seq_logps = (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    return seq_logps  # (B,)


def compute_full_log_probs(model, input_ids, labels, pad_token_id=0):
    """
    返回逐 token 的 log-prob（用于分析 DPO 训练质量）
    """
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    mask = (shift_labels != pad_token_id).float()
    return token_logps, mask


class DPOTrainer:
    """DPO 训练器 — 封装训练循环"""

    def __init__(self, policy_model, ref_model, tokenizer, beta=0.1, device="cpu"):
        self.policy_model = policy_model    # π_θ  — 要训练
        self.ref_model = ref_model          # π_ref — 冻结
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device

        # 冻结参考模型
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        self.pad_token_id = tokenizer.token_to_id("[PAD]") or 0
        self.history = []

    def encode_pair(self, prompt, chosen, rejected, max_length=256):
        """将 (prompt, chosen, rejected) 编码为 model inputs

        对于生成式模型的 DPO，我们直接把 (prompt + response) 拼接在一起编码。
        labels 就是 input_ids 本身（自回归），但在 prompt 部分设为 padding token
        这样模型只在 response 部分计算 loss。

        简化版：直接编码完整的 chosen 和 rejected，因为我们的 data 已经是完整文本了。
        """
        # 编码整个序列
        chosen_enc = self.tokenizer.encode(chosen)
        chosen_ids = chosen_enc.ids[:max_length]

        rejected_enc = self.tokenizer.encode(rejected)
        rejected_ids = rejected_enc.ids[:max_length]

        return chosen_ids, rejected_ids

    def pad_to_max(self, sequences, pad_value=None):
        """将不等长的序列列表填充到相同长度"""
        if pad_value is None:
            pad_value = self.pad_token_id
        max_len = max(len(s) for s in sequences)
        padded = []
        for s in sequences:
            pad_len = max_len - len(s)
            padded.append(s + [pad_value] * pad_len)
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def train_step(self, batch):
        """一个 DPO 训练步骤"""
        chosen_inputs = batch["chosen_input_ids"].to(self.device)
        rejected_inputs = batch["rejected_input_ids"].to(self.device)

        # 前向：计算当前策略的 log-prob
        self.policy_model.train()
        policy_chosen_logps = compute_log_probs(
            self.policy_model, chosen_inputs, chosen_inputs,
            pad_token_id=self.pad_token_id
        )
        policy_rejected_logps = compute_log_probs(
            self.policy_model, rejected_inputs, rejected_inputs,
            pad_token_id=self.pad_token_id
        )

        # 前向：计算参考策略的 log-prob（冻结）
        with torch.no_grad():
            ref_chosen_logps = compute_log_probs(
                self.ref_model, chosen_inputs, chosen_inputs,
                pad_token_id=self.pad_token_id
            )
            ref_rejected_logps = compute_log_probs(
                self.ref_model, rejected_inputs, rejected_inputs,
                pad_token_id=self.pad_token_id
            )

        # DPO loss
        loss, chosen_reward, rejected_reward, accuracy = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )

        return loss, {
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
            "reward_margin": (chosen_reward - rejected_reward).item(),
            "accuracy": accuracy.item(),
        }

    def train(self, dataloader, optimizer, scheduler=None, epochs=5, log_interval=5):
        print(f"[DPO] 开始训练: {epochs} epochs, β={self.beta}")
        print(f"[DPO] Policy model trainable params: {sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad):,}")
        print(f"[DPO] Ref model params (frozen): {sum(p.numel() for p in self.ref_model.parameters()):,}")
        print("=" * 70)

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            epoch_margin = 0
            n_steps = 0

            for step, batch in enumerate(dataloader):
                loss, metrics = self.train_step(batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), max_norm=1.0
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_loss += loss.item()
                epoch_acc += metrics["accuracy"]
                epoch_margin += metrics["reward_margin"]
                n_steps += 1

                if step % log_interval == 0:
                    print(f"  Epoch {epoch:3d} Step {step:3d} | "
                          f"Loss {loss.item():.4f} | "
                          f"Acc {metrics['accuracy']:.2%} | "
                          f"Margin {metrics['reward_margin']:.4f}")

            avg_loss = epoch_loss / max(n_steps, 1)
            avg_acc = epoch_acc / max(n_steps, 1)
            avg_margin = epoch_margin / max(n_steps, 1)

            print(f"--- Epoch {epoch:3d} Avg | Loss {avg_loss:.4f} | "
                  f"Acc {avg_acc:.2%} | Margin {avg_margin:.4f} ---")

            self.history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "accuracy": round(avg_acc, 4),
                "reward_margin": round(avg_margin, 4),
            })

        print(f"[DPO] 训练完成！最终 accuracy: {self.history[-1]['accuracy']:.2%}")

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
        """使用训练后的策略模型生成"""
        self.policy_model.eval()
        enc = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.policy_model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k
            )
        return self.tokenizer.decode(output[0].tolist())

    def compare_with_ref(self, prompts, max_new_tokens=50):
        """对比 policy 和 ref 两个模型的生成结果"""
        results = []
        for prompt in prompts:
            policy_text = self.generate(prompt, max_new_tokens=max_new_tokens)

            # 用 ref model 生成
            self.ref_model.eval()
            enc = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([enc.ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                ref_out = self.ref_model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    temperature=0.8, top_k=40
                )
            ref_text = self.tokenizer.decode(ref_out[0].tolist())

            results.append({
                "prompt": prompt,
                "ref": ref_text,
                "policy": policy_text,
            })
        return results
