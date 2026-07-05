"""
LoRA 训练器

与 BERT 分类 Trainer 的核心区别：
1. 数据是 JSONL 格式的 (instruction, output)
2. 需要拼成对话格式：### Instruction: ... ### Response: ...
3. Loss 只计算 output 部分的 token（instruction 部分 mask 掉）
4. 评估看 Loss 和 Perplexity，而非 Accuracy
"""

import os
import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup


class InstructionDataset(Dataset):
    """指令微调数据集"""

    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"]
        output = sample["output"]

        # 拼成对话格式
        # 注意：output 后面加 eos_token，让模型学会停止
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        eos_text = text + self.tokenizer.eos_token

        # Tokenize
        encoded = self.tokenizer(
            eos_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 构造 labels：只计算 Response 部分的 loss
        # instruction 部分的 token 设 -100（被 CrossEntropyLoss 忽略）
        labels = input_ids.clone()

        # 找到 "### Response:\n" 在 token 序列中的位置
        response_start = text.find("### Response:\n") + len("### Response:\n")
        prefix_text = text[:response_start]
        prefix_ids = self.tokenizer(
            prefix_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        instruction_len = len(prefix_ids)
        # instruction 部分的 label 设为 -100
        labels[:instruction_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LoRATrainer:
    """
    LoRA 训练器

    和 BERT Trainer 的区别：
    - 输入是 instruction + output（不是分类标签）
    - Loss 只计算 output 部分（不是整句）
    - 评估指标看 Loss / Perplexity（不是 Accuracy）
    """

    def __init__(self, model, tokenizer, train_dataset, val_dataset, config, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.system.device == "cpu":
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )

        # 优化器（只训练 LoRA 参数）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
        )

        # 学习率调度
        total_steps = len(self.train_loader) * config.training.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * config.training.warmup_ratio),
            num_training_steps=total_steps,
        )

        # 计时
        self.start_time = None
        self.best_loss = float("inf")

    def train(self):
        self.logger.info(f"Start LoRA training for {self.config.training.epochs} epochs")
        self.logger.info(f"Train samples: {len(self.train_dataset)}, "
                         f"Val samples: {len(self.val_dataset)}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")

        self.start_time = time.time()
        train_losses = []
        val_losses = []

        for epoch in range(1, self.config.training.epochs + 1):
            # ====== 训练 ======
            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                if step % self.config.logging.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch} | Step {step}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )

            avg_train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # ====== 验证 ======
            avg_val_loss = self._validate()
            val_losses.append(avg_val_loss)
            perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

            self.logger.info(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}"
            )

            # 保存最佳模型
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self._save_adapter()

        elapsed = time.time() - self.start_time
        self.logger.info(f"Training complete! Best val loss: {self.best_loss:.4f}")
        self.logger.info(f"Time: {elapsed:.1f}s")

        # 显存统计
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated(0) / 1024**3
            self.logger.info(f"Max GPU memory: {max_mem:.2f} GB")

        return {"best_loss": self.best_loss, "time": elapsed}

    def _validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        return total_loss / len(self.val_loader)

    def _save_adapter(self):
        save_dir = f"{self.config.system.output_dir}/checkpoints/lora_adapter"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # 保存 adapter 大小
        adapter_size = 0
        for f in os.listdir(save_dir):
            fp = os.path.join(save_dir, f)
            if os.path.isfile(fp):
                adapter_size += os.path.getsize(fp)
        self.logger.info(f"Adapter saved to {save_dir} ({adapter_size/1024**2:.1f} MB)")

    def generate(self, instruction, max_new_tokens=50):
        """训练后推理测试"""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只返回 Response 部分
        if "### Response:\n" in full:
            return full.split("### Response:\n")[-1].strip()
        return full
