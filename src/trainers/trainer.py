"""
训练器（V0.1 含验证）

V0.1 不拆分独立的 Evaluator 类。
验证逻辑（_validate）直接写在 Trainer 内部，只有 ~20 行。

等第二阶段（文本匹配）需要 Precision/Recall/F1 时再抽成独立文件。
"""

import os
import json
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # 不显示图形窗口，直接保存文件
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from src.debug.shape_tracker import ShapeTracker


class Trainer:
    """
    训练器

    职责：
    1. 训练循环（epoch → batch → forward → backward → step）
    2. 验证循环（每个 epoch 结束后算 accuracy）
    3. 保存最佳模型
    4. 记录日志 + 画图
    """

    def __init__(self, model, tokenizer, train_loader, val_loader,
                 config, logger, label2id=None):
        """
        参数：
            model: 模型（from factory.build_model）
            tokenizer: HuggingFace tokenizer
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader
            config: OmegaConf 配置对象
            logger: 日志实例
            label2id: 标签→数字映射（用于保存）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.label2id = label2id

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
        if config.system.device == "cpu":
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")

        # 优化器（AdamW 是 BERT 微调的标配）
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            eps=config.training.adam_epsilon,
            weight_decay=config.training.weight_decay,
        )

        # 学习率调度器（先 warmup，再线性衰减）
        total_steps = len(train_loader) * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Shape Tracker（默认开启）
        self.tracker = ShapeTracker(enabled=True)

        # 最佳准确率
        self.best_acc = 0.0

    def train(self):
        """
        完整训练流程

        伪代码：
        for epoch in 1..epochs:
            for batch in train_loader:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()       # 反向传播
                optimizer.step()      # 更新参数
                scheduler.step()      # 调整学习率

            val_acc = _validate()     # 验证

            if val_acc > best_acc:
                _save_checkpoint()    # 保存最佳模型

            _plot_curves()            # 画图
        """
        self.logger.info(f"Start training for {self.config.training.epochs} epochs")
        self.logger.info(f"Train batches: {len(self.train_loader)}, "
                         f"Val batches: {len(self.val_loader)}")

        train_losses = []
        val_accs = []

        for epoch in range(1, self.config.training.epochs + 1):
            self.logger.info(f"{'='*20} Epoch {epoch}/{self.config.training.epochs} {'='*20}")

            # ====== 训练阶段 ======
            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                # 数据搬到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # ====== DEBUG: 第一个 batch 打印详细信息 ======
                if step == 0 and epoch == 1:
                    print("\n" + "=" * 60)
                    print("--- FIRST BATCH DEBUG ---")
                    print("=" * 60)
                    print(f"  input_ids shape:      {batch['input_ids'].shape}")
                    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
                    print(f"  labels shape:         {batch['labels'].shape}")
                    print(f"  labels value:         {batch['labels'].tolist()}")
                    print(f"  input_ids sample:     {batch['input_ids'][0, :10].tolist()}...")
                    print(f"  device:               {batch['input_ids'].device}")
                    print("=" * 60 + "\n")

                # 追踪 input shape（只在第一个 epoch 第一步追踪）
                if epoch == 1 and step == 0:
                    self.tracker.track("输入", input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"],
                                       labels=batch["labels"])

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss

                # ====== DEBUG: 第一个 batch 打印 logits ======
                if step == 0 and epoch == 1:
                    print("\n  [Model Forward]")
                    print(f"  logits shape:         {outputs.logits.shape}")
                    print(f"  logits first 5 rows:  {outputs.logits[:5].tolist()}")
                    print(f"  loss value:           {loss.item():.4f}")
                    print("-" * 60 + "\n")

                # 追踪 logits shape（只在第一个 epoch 第一步追踪）
                if epoch == 1 and step == 0:
                    self.tracker.track("Logits", logits=outputs.logits)

                # 反向传播 + 优化
                loss.backward()

                # ====== DEBUG: 第一个 batch 打印梯度 ======
                if step == 0 and epoch == 1:
                    # 取第一层参数的梯度看看
                    first_param = list(self.model.parameters())[0]
                    print("\n  [Backward Done]")
                    print(f"  grad shape:           {first_param.grad.shape}")
                    print(f"  grad mean:            {first_param.grad.mean().item():.6f}")
                    print(f"  grad max:             {first_param.grad.max().item():.6f}")
                    print(f"  grad min:             {first_param.grad.min().item():.6f}")
                    print("-" * 60 + "\n")

                # 梯度裁剪（防梯度爆炸）
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                # ====== 每步都打印 loss（第一次训练重点观察）======
                if step < 5 or step % self.config.logging.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"  Step {step:3d}/{len(self.train_loader)} | "
                          f"Loss: {loss.item():.4f} | LR: {lr:.2e}")

            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)

            # 打印 Shape Track（第一个 epoch 结束后打印一次）
            if epoch == 1:
                self.tracker.print_summary()

            # ====== 验证阶段 ======
            print("\n" + "-" * 40)
            print("  [Validation] Running on val_loader...")
            val_acc = self._validate()
            val_accs.append(val_acc)

            # ====== DEBUG: 验证结果 ======
            print(f"  [Validation] Accuracy: {val_acc:.4f}")
            if val_acc > self.best_acc:
                print(f"  >>> New best accuracy! <<<")
            print("-" * 40 + "\n")

            self.logger.info(
                f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | "
                f"Val Accuracy: {val_acc:.4f}"
            )

            # ====== 保存最佳模型 ======
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint()
                self.logger.info(f"★ New best model saved with accuracy {val_acc:.4f}")

        # ====== 画图 ======
        self._plot_curves(train_losses, val_accs)

        self.logger.info(f"Training complete! Best accuracy: {self.best_acc:.4f}")
        return {"best_accuracy": self.best_acc}

    def _validate(self):
        """
        验证（V0.1 简易版）

        流程：
        1. model.eval() — 切换到评估模式（关掉 dropout）
        2. torch.no_grad() — 关掉梯度计算（省显存 + 提速）
        3. 遍历所有 batch，收集预测结果
        4. 算 accuracy
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)

                # ====== DEBUG: 第一个验证 batch 打印预测 vs 真实 ======
                if step == 0:
                    print(f"    [Eval Batch 1] Predicted: {preds[:10].tolist()}")
                    print(f"    [Eval Batch 1] Actual:    {batch['labels'][:10].tolist()}")
                    correct = (preds[:10] == batch['labels'][:10]).sum().item()
                    print(f"    [Eval Batch 1] Correct:   {correct}/{min(10, len(preds))}")

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        return accuracy_score(all_labels, all_preds)

    def _save_checkpoint(self):
        """保存模型 + tokenizer + label 映射"""
        save_dir = f"{self.config.system.output_dir}/checkpoints/best_model"

        # 保存模型权重
        self.model.save_pretrained(save_dir)

        # 保存 tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # 保存 label 映射
        if self.label2id:
            with open(f"{save_dir}/label_mapping.json", "w", encoding="utf-8") as f:
                json.dump(self.label2id, f, ensure_ascii=False, indent=2)

        # 保存训练配置（方便以后回顾）
        with open(f"{save_dir}/training_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(dict(self.config), f, allow_unicode=True)

        self.logger.info(f"Checkpoint saved to {save_dir}")

    def _plot_curves(self, losses, accs):
        """
        绘制 loss 和 accuracy 曲线

        两张图并排显示：
        - 左图：训练 Loss（应该下降）
        - 右图：验证 Accuracy（应该上升）
        """
        save_dir = f"{self.config.system.output_dir}/figures"
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(1, len(losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss 曲线
        ax1.plot(epochs, losses, "b-o")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # Accuracy 曲线
        ax2.plot(epochs, accs, "r-o")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
        self.logger.info(f"Training curves saved to {save_dir}/training_curves.png")
        plt.close()
