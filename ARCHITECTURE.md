# NLP Training Lab — 架构设计文档

> 🎯 **定位**：一个可复用的 NLP 模型训练实验框架，支持从文本分类、文本匹配到 LoRA/Qwen 微调的渐进式能力积累，同时为所有未来模型项目提供统一模板。
>
> 🟢 **当前阶段**：第一阶段 — BERT 中文新闻分类（4 分类）
>
> 👤 **读者**：AI/模型初学者（我自己）

---

## 📖 目录

1. [项目概述](#1-项目概述)
2. [核心概念先修](#2-核心概念先修)
3. [整体目录结构](#3-整体目录结构)
4. [数据流全景图](#4-数据流全景图)
5. [模块详解](#5-模块详解)
6. [类与接口设计](#6-类与接口设计)
7. [配置文件设计](#7-配置文件设计)
8. [扩展性设计](#8-扩展性设计)
9. [Checkpoint 与输出管理](#9-checkpoint-与输出管理)
10. [日志与可视化](#10-日志与可视化)
11. [第一阶段交付标准](#11-第一阶段交付标准)
12. [后续任务对接](#12-后续任务对接)
13. [常见问题 FAQ](#13-常见问题-faq)
14. [技术栈](#14-技术栈)

---

## 1. 项目概述

### 1.1 我们要做什么？

一句话：**用 BERT 模型对中文新闻进行分类**。

比如输入：
> "国足今晚迎战日本队"

输出：
> "类别：体育 (0.94)"

这就是「文本分类」—— NLP（自然语言处理）中最基础、最核心的任务之一。

整个训练链路：
```
原始文本 → 转成数字(Tokenize) → 送入BERT → 计算损失 → 反向传播 → 更新参数 → 保存模型
```

### 1.2 核心原则

| 原则 | 含义 | 为什么重要 |
|------|------|-----------|
| **极简起步** | V0.1 只有 Dataset + Model(factory) + Trainer + Inference，不提前抽象 | 先看到 loss 下降，再做工程 |
| **可配置** | 所有超参数、模型名、路径通过 YAML 注入 | 不改代码就能调参 |
| **可观测** | 自动记录 loss/accuracy 曲线、实验日志、Tensor Shape | 知道每层数据长什么样 |
| **可演进** | 目录结构和接口设计支持后续扩展 | 不用推翻重来 |

### 1.3 以后还要做什么？

| 阶段 | 任务 | 说明 |
|------|------|------|
| 🟢 **第一阶段（当前）** | BERT 新闻分类 | 文本分类，学完整流程 |
| 🟡 **第二阶段** | 文本匹配（Pair Match） | 判断两句话是否相似 |
| 🟠 **第三阶段** | LoRA 微调（Qwen-0.5B） | 用少量参数微调大模型 |
| 🔴 **第四阶段** | GIS 知识库（RAG + 微调模型） | 独立 Service 层，复用微调模型和推理接口 |

> 这份架构保证：**后面每个阶段，不需要重写项目，只需要增加模块**。

---

## 2. 核心概念先修

> 如果你是第一次做模型训练，先搞懂这些概念。

### 2.1 什么是 BERT？

BERT = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers

简单理解：
- 一个**预训练**的语言模型，Google 在 2018 年发布
- 在海量文本上先学了一遍（预训练）
- 我们只需在它的基础上**微调（Fine-tune）**就能做具体任务

就像：
> 一个读了万卷书的博士 → 稍微培训一下就能做新闻分类

### 2.2 什么是 Tokenizer（分词器）？

模型看不懂中文，需要转成数字编号：

```
"国足今晚迎战日本队"
        ↓   Tokenizer
[101, 3613, 5477, 4963, ..., 102]    ← 这叫 input_ids
```

Tokenizer 还会产生 `attention_mask`（告诉模型哪些位置是真正的文本、哪些是填充的）。

### 2.3 什么是 Dataset 和 DataLoader？

| 概念 | 类比 | 作用 |
|------|------|------|
| **Dataset** | 一本字典 | 定义"怎么读一条数据" |
| **DataLoader** | 自动翻页器 | 每次取一批（batch）送去训练 |

PyTorch 里这两个是分开的，**解耦**了"数据怎么读"和"数据怎么送"。

### 2.4 什么是 Batch（批次）？

不一次送全部数据，而是分小批：

```
10000 条新闻
    ↓  batch_size = 16
逐批送入模型，每批 16 条
    ↓
一共 625 批（= 10000 ÷ 16，每次训练循环走 625 步）
```

为什么要分批？因为 GPU 显存有限。

### 2.5 什么是 Loss（损失）？

模型预测和真实标签之间的差距。

> 模型预测：体育 0.8，政治 0.1，娱乐 0.1
> 真实标签：体育
> Loss（损失）：很小的数字 ✅ （因为猜对了）

训练目标就是**让 Loss 越来越小**。

### 2.6 什么是 Epoch（轮次）？

把全部训练数据过一遍 = 1 epoch。

> epochs = 3 表示：全部数据看 3 遍

### 2.7 什么是 Accuracy（准确率）？

```
预测正确的数量 ÷ 总数量 = Accuracy

比如 100 条预测对了 82 条 → Accuracy = 0.82
```

---

## 3. 整体目录结构

```
nlp-training-lab/
│
├── configs/                          # 🛠 配置文件目录
│   └── train.yaml                    #    主配置文件（所有超参数）
│
├── data/                             # 📦 数据目录
│   ├── raw/                          #    原始数据（下载后放这里，不动它）
│   │   └── news.csv                  #    新闻分类 CSV
│   ├── processed/                    #    清洗/预处理后的数据（可选）
│   └── splits/                       #    训练/验证/测试集划分（可选）
│
├── experiments/                      # 🔬 实验记录（科研思维！）
│   ├── exp001_bert_base/             #    每个实验独立目录
│   └── .gitkeep
│
├── src/                              # 🧠 核心代码
│   │
│   ├── datasets/                     #    数据加载模块
│   │   └── news_dataset.py          #        Dataset 定义（CSV → Tensor）
│   │
│   ├── models/                       #    模型定义模块
│   │   └── factory.py               #        模型工厂（AutoModel 一行返回）
│   │
│   ├── trainers/                     #    训练引擎模块
│   │   └── trainer.py               #        训练循环 + 验证（核心！）
│   │
│   ├── debug/                        # 🔍 Tensor Shape 追踪
│   │   └── shape_tracker.py         #        打印每层 tensor shape
│   │
│   ├── utils/                        #    工具模块
│   │   ├── seed.py                  #        固定随机种子（可复现）
│   │   └── logger.py                #        日志记录（控制台+文件）
│   │
│   └── inference/                    #    推理模块
│       └── predict.py               #        单条/批量推理脚本
│
├── outputs/                          # 📊 输出目录（自动生成）
│   ├── checkpoints/
│   │   └── best_model/               #    验证集最优模型
│   ├── logs/
│   │   └── train.log                 #    训练日志文件
│   └── figures/
│       ├── loss_curve.png
│       └── accuracy_curve.png
│
├── train.py                          # 🚀 入口：启动训练（含 validate）
├── evaluate.py                       # 📋 入口：模型评估
├── requirements.txt                  # 📄 依赖清单
├── README.md                         # 📖 项目说明（含实验记录）
└── ARCHITECTURE.md                   # 📖 架构文档（本文）
```

### 目录职责一句话总结

| 目录 | 用途 | 谁管它 |
|------|------|--------|
| `configs/` | 放配置文件，不写死超参数 | 手动编辑 |
| `data/` | 所有数据文件 | 脚本生成 |
| `src/` | 核心 Python 代码 | 手动编写 |
| `outputs/` | 训练产出物（模型、日志、图表） | 脚本自动生成 |

> **核心原则：代码和数据分离，配置和逻辑分离。**

---

## 4. 数据流全景图

> 这是整个项目**最重要的一张图**——理解了这个数据流，就理解了全部代码。

### 训练数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            训练数据流                                     │
│                                                                          │
│   CSV 文件                                                                │
│   (原始数据: text + label)                                                │
│      │                                                                   │
│      ▼                                                                   │
│   NewsDataset.__init__()         ← 读取数据、加载 Tokenizer              │
│      │                                                                   │
│      ▼                                                                   │
│   NewsDataset.__getitem__(idx)   ← 取一条，Tokenize，返回 Tensor         │
│      │                                                                   │
│      ▼                                                                   │
│   DataLoader                     ← 自动打包成 batch + shuffle            │
│      │                                                                   │
│      ▼                                                                   │
│   BertClassifier.forward()       ← 前向传播，输出 logits                │
│      │                                                                   │
│      ▼                                                                   │
│   CrossEntropyLoss               ← 计算 loss（预测和真实标签的差距）     │
│      │                                                                   │
│      ▼                                                                   │
│   loss.backward()                ← 反向传播（计算每个参数的梯度）        │
│      │                                                                   │
│      ▼                                                                   │
│   optimizer.step()               ← 更新参数，让 loss 下降               │
│      │                                                                   │
│      ▼                                                                   │
│   Evaluator (val)                ← 每个 epoch 结束后验证                 │
│      │                                                                   │
│      ▼                                                                   │
│   Save Checkpoint                ← val_accuracy 创新高时保存             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 推理数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            推理数据流                                     │
│                                                                          │
│   用户输入："国足今晚迎战日本队"                                          │
│      │                                                                   │
│      ▼                                                                   │
│   Tokenizer → input_ids + attention_mask                                │
│      │                                                                   │
│      ▼                                                                   │
│   加载模型 best_model/ ← 从 checkpoint 恢复                              │
│      │                                                                   │
│      ▼                                                                   │
│   model.eval() + torch.no_grad()  ← 关闭梯度计算，省显存+加速            │
│      │                                                                   │
│      ▼                                                                   │
│   前向传播 → logits (每个类别的分数)                                     │
│      │                                                                   │
│      ▼                                                                   │
│   argmax → 找到分数最高的类别                                            │
│      │                                                                   │
│      ▼                                                                   │
│   id → label 映射                                                        │
│      │                                                                   │
│      ▼                                                                   │
│   输出："类别：体育 (0.94)"                                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键数据格式变化

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| CSV 源文件 | `"国足今晚迎战日本队",体育` | 文本字符串 + 标签字符串 | 人类可读 |
| Dataset 输出 | 一条文本 | `{"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}` | 机器可读 |
| DataLoader 输出 | N 条文本打包 | 同上，但第 0 维是 batch | 送模型 |
| 模型输出 | Tensor batch | `logits` shape=(batch_size, num_classes) | 每个类别的分数 |
| 推理输出 | 一句文本 | "体育 (0.94)" | 人类可读 |

---

## 5. 模块详解

### 5.1 配置系统 (`configs/`)

#### 为什么要用 YAML 配置文件？

新手常犯的错误：**超参数硬编码在代码里**。

```python
# ❌ 错误做法 — 每次改参数都要改代码
batch_size = 16
learning_rate = 2e-5
```

```yaml
# ✅ 正确做法 — 所有参数集中在 configs/train.yaml
batch_size: 16
learning_rate: 2e-5
```

#### 配置加载方式

使用 **OmegaConf**（比普通 yaml 更强大，支持类型检查、命令行覆盖）：

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/train.yaml")

# 使用（支持点号访问和自动补全）
model_name = cfg.model.model_name
batch_size = cfg.training.batch_size
device = cfg.system.device

# 命令行覆盖（训练时临时改参数）
# python train.py training.batch_size=32
```

#### 配置分类（详见第 7 节）

所有参数按功能分组：
- `experiment` — 实验名称、随机种子
- `data` — 数据路径、文本列名、标签列名、最大长度
- `model` — 模型名称、分类数
- `training` — batch size、学习率、epochs、优化器
- `system` — 输出目录、设备、并行工作数
- `logging` — 日志级别、是否画图

---

### 5.2 数据层 (`src/datasets/`)

**一句话职责：把原始数据变成模型能吃的 Tensor。**

#### 核心文件：`news_dataset.py`

```python
class NewsDataset(Dataset):
    """
    新闻分类数据集

    流程：
    1. __init__: 读取 CSV，加载 Tokenizer，建立 label↔id 映射
    2. __getitem__: 取出一条文本 → tokenize → 返回 Tensor 字典
    """

    def __init__(self, csv_path, tokenizer, max_length, text_col, label_col, label2id=None):
        # 1. 用 Pandas 读取 CSV
        # 2. 保存 tokenizer 和参数
        # 3. 如果 label2id 未提供，从数据中自动构建（首次）
        # 4. 保存 id2label（推理时要用）
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 取第 idx 行
        # 2. tokenizer(text, padding, truncation, max_length)
        # 3. 返回标准化字典
        return {
            "input_ids":       Tensor,  # (seq_len,) 文本转成的数字编号
            "attention_mask":  Tensor,  # (seq_len,) 1=真实文本，0=填充
            "labels":          Tensor   # (1,) 类别数字标签
        }
```

#### 输入/输出示例

**CSV 输入**（`data/raw/news.csv`）：
```csv
text,label
"国足今晚迎战日本队",体育
"美国大选最新进展",政治
"iPhone17正式发布",科技
"欧冠决赛皇马夺冠",体育
```

**Dataset 输出**（`__getitem__` 返回的 Tensor 字典）：
```python
{
    "input_ids": tensor([101, 3613, 5477, 4963, ..., 102]),    # shape: (128,)
    "attention_mask": tensor([1, 1, 1, 1, ..., 0, 0]),         # shape: (128,)
    "labels": tensor(2)                                         # 体育→2（数字标签）
}
```

#### 设计要点

| 设计点 | 为什么 |
|--------|--------|
| `label2id` 由外部传入 | 保证训练集和测试集用同一套映射 |
| 返回 dict 格式 | 和 Transformers 模型的输入格式完全一致 |
| Tokenizer 在 Dataset 里做 | 每条数据只转一次，不重复 |
| `text_col` / `label_col` 可配置 | 不同 CSV 格式不用改代码 |

#### 未来扩展

| 阶段 | 数据集类 | 变化 |
|------|---------|------|
| 文本匹配 | `PairDataset` | `__getitem__` 返回 (text_a, text_b) 两组 Tensor |
| LoRA 微调 | 复用 `NewsDataset` | 无需改造 |
| Qwen 微调 | `InstructionDataset` | 需处理 ChatML 对话格式 |

---

### 5.3 模型层 (`src/models/`)

**一句话职责：返回一个模型，够用就行。**

V0.1 没有任何封装，直接返回 HuggingFace 的 `AutoModelForSequenceClassification`。

#### 为什么不用 BertClassifier？

因为当前阶段的目标是**理解训练流程**，不是设计模型抽象层。

```
V0.1:    factory.py → AutoModelForSequenceClassification  ← 先跑起来
V0.2+:   bert_classifier.py → 再加封装                    ← 等需要复用时
LoRA:    lora_model.py → 再加 peft                       ← 自然扩展
```

过早抽象的风险：
- 花 2 小时设计架构，花 10 分钟训练
- **本末倒置**

#### 核心文件：`factory.py`

```python
from transformers import AutoModelForSequenceClassification

def build_model(model_name: str, num_classes: int, device: str):
    """
    构建模型

    就一行：AutoModelForSequenceClassification

    参数：
        model_name: "bert-base-chinese" 或后续的 RoBERTa/Qwen
        num_classes: 分类数（如 4）
        device: "cuda" 或 "cpu"

    返回：
        model (已移到 device 上)
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    return model.to(device)
```

#### 为什么这样够用？

| 原方案（BertClassifier） | 现方案（factory.py） |
|------------------------|---------------------|
| 一个类，封装 forward | 一个函数，返回 model |
| 要维护 | 一行代码 |
| 更适合第二版 | 更适合第一版 |

等做到 LoRA 时，再在这个目录里加 `lora_model.py`。

---

### 5.4 训练器 + 验证 (`src/trainers/`)

**一句话职责：训练 + 验证，都在这里。V0.1 不拆分 Evaluator。**

为什么验证不单独拆文件？
- 第一个项目只有 Accuracy 一个指标
- `validate()` 写在 trainer.py 里只有 20 行
- **等文本匹配（第二阶段）再抽成独立 Evaluator**

#### 核心文件：`trainer.py`

```python
class Trainer:
    """
    训练器

    职责：
    1. 训练循环（epoch → batch → forward → backward → step）
    2. 验证循环（每个 epoch 结束后算 accuracy）
    3. 保存最佳模型
    4. 记录日志 + 画图
    """

    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        # 优化器（AdamW 是 BERT 微调的标配）
        self.optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

        # 学习率调度器（先用 warmup，再线性衰减）
        total_steps = len(train_loader) * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """
        完整训练流程

        伪代码：
        for epoch in 1..epochs:
            for batch in train_loader:          # 训练
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            val_acc = validate(val_loader)       # 验证

            if val_acc > best:
                save_checkpoint()                # 保存最佳
        """
        best_acc = 0
        train_losses, val_accs = [], []

        for epoch in range(1, self.config.training.epochs + 1):
            # ====== 训练 ======
            self.model.train()
            epoch_loss = 0

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
                        f"Epoch {epoch} | Step {step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )

            avg_loss = epoch_loss / len(self.train_loader)

            # ====== 验证（就 20 行，不拆文件）=======
            val_acc = self._validate()

            self.logger.info(
                f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | "
                f"Val Accuracy: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint()
                self.logger.info(f"★ New best! Accuracy: {val_acc:.4f}")

            train_losses.append(avg_loss)
            val_accs.append(val_acc)

        self._plot_curves(train_losses, val_accs)
        return {"best_accuracy": best_acc}

    def _validate(self):
        """
        验证（V0.1 简易版）

        model.eval() + torch.no_grad() → 算 accuracy
        """
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        from sklearn.metrics import accuracy_score
        return accuracy_score(all_labels, all_preds)

    def _save_checkpoint(self):
        """保存模型 + tokenizer + label 映射"""
        save_dir = f"{self.config.system.output_dir}/checkpoints/best_model"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        import json, yaml
        with open(f"{save_dir}/label_mapping.json", "w") as f:
            json.dump(self.label2id, f, ensure_ascii=False)
        with open(f"{save_dir}/training_config.yaml", "w") as f:
            yaml.dump(dict(self.config), f)
        self.logger.info(f"Model saved to {save_dir}")

    def _plot_curves(self, losses, accs):
        """绘制 loss 和 accuracy 曲线"""
        import matplotlib.pyplot as plt
        epochs = range(1, len(losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, losses, "b-o")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax2.plot(epochs, accs, "r-o")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(f"{self.config.system.output_dir}/figures/training_curves.png")
        self.logger.info("Curves saved to outputs/figures/")

---

### 5.5 Tensor Shape 追踪 (`src/debug/`)

> 这是新手最容易忽略、也最有帮助的东西。

**为什么需要 Shape Tracking？**

新手做模型训练时，最大的困惑是数据在每一层之后变成了什么形状。

有了 Shape Tracking，你能**亲眼看到**数据流经模型的过程：

```
=== Shape Track [embedding] ===
input_ids:         [16, 128]
attention_mask:    [16, 128]
    ↓ Embedding
hidden_states:     [16, 128, 768]    ← 每个字变成了768维向量
    ↓ BERT Encoder
pooled_output:     [16, 768]         ← 整篇新闻压缩成768维向量
    ↓ Classifier
logits:            [16, 4]           ← 4个类别的分数
```

一下就理解了：batch=16, max_len=128, hidden=768, classes=4。

#### 核心文件：`debug/shape_tracker.py`

```python
class ShapeTracker:
    """Tensor Shape 追踪器"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.records = []

    def track(self, stage_name, **tensors):
        """记录一个阶段的 tensor shape"""
        if not self.enabled:
            return
        shapes = {k: list(v.shape) for k, v in tensors.items()}
        self.records.append({"stage": stage_name, "shapes": shapes})

    def print_summary(self):
        """打印所有记录的 shape"""
        if not self.enabled or not self.records:
            return
        print("\n" + "=" * 60)
        print("📐 Shape Track Summary")
        print("=" * 60)
        for rec in self.records:
            print(f"\n  [{rec['stage']}]")
            for name, shape in rec["shapes"].items():
                print(f"    {name:20s} {str(shape):20s}")
        print("=" * 60)
```

使用方式（在 trainer.py 中插桩）：
```python
tracker = ShapeTracker()
tracker.track("输入", input_ids=batch["input_ids"])
# ... 模型 forward 之后 ...
tracker.track("Logits", logits=outputs.logits)
tracker.print_summary()
```

---

### 5.6 工具模块 (`src/utils/`)

#### `seed.py` — 随机种子管理

```python
def set_seed(seed: int):
    """固定所有随机种子，保证实验结果可复现"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性算法（牺牲一点速度，保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

> 为什么必须固定种子？模型训练有大量随机性（参数初始化、shuffle、dropout），不固定的话同样代码跑两次结果不同，**没法判断改动是否有效**。

#### `logger.py` — 日志工具

```python
def setup_logger(name, log_file, level=logging.INFO):
    """
    设置同时输出到控制台和文件的日志

    输出格式：[2026-06-22 14:30:00] Epoch 1 | Train Loss: 1.21 | LR: 2e-5
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件 Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

**日志输出示例**：
```
2026-06-22 10:00:01 | INFO | ====== Epoch 1/3 ======
2026-06-22 10:01:20 | INFO | Train Step 10 | Loss: 1.5432 | LR: 2.0e-05
2026-06-22 10:01:25 | INFO | Train Step 20 | Loss: 1.2100 | LR: 2.0e-05
2026-06-22 10:02:00 | INFO | Epoch 1 | Train Loss: 1.2134 | Val Accuracy: 0.8215
2026-06-22 10:02:00 | INFO | ★ New best model saved with accuracy 0.8215
```

同时保存到 `outputs/logs/train.log`。

> **V0.1 没有独立的 metrics.py**。指标只在 trainer.py 的 `_validate()` 中用 `sklearn.metrics.accuracy_score` 一行算完。
>
> 等第二阶段（文本匹配）需要 Precision/Recall/F1 时再抽成独立文件。

---

### 5.7 推理模块 (`src/inference/`)

**一句话职责：加载训练好的模型，对输入文本进行预测。**

#### 核心文件：`predict.py`

```python
class Predictor:
    """
    推理器（V0.1 极简版）

    职责：
    1. 加载 best_model checkpoint
    2. 预测 → 返回类别 + 置信度
    """

    def __init__(self, checkpoint_dir, model_name, num_classes, device):
        # 直接加载（没有 BertClassifier 包装）
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.device = device

        # 加载 label 映射
        import json
        with open(f"{checkpoint_dir}/label_mapping.json") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

    def predict(self, text: str) -> dict:
        """
        单条文本预测

        流程：
        1. Tokenize → input_ids + attention_mask
        2. torch.no_grad() 推理
        3. softmax → argmax → id2label

        返回：{"label": "体育", "confidence": 0.94}
        """
        inputs = self.tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)

        return {
            "label": self.id2label[pred_id.item()],
            "confidence": round(confidence.item(), 4)
        }
```

#### 使用方式（最终检验）

```bash
# 单条预测
python src/inference/predict.py --text "国足今晚迎战日本队"

# 输出：
# 文本：国足今晚迎战日本队
# 类别：体育 (0.94)
```

> 这条命令跑通了，就说明整个训练项目成功了。

---

### 5.8 入口脚本 (`train.py` / `evaluate.py`)

#### `train.py` — 训练入口（含加载 → 训练 → 验证 → 保存）

```python
"""
训练入口脚本

V0.1 职责：组装一切，然后交给 Trainer

流程：
1. OmegaConf.load("configs/train.yaml")     ← 加载配置
2. set_seed(cfg.experiment.seed)            ← 固定种子
3. NewsDataset → DataLoader                 ← 加载数据
4. build_model(...)                          ← 创建模型（factory.py）
5. Trainer(model, loaders, config).train()   ← 训练 + 验证 + 保存
"""
```

注意：V0.1 的 evaluate.py 暂时不需要独立文件，因为验证已经在 trainer.train() 里做了。
等第二阶段（文本匹配）需要更复杂的评估时再补。

---

## 6. 类与接口设计

### 6.1 V0.1 类关系总览

> V0.1 只有 3 个核心类 + 1 个工厂函数，没有多余抽象。

```
            ┌─────────────┐
            │  train.yaml │  ← 配置文件
            └──────┬──────┘
                   │ OmegaConf.load()
                   ▼
            ┌─────────────┐
            │  train.py   │  ← main函数：组装所有组件
            └──┬──┬──┬────┘
               │  │  │
       ┌───────┘  │  └──────────┐
       ▼          ▼             ▼
┌──────────┐ ┌──────────┐ ┌───────────┐
│NewsDataset│ │build_model│ │  Trainer  │  ← 核心（含_validate）
└──────────┘ │(factory) │ └─────┬─────┘
             └──────────┘       │
                                ▼
                          ┌───────────┐
                          │ShapeTracker│  ← debug 用
                          └───────────┘

依赖方向：train.py → Trainer → Model (factory), Dataset, Logger, ShapeTracker
```

### 6.2 详细类图

```
┌─────────────────────────────────────────────────────────────────┐
│                        NewsDataset                              │
├─────────────────────────────────────────────────────────────────┤
│ - df: DataFrame              # CSV 数据                         │
│ - tokenizer: AutoTokenizer   # 分词器                           │
│ - max_length: int            # 最大序列长度                     │
│ - label2id: Dict[str,int]    # 标签→数字映射                    │
│ - id2label: Dict[int,str]    # 数字→标签映射                    │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(csv_path, tokenizer, max_length, text_col,           │
│            label_col, label2id)                                  │
│ + __len__() → int                                               │
│ + __getitem__(idx) → {"input_ids", "attention_mask", "labels"}  │
└─────────────────────────────────────────────────────────────────┘
                             │ 被 DataLoader 包装
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  build_model() — 工厂函数                         │
│  (不是类！只是一个函数)                                          │
├─────────────────────────────────────────────────────────────────┤
│ 输入: model_name, num_classes, device                           │
│ 逻辑: AutoModelForSequenceClassification.from_pretrained(...)    │
│ 输出: model (HuggingFace 原生模型)                               │
└─────────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                           Trainer                               │
├─────────────────────────────────────────────────────────────────┤
│ - model: AutoModelForSequenceClassification                     │
│ - train_loader: DataLoader                                      │
│ - val_loader: DataLoader                                        │
│ - optimizer: AdamW                                              │
│ - scheduler: get_linear_schedule_with_warmup                    │
│ - logger: Logger                                                │
│ - config: DictConfig                                            │
│ - best_acc: float                                               │
├─────────────────────────────────────────────────────────────────┤
│ + train() → Dict                                                │
│ - _validate() → float           # V0.1: 在Trainer内部           │
│ - _save_checkpoint()            # 保存模型+tokenizer+label映射   │
│ - _plot_curves(losses, accs)    # loss/accuracy曲线             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       ShapeTracker                              │
│  (debug 工具，可插拔)                                            │
├─────────────────────────────────────────────────────────────────┤
│ + track(stage_name, **tensors)   # 记录 tensor shape           │
│ + print_summary()                # 打印所有 shape              │
└─────────────────────────────────────────────────────────────────┘
```

> **对比原版**：V0.1 删除了 Evaluator 类、BertClassifier 类，把 validate() 收进 Trainer。
> 等第二阶段再抽出来。

┌─────────────────────────────────────────────────────────────────┐
│                         Predictor                               │
├─────────────────────────────────────────────────────────────────┤
│ - model: BertClassifier                                         │
│ - tokenizer: AutoTokenizer                                      │
│ - id2label: Dict[int, str]                                      │
│ - device: str                                                   │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(checkpoint_dir, ...)                                 │
│ + predict(text) → {"label": str, "confidence": float}           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 配置文件设计

### `configs/train.yaml`

```yaml
# ============================================================
# NLP Training Lab — 训练配置文件
# ============================================================
# 使用：python train.py --config configs/train.yaml
# 覆盖：python train.py training.batch_size=32

# ---------- 实验元信息 ----------
experiment:
  name: "news_classification_bert_base"   # 实验名称（用于区分不同实验）
  seed: 42                                 # 随机种子（保证可复现）

# ---------- 数据配置 ----------
data:
  csv_path: "data/raw/news.csv"          # CSV 数据文件路径
  text_col: "text"                        # 文本列名
  label_col: "label"                      # 标签列名
  num_labels: 4                           # 分类数量
  max_length: 128                         # 最大序列长度（越长越吃显存）
  test_size: 0.2                          # 测试集比例
  val_size: 0.1                           # 验证集比例（从训练集划分）

# ---------- 模型配置 ----------
model:
  model_name: "bert-base-chinese"         # 预训练模型名称
                                          # 可选：
                                          #   - "bert-base-chinese"
                                          #   - "hfl/chinese-roberta-wwm-ext"
                                          #   - "hfl/chinese-macbert-base"

# ---------- 训练配置 ----------
training:
  batch_size: 16                          # 每批样本数（显存不够时减半：8 或 4）
  epochs: 3                               # 训练轮数（BERT 微调通常 2~5）
  learning_rate: 2.0e-5                   # 学习率（BERT 微调通常 1e-5 ~ 5e-5）
  adam_epsilon: 1e-8                      # AdamW 优化器的 epsilon
  weight_decay: 0.01                      # 权重衰减（L2 正则化，防过拟合）
  max_grad_norm: 1.0                      # 梯度裁剪最大值（防梯度爆炸）
  warmup_ratio: 0.1                       # 学习率预热比例（前 10% 的 step 逐渐升温）
  early_stopping_patience: 2              # 早停耐心值（0=不使用）

# ---------- 系统配置 ----------
system:
  output_dir: "outputs"                   # 输出根目录
  device: "auto"                          # "auto"=有 GPU 用 GPU，否则 CPU
  num_workers: 2                          # DataLoader 并行加载数

# ---------- 日志与可视化 ----------
logging:
  log_level: "INFO"                       # 日志级别
  log_interval: 10                        # 每多少步打印一次日志
  save_figures: true                      # 是否自动保存训练曲线
```

### 参数调整指南

| 参数 | 路径 | 调大 → | 调小 → | 新手建议 |
|------|------|--------|--------|----------|
| batch_size | `training.batch_size` | 训练稳定，但吃显存 | 震荡大，但省显存 | 先 16，不够减到 8 |
| learning_rate | `training.learning_rate` | 学得快，但可能发散 | 学得慢，但稳定 | BERT 固定 2e-5 |
| epochs | `training.epochs` | 可能过拟合 | 可能欠拟合 | 先试 3 |
| max_length | `data.max_length` | 保留更多信息，但吃显存 | 可能截断重要信息 | 中文新闻 128 够用 |
| seed | `experiment.seed` | — | — | 固定 42，不要改 |

---

## 8. 扩展性设计

### 8.1 核心理念：架构与任务解耦

这套架构的核心思想是：**不同任务只插拔不同的 Dataset 和 Model，Trainer 保持稳定。**

```
              ┌──────────────┐
              │   configs/   │  ← 每个任务一份配置
              └──────┬───────┘
                     │
┌──────────────────────────────────────────┐
│                src/                      │
│                                          │
│  datasets/   ───  每个任务一个 Dataset 类 │
│  models/     ───  先 factory，后逐步抽象  │
│  trainers/   ───  ⭐ V0.1 先简易，再加固 │
│  debug/      ───  可插拔 shape tracker   │
│  inference/  ───  每个任务一个 Predictor  │
└──────────────────────────────────────────┘
                     │
              ┌──────┴──────┐
              │  outputs/   │  ← 每个任务独立输出
              └─────────────┘
```

### 8.2 为什么 Trainer 能复用？

因为所有模型的接口一致：

| 组件 | 输入 | 输出 |
|------|------|------|
| 分类模型 (BERT) | `input_ids`, `attention_mask`, `labels` | `loss`, `logits` |
| 匹配模型 (Pair) | 同上 | `loss`, `logits` |
| LoRA 模型 | 同上 | `loss`, `logits` |

只要遵守这个接口约定，Trainer 不用改一行代码。

### 8.3 各阶段的扩展路径

#### 第二阶段：文本匹配

| 变动 | 说明 |
|------|------|
| 新增 `src/datasets/pair_dataset.py` | `__getitem__` 返回双句 (text_a, text_b) 的 Tensor |
| 配置新增 `data.task_type: "pair"` | Trainer 根据该字段路由不同 Dataset |
| 模型不变 | `AutoModelForSequenceClassification` 原生支持双句 |
| 从 Trainer 抽出 Evaluator | 验证逻辑独立成类，增加 Precision/Recall/F1 |

#### 第三阶段：LoRA 微调（Qwen-0.5B）

| 变动 | 说明 |
|------|------|
| 新建 `src/models/qwen_lora_model.py` | 内部使用 `peft.get_peft_model()` 注入 LoRA，仍保持 `forward()` 接口 |
| 新建 `src/datasets/instruction_dataset.py` | 将 (question, answer) 封装为 ChatML 格式 |
| Trainer 微调 | 继承 Trainer 并重写 `_train_epoch()` 的 loss 获取方式（因果 LM loss 不同） |
| 配置增加 | `peft.r`, `peft.lora_alpha`, `peft.target_modules` |

#### 第四阶段：Qwen 全模型微调 / RAG 知识库

| 变动 | 说明 |
|------|------|
| 可接入 `accelerate` 或 `deepspeed` | 目录结构不变，新增 `configs/deepspeed.yaml` |
| 新增独立的 Service 层 | 复用微调模型和推理接口 |

> 每一阶段完成后，项目仍保持目录清晰，可单独演示，也可整体演进。

---

## 9. Checkpoint 与输出管理

### 9.1 保存策略

| 策略 | 说明 |
|------|------|
| 触发条件 | `val_accuracy` 创新高时 |
| 保存位置 | `outputs/checkpoints/best_model/` |
| 保存内容 | 模型权重 + Tokenizer + 配置 + Label 映射 |
| 保留数量 | 只保留最好的一个 |

### 9.2 保存目录结构

```
outputs/
├── checkpoints/
│   └── best_model/              # 验证集最优模型（可直接用于推理）
│       ├── pytorch_model.bin    # 模型权重（~400 MB）
│       ├── config.json          # 模型配置
│       ├── tokenizer.json       # 分词器
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       └── label_mapping.json   # label ↔ id 映射
├── logs/
│   └── train.log                # 训练日志
└── figures/
    ├── loss_curve.png           # Loss 曲线
    └── accuracy_curve.png       # Accuracy 曲线
```

### 9.3 断点续训（可选）

第一阶段暂不实现，但预留了设计：
```python
# 未来可保存 training_state.pt
torch.save({
    "epoch": epoch,
    "optimizer_state_dict": optimizer.state_dict(),
    "best_acc": best_acc,
}, f"{checkpoint_dir}/training_state.pt")
```

---

## 10. 实验记录 (`experiments/`)

> 半年后你会忘了当时用了什么参数。但 `experiments/exp001_bert_base/` 还能看。

### 10.1 为什么需要实验目录？

| 不记录 | 记录 |
|--------|------|
| 跑完了就忘 | 随时回溯 |
| 不知道哪个参数好 | 对比可见 |
| 结论丢失 | README 里写着"调大 batch_size 后 accuracy 提升 2%" |

### 10.2 每轮实验的结构

```
experiments/
├── exp001_bert_base/        ← 实验编号 + 简短描述
│   ├── config.yaml          ← 本次实验用的配置（备份）
│   ├── results.md           ← 实验结果记录
│   ├── train.log            ← 训练日志
│   └── figures/             ← loss/accuracy 曲线
│       ├── loss_curve.png
│       └── accuracy_curve.png
├── exp002_roberta/
└── exp003_lora/
```

### 10.3 实验记录模板 (`experiments/exp001_bert_base/results.md`)

```markdown
## Experiment 001 — BERT Base 中文新闻分类

### 基本信息

- **日期**: 2026-06-22
- **模型**: bert-base-chinese
- **数据**: data/raw/news.csv (800 条, 4 分类)
- **配置**: 见 config.yaml

### 超参数

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| learning_rate | 2e-5 |
| max_length | 128 |
| epochs | 3 |
| seed | 42 |

### 结果

| 指标 | 值 |
|------|-----|
| 训练 Loss | 0.31 |
| 验证 Accuracy | 0.89 |

### Loss / Accuracy 曲线

![loss_curve](figures/loss_curve.png)
![accuracy_curve](figures/accuracy_curve.png)

### 结论

- ✅ 模型成功收敛，accuracy 达到 0.89
- ✅ 测试集推理准确

### 下一步

- [ ] 尝试 hfl/chinese-roberta-wwm-ext
- [ ] 增大 batch_size 到 32

### 笔记

第一次完整跑通训练流程。
```

---

## 11. 日志与可视化

### 10.1 日志格式

```
2026-06-22 10:00:01 | INFO | ====== Epoch 1/3 ======
2026-06-22 10:01:20 | INFO | Train Step 10 | Loss: 1.5432 | LR: 2.0e-05
2026-06-22 10:01:25 | INFO | Train Step 20 | Loss: 1.2100 | LR: 2.0e-05
2026-06-22 10:02:00 | INFO | Epoch 1 | Train Loss: 1.2134 | Val Accuracy: 0.8215
2026-06-22 10:02:00 | INFO | ★ New best model saved with accuracy 0.8215
2026-06-22 10:03:15 | INFO | ====== Epoch 2/3 ======
2026-06-22 10:04:20 | INFO | Train Step 10 | Loss: 0.8921 | LR: 1.8e-05
...
2026-06-22 10:06:00 | INFO | Training complete. Best accuracy: 0.8912
```

### 10.2 可视化曲线

训练结束后自动生成两张图，保存在 `outputs/figures/`：

```
Loss 曲线（应该下降）：          Accuracy 曲线（应该上升）：
                              ╱
 2.0┤╲                       0.9┤╱      ╱
 1.5┤ ╲                      0.8┤╱    ╱
 1.0┤  ╲                     0.7┤  ╱╱
 0.5┤   ╲                    0.6┤╱
    └───┬───                      └───┬───
        1   2   3                     1   2   3
           Epoch                          Epoch
```

**怎么看这两张图？**
- Loss 下降 + Accuracy 上升 ✅ → 模型在正常学习
- Loss 下降但 Accuracy 也下降 ❌ → 有问题（可能 label 错了）
- Loss 下降后反弹 ⚠️ → 过拟合，应该早停或用更多数据
- Loss 不动 ❌ → 学习率可能不对

---

## 11. 第一阶段交付标准

在完成代码实现后，项目应当达到以下标准：

### 11.1 功能标准

| # | 标准 | 验证方式 |
|---|------|----------|
| 1 | `python train.py` 可完整跑通训练 | 从 `data/raw/news.csv` 开始到训练完成 |
| 2 | Loss 曲线和 Accuracy 曲线自动保存 | `outputs/figures/` 下有 png 文件 |
| 3 | 最佳模型自动保存 | `outputs/checkpoints/best_model/` 存在 |
| 4 | 推理脚本可用 | `python predict.py --text "..."` 返回正确类别 |
| 5 | Tensor Shape 打印正常 | 训练时能看到每层的 shape 输出 |

### 11.2 代码质量标准

| # | 标准 | 说明 |
|---|------|------|
| 1 | 所有超参数在 YAML 中，不硬编码 | configs/train.yaml |
| 2 | 只有 4 个核心文件 | news_dataset.py, factory.py, trainer.py, predict.py |
| 3 | 随机种子固定 | 两次运行结果一致 |
| 4 | 日志同时输出控制台和文件 | outputs/logs/train.log |
| 5 | 实验记录模板就绪 | experiments/ 目录存在，README 有记录模板 |

### 11.3 V0.1 成功标志

```
执行 python train.py

输出：
[2026-06-22 14:30:00] Epoch 1 | Step 0 | Loss: 1.58 | LR: 2.0e-05
[2026-06-22 14:30:05] Epoch 1 | Step 10 | Loss: 1.12 | LR: 2.0e-05
[2026-06-22 14:30:10] Epoch 1 | Step 20 | Loss: 0.74 | LR: 2.0e-05
...
★ New best! Accuracy: 0.87

📐 Shape Track Summary:
  [输入]        input_ids:       [16, 128]
  [Logits]      logits:          [16, 4]
```

> **第一次看到 loss 从 1.58 → 0.74 → 0.31，你就正式进入模型训练的世界了。**

---

## 12. 后续任务对接

| 阶段 | 任务 | 新增/改动模块 |
|------|------|-------------|
| 2 | 文本匹配（Pair Match） | `pair_dataset.py`，`data.task_type` 路由 |
| 3 | LoRA 微调 Qwen-0.5B | `qwen_lora_model.py`，`instruction_dataset.py`，扩展 Trainer |
| 4 | GIS 知识库（RAG + 微调） | 独立 Service 层，复用微调模型和推理接口 |

每一阶段完成后：
1. 新增的代码放在独立文件中，不修改已有模块的核心逻辑
2. 新增一份对应的配置文件（如 `configs/train_lora.yaml`）
3. 更新 README 和 ARCHITECTURE 文档

---

## 13. 常见问题 FAQ

### Q1: 我没有 GPU，能跑吗？

**能。** 代码会自动检测：
- 有 GPU → 用 CUDA 加速
- 没有 → 自动用 CPU

但 BERT 在 CPU 上训练会很慢（一次可能要数小时）。
**建议先用少量数据（1000 条）测试流程能否跑通。**

### Q2: 显存不够怎么办？

| 方法 | 效果 |
|------|------|
| 减小 `training.batch_size` | 16 → 8 → 4 → 2 |
| 减小 `data.max_length` | 128 → 64 |
| 关 GPU 用 CPU | `system.device: "cpu"` |

### Q3: 怎么判断模型训练好了？

看两条曲线：

| 现象 | 判断 | 怎么办 |
|------|------|--------|
| Loss ↓, Accuracy ↑ | ✅ 正常学习 | 不用管 |
| Loss ↓, 但 Accuracy 也 ↓ | ❌ 有问题 | 检查数据标注 |
| Loss 先降后升 | ⚠️ 过拟合 | 减少 epochs 或增大 weight_decay |
| Loss 几乎不动 | ❌ 没学到 | 检查学习率是否太大/太小 |

### Q4: 训练完怎么用？

```bash
python src/inference/predict.py --text "国足今晚迎战日本队"
# 输出：类别：体育 (0.94)
```

### Q5: 以后加新任务怎么加？

每阶段三步走：
1. **新建 Dataset 类** — 定义新数据怎么读
2. **新建 Model 类**（或复用旧类）— 定义新模型结构
3. **新建配置文件** — 定义新参数

Trainer 和 Evaluator **尽量不改**。

### Q6: 什么是过拟合（Overfitting）？

模型死记硬背了训练数据，没学会"举一反三"。

**现象**：训练 loss 持续下降，但验证 accuracy 先升后降。

**解决办法**：
- 减少 epochs
- 增大 `weight_decay`
- 用更多数据
- 减小模型

### Q7: 为什么 BERT 微调的学习率这么小（2e-5）？

BERT 已经在大规模数据上预训练好了，参数处于一个很好的位置。
学习率太大就会把它"震"出好位置。

> 你已经坐在最舒服的位置上，稍微调整一下就好（小学习率）
> 而不是站起来换个位置（大学习率）

---

## 14. 技术栈

| 工具 | 用途 | 版本要求 |
|------|------|----------|
| **Python** | 编程语言 | 3.10+ |
| **PyTorch** | 深度学习框架 | 2.0+ |
| **Transformers** | HuggingFace 模型库 | 4.30+ |
| **Datasets** | HuggingFace 数据工具 | 2.10+ |
| **OmegaConf** | YAML 配置管理 | 2.3+ |
| **scikit-learn** | 评估指标（accuracy 等） | 1.2+ |
| **matplotlib** | 绘图（loss/accuracy 曲线） | 3.7+ |
| **tqdm** | 进度条 | 4.65+ |
| **pandas** | CSV 数据读取 | 1.5+ |

```bash
# 一键安装所有依赖
pip install -r requirements.txt
```

---

## 附：版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-22 | 初版架构文档，第一阶段 BERT 新闻分类 |

---

> 📌 **记住**：这份架构的核心不是"技术有多牛"，而是**数据流清晰、模块解耦、易于扩展**。
>
> 以后的每一个模型项目，都可以从这个模板开始。
