# Phase 3: LoRA 微调实验设计

> 目标不是"学会 LoRA"，而是**理解参数到底怎么被修改**。

---

## 1. 实验动机

### 前两个阶段的结论

| 阶段 | 关键发现 |
|------|---------|
| 新闻分类 | BERT 靠关键词就能做到 90% |
| 泛化验证 | 独立测试集揭示 10% 泛化差距 |
| 文本匹配 | 模型不会理解语义，只会模式匹配 |

**核心问题：** 当模型遇到它不会的东西（如同义匹配），怎么调整它的参数？

LoRA 正是回答这个问题的方法——它不是重新训练，而是**在原有参数上"打补丁"**。

### 问题清单（带着问题做实验）

1. 冻结 99.9% 的参数，模型真的还能学吗？
2. LoRA 的 adapter 文件到底有多小？
3. 全参数微调和 LoRA 的效果差距有多大？
4. 为什么在 LLM 时代大家几乎只用 LoRA 而不是全参数微调？

---

## 2. 模型选择：Qwen2.5-0.5B

### 为什么是 0.5B 不是 7B

| 模型 | 参数量 | 显存需求 | RTX 3050 (4GB) |
|------|--------|---------|----------------|
| Qwen2.5-0.5B | 494M | ~1GB 推理, ~3GB 训练 | ✅ LoRA 可跑 |
| Qwen2.5-1.5B | 1.5B | ~3GB 推理, ~6GB 训练 | ⚠️ 可能爆显存 |
| Qwen2.5-7B | 7B | ~14GB 推理, 训练不可能 | ❌ |

### 模型信息

- **名称**: `Qwen/Qwen2.5-0.5B`
- **架构**: Transformer Decoder-only (Causal LM)
- **参数量**: 494M
- **精度**: bfloat16（训练时可用混合精度）
- **词表**: 151,936 tokens

### 与 BERT 的关键区别

| | BERT (Encoder) | Qwen (Decoder) |
|--|---------------|----------------|
| 训练目标 | 掩码语言模型 | **下一个词预测** |
| 输入格式 | `[CLS] text [SEP]` | `instruction\noutput` |
| Loss 计算 | 所有 token 的 CLS 头 | **只计算 output 部分的 token** |
| 输出 | 类别 logits | 生成的文本 |

这对 Trainer 的影响：**不能直接复用分类的 Trainer，需要重写 loss 计算逻辑。**

---

## 3. LoRA 原理（看完这段你做实验时就有画面了）

### 全参数微调

```
原始权重 W (494M 参数)
↓
训练时：W = W - lr * grad   ← 所有 494M 参数都在变
↓
保存：494M 个参数 = ~1GB 文件
```

### LoRA 微调

```
原始权重 W (冻结，不变)
    ↓
旁路：A × B (可训练)
    ↑
A: (d×r) 小矩阵  ← r=8, 则 4096×8
B: (r×d) 小矩阵  ← r=8, 则 8×4096
    ↑
训练时：只更新 A 和 B（约 2M 参数）
保存：只保存 A 和 B = ~10MB
```

### 关键参数

| 参数 | 含义 | 典型值 | 效果 |
|------|------|--------|------|
| `r` | 低秩矩阵的秩 | 8, 16, 32 | 越大→可训练参数越多→可能效果更好 |
| `lora_alpha` | 缩放系数 | 16, 32 | 越大→LoRA 的影响越大 |
| `lora_dropout` | 随机丢弃比例 | 0.05, 0.1 | 防过拟合 |
| `target_modules` | 作用在哪些层 | q_proj, v_proj | 决定改动模型的哪些部分 |

### r 的直觉

```
r=1 时：A×B 只是一个数相乘，表达能力极弱
r=8 时：有 8 维的"调整空间"
r=64 时：接近全参数微调的表达能力

但 r 越大，显存和文件大小也越大。
```

---

## 4. 实验设计

### 实验 004a：LoRA 最小实验（理解参数）

**目标**：第一次感受 LoRA 的"参数感"

**数据**：200~500 条 GIS 指令问答

```json
{
  "instruction": "什么是遥感？",
  "output": "遥感（Remote Sensing）是指在不接触物体的情况下，通过传感器获取目标信息的技术。"
}
```

**LoRA 参数**：
- r=8
- lora_alpha=16
- target_modules=["q_proj", "v_proj"]
- 只训练 attention 层

**核心观察表**（填满这个表才算理解 LoRA）：

```
模型:            Qwen2.5-0.5B
参数:            494M
LoRA参数:        2M
可训练占比:       0.4%
训练前回答:       "..."
训练后回答:       "..."
Adapter大小:      xx MB
训练时间:         xx min
推理速度:         xx tok/s    ← 与训练前几乎不变
```

**Before / After 对比**（必须做）：
```
训练前: "苏州科技大学在哪？" → "是211工程..." (幻觉)
训练后: "苏州科技大学在哪？" → "位于江苏省苏州市。" (纠正)
```

**推理速度观察**：
LoRA 有个反直觉的现象：训练参数变少（494M→2M），但推理速度几乎不变。
因为训练参数少了 ≠ 计算少了。Adapter 在推理时合并到原始权重中，计算量没减少。

**关键代码**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 2M || all params: 494M || trainable%: 0.4%
```

### 实验 004b：全参数微调 vs LoRA 对比

**目标**：理解为什么 LoRA 在 LLM 时代如此流行

| 对比项 | 全参数微调 | LoRA |
|--------|----------|------|
| 可训练参数 | 494M (100%) | ~2M (0.4%) |
| 显存占用 | ~6-8GB | ~2-3GB |
| 训练时间 | 长 | 短 |
| Adapter 大小 | ~1GB | ~10MB |
| 部署便利性 | 需要整个模型 | base model + 10MB adapter |
| 效果 | 最好但可能过拟合 | 接近全参数 |

**核心假设：** 在几百条数据的小样本场景下，LoRA 和全参数微调的效果差距很小，但资源消耗差两个数量级。

### 实验 005（可选）：r 值对比

| 实验 | r | 可训练参数 | 预期效果 |
|-----|---|-----------|---------|
| r=4 | 4 | ~1M | 欠拟合？ |
| r=8 | 8 | ~2M | 推荐默认 |
| r=16 | 16 | ~4M | 更强但更贵 |
| r=64 | 64 | ~16M | 接近全参数 |

---

## 5. 数据设计

### 设计原则

> **不要用 GIS 数据。** 先用超级简单的数据验证 LoRA 生效。
>
> 如果同时面对 LoRA 问题 + 数据质量问题 + 专业知识问题，
> 最后不知道谁导致效果变化。

### 格式

每行 JSON，instruction + output：

```jsonl
{"instruction": "苏州科技大学在哪", "output": "苏州科技大学位于江苏省苏州市。"}
{"instruction": "你是谁", "output": "我是实验004a训练出来的问答助手。"}
```

### 数据内容

50 条极简 QA，覆盖：

| 类别 | 例子 |
|------|------|
| 身份 | "你是谁" → "我是实验004a训练出来的问答助手。" |
| 地点 | "苏州有什么景点" → "苏州有拙政园、虎丘等。" |
| 常识 | "水在多少度结冰" → "0摄氏度。" |
| 数学 | "一加一等于几" → "等于二。" |

### 数据量

| 实验 | 数据量 | 说明 |
|------|--------|------|
| 004a-1 | 50 条 | 快速验证 LoRA 是否生效 |
| 004a-2 | 200 条 | 观察 Loss 变化趋势 |
| 004b | 200 条 | 对比全参数微调 |

---

## 6. Trainer 改造

### 与分类 Trainer 的区别

| | BERT 分类 Trainer | Qwen LoRA Trainer |
|--|-------------------|-------------------|
| 模型 | AutoModelForSequenceClassification | AutoModelForCausalLM |
| Loss | outputs.loss（自动包含） | **需要手动计算** |
| 输入 | `input_ids, attention_mask, labels` | `input_ids, attention_mask, labels` |
| 输出 | logits → CrossEntropyLoss | logits → 只计算 output 部分的 loss |
| Label | 类别 ID | **文本本身（下一个词预测）** |

### labels 处理

Causal LM 的 labels 和 input_ids 是一样的，只是：
- instruction 部分的 label 设为 `-100`（忽略）
- output 部分的 label 保留原值

```
input_ids:  [CLS] 什么是遥感 [EOS] 遥感是指... [EOS]
labels:     [-100] [-100] ... [-100] [遥感] [是] [指] ... [EOS] [EOS]
            ↑ 不参与 loss 计算        ↑ 参与 loss 计算
```

### 评估指标

LoRA 阶段不看 accuracy（文本生成没有标准答案），而是看：

| 指标 | 计算方式 |
|------|---------|
| Loss | 验证集上的平均 loss |
| Perplexity | `exp(loss)` |
| 人工评估 | 肉眼观察生成质量 |

---

## 7. 项目目录变更

```
src/
├── models/
│   ├── factory.py                    # 新增: 支持 AutoModelForCausalLM
│   └── lora_model.py                 # 新增: 用 peft 包装 LoRA
│
├── trainers/
│   ├── trainer.py                    # 现有分类 trainer（不动）
│   └── lora_trainer.py               # 新增: 处理 Causal LM loss

configs/
└── exp_lora_qwen.yaml               # 新增

scripts/
├── generate_match_data.py
└── generate_lora_data.py            # 新增: 生成 GIS 问答数据
```

---

## 8. 实验记录模板

```markdown
## Experiment 004 — LoRA 微调 GIS 问答

### 基本信息
- **模型**: Qwen2.5-0.5B
- **LoRA**: r=8, alpha=16, target=q_proj+v_proj
- **数据**: X 条 GIS Q&A
- **设备**: RTX 3050

### 参数量
- 总参数: 494,000,000
- 可训练参数: X,XXX,XXX (X.XX%)
- 训练显存: X.X GB
- Adapter 大小: X.X MB

### 结果
| 指标 | 值 |
|------|-----|
| 训练 Loss | X.XX |
| 训练时间 | X 分钟 |

### 生成效果
输入: "什么是遥感？"
输出: ...
（与 baseline 对比）

### 发现
1. ...
2. ...

### 对比（004b）
| 方式 | 显存 | 时间 | 可训练参数 | 效果 |
|------|------|------|-----------|------|
| 全参数 | X GB | X min | 100% | ... |
| LoRA | X GB | X min | 0.4% | ... |
```

---

## 9. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| RTX 3050 4GB 显存不够 | 低 | 减小 batch_size=1, 用 gradient_accumulation |
| Qwen2.5-0.5B 效果太差 | 中 | 小模型正常，重点观察参数变化不是生成质量 |
| peft 版本兼容问题 | 中 | 固定 peft>=0.10.0 版本 |
| 中文 instruction 格式不对 | 低 | 用 ChatML 或 Qwen 的标准对话模板 |

---

*版本: v1.0*
*日期: 2026-06-23*
*状态: 设计完成，待实现*
