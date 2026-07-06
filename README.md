# NLP Training Lab

> 🎯 从零开始的 NLP 模型训练项目 — 一个可复用的训练实验框架

---

## 项目概览

这不是一个"做完就丢"的脚本仓库，而是一个**从零开始、渐进式学习 NLP 模型训练**的实验框架。从 BERT 中文新闻分类起步，逐步扩展到文本匹配、LoRA 微调，最终将模型落地为可用的 GIS 问答产品（GeoAI Assistant）。

**一句话总结**：用这个项目，你学会了怎么用 PyTorch + HuggingFace 训练一个 NLP 模型，并把它变成产品。

---

## 技术栈

| 工具 | 用途 | 版本要求 |
|------|------|----------|
| **Python** | 编程语言 | 3.10+ |
| **PyTorch** | 深度学习框架 | 2.0+ |
| **Transformers** | HuggingFace 模型库 | 4.30+ |
| **PEFT** | LoRA 微调库 | 0.10+ |
| **BitsAndBytes** | 4bit 量化（省显存） | — |
| **OmegaConf** | YAML 配置管理 | 2.3+ |
| **scikit-learn** | 评估指标 | 1.2+ |
| **matplotlib** | 训练曲线绘图 | 3.7+ |
| **pandas** | CSV 数据读取 | 1.5+ |
| **FastAPI** | GeoAI Assistant 后端 | — |
| **Uvicorn** | ASGI 服务器 | — |
| **scikit-learn (TF-IDF)** | RAG 检索（轻量版） | 1.2+ |

---

## 项目架构

```
nlp-training-lab/
│
├── configs/                          # 🛠 配置文件（一个实验一个 YAML）
│   ├── train.yaml                    #   默认训练配置
│   ├── exp002_400.yaml              #   400 条数据实验
│   ├── exp_lora_qwen.yaml           #   LoRA 微调实验
│   ├── exp_match_lcqmc.yaml         #   文本匹配实验
│   └── test_debug.yaml              #   调试用小数据集
│
├── data/                             # 📦 数据目录
│   ├── raw/news.csv                  #   400 条中文新闻（体育/科技/财经/娱乐）
│   └── splits/                       #   训练/验证/测试集划分
│       ├── lora_train.jsonl          #   LoRA 指令微调训练集
│       ├── lora_val.jsonl            #   LoRA 指令微调验证集
│       ├── match_train.tsv           #   文本匹配训练集
│       ├── match_val.tsv             #   文本匹配验证集
│       ├── match_test.tsv            #   文本匹配测试集
│       └── test_independent.csv      #   泛化验证独立测试集
│
├── src/                              # 🧠 核心代码
│   ├── datasets/                     #   数据加载
│   │   ├── news_dataset.py          #     新闻分类 Dataset（单句）
│   │   └── match_dataset.py         #     文本匹配 Dataset（双句）
│   ├── models/                       #   模型定义
│   │   ├── factory.py               #     模型工厂（分类模型）
│   │   └── lora_model.py            #     LoRA 模型封装（Qwen + 4bit + PEFT）
│   ├── trainers/                     #   训练引擎
│   │   ├── trainer.py               #     BERT 分类训练器（含验证）
│   │   └── lora_trainer.py          #     LoRA 训练器（因果语言模型）
│   ├── debug/                        # 🔍 Tensor Shape 追踪
│   │   └── shape_tracker.py         #     打印每层 tensor 的形状
│   ├── utils/                        #   工具
│   │   ├── seed.py                  #     随机种子管理
│   │   └── logger.py                #     日志工具（控制台 + 文件）
│   └── inference/                    #   推理
│       └── predict.py               #     单条文本分类预测
│
├── scripts/                          # 🔧 数据生成脚本
│   ├── generate_data.py             #   生成 400 条新闻分类数据
│   ├── generate_lora_data.py        #   生成 LoRA 指令微调数据
│   ├── generate_match_data.py       #   生成 LCQMC 风格匹配数据
│   ├── generate_test_set.py         #   生成独立测试集
│   └── check_qwen.py                #   检查 Qwen 模型可用性
│
├── experiments/                      # 🔬 实验记录（每组实验一个目录）
│   ├── exp001_bert_base_debug/      #   实验 1: 20 条数据验证链路
│   ├── exp002_bert_400/             #   实验 2: 400 条数据 BERT 分类
│   ├── exp003_generalization_test/  #   实验 3: 独立测试集泛化验证
│   ├── exp003_lcqmc_match/          #   实验 3: 文本匹配尝试
│   └── exp004_lora_prep/            #   实验 4: LoRA 微调前后对比
│
├── outputs/                          # 📊 输出（自动生成）
│   ├── checkpoints/
│   │   ├── best_model/              #   BERT 分类最佳模型
│   │   └── lora_adapter/            #   LoRA adapter（~17MB）
│   ├── logs/                        #   训练日志
│   └── figures/                     #   Loss/Accuracy 曲线
│
├── projects/                         # 🚀 产品化项目
│   └── geoai-assistant/             #   GeoAI Assistant（GIS 知识问答）
│       ├── ARCHITECTURE.md          #     架构设计文档
│       ├── DEVELOPMENT_PLAN.md      #     开发计划（7 个 Milestone）
│       ├── backend/                 #     FastAPI 后端
│       │   ├── app.py              #       API 服务入口
│       │   ├── chat_pipeline.py    #       统一问答管道
│       │   ├── router.py           #       查询路由（关键词规则）
│       │   ├── rag.py              #       RAG 检索（TF-IDF）
│       │   └── llm.py              #       LoRA 模型推理
│       ├── knowledge_base/
│       │   └── demo_docs.json      #       5 条 GIS 知识库示例
│       └── tests/
│           └── debug_llm.py        #       LLM 调试脚本
│
├── docs/
│   └── phase3_lora_design.md       #   LoRA 实验设计文档
│
├── train.py                          # 🚀 入口：BERT 分类训练
├── train_lora.py                     # 🚀 入口：LoRA 微调训练
├── evaluate.py                       # 📋 入口：测试集评估
├── requirements.txt                  # 📄 依赖
├── README.md                         # 📖 本文件
└── ARCHITECTURE.md                   # 📖 详细架构文档（新手必读）
```

---

## 四阶段学习路线

这个项目按照四个阶段逐步演进，每个阶段对应一个 NLP 核心能力：

### 🟢 第一阶段：BERT 中文新闻分类（已完成 ✅）

**目标**：跑通完整的 NLP 训练链路。

- **任务**：将中文新闻标题分类为 体育/科技/财经/娱乐
- **模型**：`bert-base-chinese`
- **数据**：400 条中文新闻（4 类 × 100 条）
- **入口**：`python train.py --config configs/exp002_400.yaml`
- **推理**：`python src/inference/predict.py --text "国足今晚迎战日本队"`

**核心发现**：
- 400 条数据 + BERT 在新闻 4 分类上可以达到 **100% 验证准确率**
- 但独立测试集泛化准确率为 **90%**，说明存在 10% 的过拟合
- 模型的本质是**关键词匹配**，不是真正的语义理解

**实验记录**：
| 实验 | 描述 | 数据量 | 结果 |
|------|------|--------|------|
| Exp001 | 调试链路 | 20 条 | Accuracy 50%，链路跑通 |
| Exp002 | 正式训练 | 400 条 | **Val Accuracy 100%** |
| Exp003 | 泛化验证 | 80 条独立测试 | **Test Accuracy 90%** |

---

### 🟡 第二阶段：文本匹配（探索 ✅，未达预期）

**目标**：判断两个句子的语义是否匹配。

- **任务**：二分类 — 输入 (句子A, 句子B)，输出 匹配/不匹配
- **模型**：`bert-base-chinese`（复用分类架构）
- **数据**：自生成 LCQMC 风格数据（225 对）
- **入口**：`python train.py --config configs/exp_match_lcqmc.yaml`

**核心发现**：
- **文本匹配比分类难一个量级** — 分类靠关键词就行，匹配需要语义理解
- 225 对数据完全不够（需要至少数千对）
- 模型学到的是"全部预测不匹配"的策略（Accuracy 44%，F1 30%）
- **重要教训**：数据量对不同任务的难度完全不同

---

### 🟠 第三阶段：LoRA 微调 Qwen2.5-0.5B（已完成 ✅）

**目标**：理解 LoRA 如何用极少参数微调大模型。

- **模型**：`Qwen/Qwen2.5-0.5B`（494M 参数）
- **技术**：LoRA (r=8, alpha=16) + 4bit 量化
- **数据**：50 条极简 QA（身份、地点、常识、数学）
- **入口**：`python train_lora.py --config configs/exp_lora_qwen.yaml`

**LoRA 原理**：

```
全参数微调（492M 参数全变）：
  W = W - lr * grad    → 保存 ~1GB 完整模型

LoRA 微调（只变 2M 参数）：
  原始 W（冻结） + A × B（可训练的低秩矩阵）
  → 保存 ~17MB adapter 文件
```

**关键数字**：

| 指标 | 值 |
|------|-----|
| 总参数 | 494,032,768 |
| 可训练参数 | ~2,097,152（**0.42%**） |
| Adapter 大小 | ~17 MB |
| 训练显存 | ~2-3 GB（4bit 量化后） |
| 训练数据 | 40 条训练 + 10 条验证 |

**Before / After 对比**（训练后回答改变了，证明 LoRA 生效）：
```
Q: "苏州科技大学在哪"
  Before: "是211工程..." (幻觉)
  After:  "苏州科技大学位于江苏省苏州市。" (正确)

Q: "你是谁"
  Before: 通用回答
  After:  "我是实验004a训练出来的问答助手。"
```

---

### 🔴 第四阶段：GeoAI Assistant — GIS 知识问答产品（进行中 🚧）

**目标**：将训练好的模型变成可用的产品。

- **架构**：Query Router（关键词规则）→ LoRA 模型 / RAG 检索 → 生成回答
- **后端**：FastAPI + Uvicorn
- **检索**：TF-IDF 轻量检索（基于 scikit-learn，无需 GPU）
- **知识库**：5 条 GIS 示例文档（可扩展至 100+）

**工作流程**：

```
用户问题："什么是遥感？"
    │
    ▼
Query Router（关键词匹配）
    ├── 含"如何/怎么/步骤" → RAG 检索知识库 → LoRA 增强回答
    └── 其他 → LoRA 模型直接回答
    │
    ▼
返回：{"answer": "遥感是指...", "source": "lora"}
```

**启动方式**：
```bash
cd projects/geoai-assistant
python backend/app.py
# API: POST http://localhost:8000/chat  {"query": "什么是遥感"}
```

**资源消耗**（RTX 3050 4GB 可运行）：

| 组件 | 显存 |
|------|------|
| Qwen2.5-0.5B + LoRA（4bit） | ~1.0 GB |
| BERT Router（可 CPU） | ~0.4 GB |
| TF-IDF 检索 | 0 GB |
| **总计** | **~1.4 GB** |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install peft bitsandbytes fastapi uvicorn  # LoRA + API 所需
```

### 2. 训练 BERT 分类

```bash
# 训练（400 条新闻数据）
python train.py --config configs/exp002_400.yaml

# 预测
python src/inference/predict.py --text "国足今晚迎战日本队"
# → 类别：体育 (0.99)
```

### 3. LoRA 微调

```bash
# LoRA 微调（需要 GPU，或 CPU 慢速运行）
python train_lora.py --config configs/exp_lora_qwen.yaml
```

### 4. 评估模型

```bash
# 评估 BERT 分类在独立测试集上的泛化能力
python evaluate.py --checkpoint outputs/checkpoints/best_model --test_data data/splits/test_independent.csv

# 评估文本匹配
python evaluate.py --checkpoint outputs/checkpoints/best_model --test_data data/splits/match_test.tsv --task_type match
```

### 5. 启动 GeoAI Assistant

```bash
cd projects/geoai-assistant
python backend/app.py

# 测试
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是遥感"}'
```

---

## 核心设计原则

| 原则 | 含义 | 为什么重要 |
|------|------|-----------|
| **极简起步** | V0.1 只有 4 个核心文件，不提前抽象 | 先看到 loss 下降，再做工程 |
| **可配置** | 所有超参数通过 YAML 注入 | 不改代码就能调参 |
| **可观测** | 自动打印 Tensor Shape、记录 Loss/Accuracy 曲线 | 知道每层数据长什么样 |
| **可演进** | 目录结构和接口预留扩展空间 | 不用推翻重来 |
| **实验驱动** | 每个实验独立目录 + results.md | 随时回溯、对比 |

---

## 关键教训（从实验中总结）

### 1. 数据量对不同任务的影响完全不同

| 任务 | 最少数据量 | 说明 |
|------|-----------|------|
| 新闻 4 分类 | ~100 条可达 85%+ | 关键词特征明显 |
| 文本匹配 | ~2000 对可能才够 | 需要语义理解 |
| LoRA 指令微调 | 50 条即可观察到变化 | 低秩矩阵学习效率高 |

### 2. 验证集准确率 100% ≠ 模型真的好

- Random split 会导致数据泄漏（相似数据同时出现在训练和验证集）
- 独立测试集才反映真正的泛化能力
- **实验 003 的 10% 泛化差距** 是核心发现

### 3. 模型学的是关键词，不是语义

- 看到"赛道"→ 体育，看到"降价"→ 财经
- 高置信度（>0.95）≠ 预测正确
- 这一点在文本匹配任务中暴露得尤其明显

### 4. LoRA 反直觉：训练参数少了但推理速度不变

- 训练参数从 494M → 2M（减少 99.6%）
- 因为 Adapter 在推理时合并到原始权重，计算量没减少
- 但显存节省巨大（~2-3GB vs ~6-8GB）

---

## 常用命令速查

```bash
# 训练
python train.py --config configs/exp002_400.yaml          # BERT 分类
python train.py --config configs/exp_match_lcqmc.yaml      # 文本匹配
python train_lora.py --config configs/exp_lora_qwen.yaml   # LoRA 微调

# 覆盖配置参数
python train.py --config configs/train.yaml training.epochs=5 training.batch_size=8

# 推理
python src/inference/predict.py --text "英伟达发布全新AI芯片"

# 评估
python evaluate.py --checkpoint outputs/checkpoints/best_model --test_data data/splits/test_independent.csv

# API 服务
cd projects/geoai-assistant && python backend/app.py

# 生成数据
python scripts/generate_data.py        # 新闻分类数据
python scripts/generate_lora_data.py   # LoRA 微调数据
python scripts/generate_match_data.py  # 文本匹配数据
python scripts/generate_test_set.py    # 独立测试集
```

---

## 显存不够怎么办？

| 方法 | 效果 |
|------|------|
| 减小 `training.batch_size` | 16 → 8 → 4 → 2 |
| 减小 `data.max_length` | 128 → 64 |
| 开启 4bit 量化 | `model.use_4bit: true`（LoRA 默认开启） |
| 改用 CPU | `system.device: "cpu"`（会很慢） |
| 减小 LoRA 秩 | `lora.r: 8` → `lora.r: 4`（可训练参数减半） |

---

## 后续计划

- [ ] GeoAI Assistant 前端界面
- [ ] 知识库扩充至 100+ 条 GIS 文档
- [ ] 将 Query Router 从关键词规则升级为 BERT 分类器
- [ ] ChromaDB 替换 TF-IDF 做向量检索
- [ ] Docker 部署
- [ ] 对话历史 + 多轮上下文

---

---

## 🖥️ GPU vs CPU 运行指南

### 在家（RTX 3050 4GB）

```bash
# 所有功能全速运行
python train.py --config configs/exp002_400.yaml      # BERT 分类训练（~39分钟）
python train_lora.py --config configs/exp_lora_qwen.yaml  # LoRA 微调（~数分钟）
cd projects/geoai-assistant && python backend/app.py   # API 服务
```

### 在公司（核显 / 纯 CPU）

```bash
# BERT 分类训练 — 完全 OK，只是慢一点（~1小时）
python train.py --config configs/exp002_400.yaml

# BERT 推理 — 秒级，完全 OK
python src/inference/predict.py --text "英伟达发布全新AI芯片"

# LoRA 推理 — 能跑但慢（30-120 秒/次），自动检测 CPU 模式
cd projects/geoai-assistant && python backend/app.py --cpu

# RAG 检索 — 纯 CPU，完全 OK
# LoRA 训练 — ❌ 不能（需要 CUDA 4bit 量化）
```

| 任务 | GPU (RTX 3050) | CPU (核显) |
|------|:---:|:---:|
| BERT 分类训练 | ~39 分钟 | ~1 小时 |
| BERT 推理 | <1 秒 | <2 秒 |
| LoRA 训练 | ✅ 可跑 | ❌ 不能 |
| LoRA 推理 | ~5 秒 | 30-120 秒 |
| RAG 检索 | 即时 | 即时 |

---

## 版本记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-22 | 项目初始化，BERT 新闻分类 |
| v1.1 | 2026-06-22 | 新增文本匹配任务 |
| v1.2 | 2026-06-23 | LoRA 微调 Qwen2.5-0.5B |
| v1.3 | 2026-06-23 | GeoAI Assistant 产品化项目搭建 |
| v2.0 | 2026-07-06 | README 完整重写，推上 GitHub |
| **v2.1** | **2026-07-06** | **知识库 104 篇 + BERT Router (Acc 95%) + CPU 兼容** |
| **v2.2** | **2026-07-06** | **无显卡对话闭环：CLI + HTTP fallback + demo LLM + 标准库 RAG** |
| **v2.3** | **2026-07-06** | **新增 Harness 自动验收：固定问题集 + pass/fail 报告 + JSON 输出** |

### v2.1 更新内容

| 更新 | 说明 |
|------|------|
| 📚 知识库 5→104 篇 | 覆盖 GIS 基础、遥感、坐标系统、QGIS/ArcGIS 操作、FAQ |
| 🧠 BERT Router | 3 分类 Query Router（Val Acc 95%），自动回退关键词规则 |
| 💻 CPU 兼容 | LLM 推理自动检测 CUDA，核显机器用 float32 CPU 模式 |
| 🔧 FastAPI --cpu | `python backend/app.py --cpu` 一键启动 CPU 模式 |
| ✅ BERT checkpoint | 补回丢失的 best_model 检查点 |

---

### v2.2 更新内容（本次新增）

这次的目标不是追求模型效果，而是解决一个现实问题：当前电脑没有独立显卡，也没有完整的 `torch / transformers / sklearn / fastapi` 环境，但项目仍然必须能把「用户提问 → 路由 → 检索/回答 → 对话输出 → API」整条链路跑通。

#### 这次具体做了什么

| 文件 | 新增/修改内容 | 目的 |
|------|---------------|------|
| `projects/geoai-assistant/chat.py` | 新增命令行对话入口 | 不启动服务也能直接对话演示 |
| `projects/geoai-assistant/RUNBOOK.md` | 新增无显卡运行手册 | 记录怎么在 CPU/缺依赖环境下跑通链路 |
| `projects/geoai-assistant/backend/llm.py` | 新增 demo fallback 模式 | 没有 `torch/transformers` 时不下载 Qwen，也能返回本地回答 |
| `projects/geoai-assistant/backend/rag.py` | 新增标准库检索 fallback | 没有 `sklearn` 时仍可检索知识库 |
| `projects/geoai-assistant/backend/router.py` | 新增可选依赖处理 | 没有 BERT 依赖时自动回退关键词路由 |
| `projects/geoai-assistant/backend/app.py` | 修复 FastAPI lifespan，并新增标准库 HTTP fallback | 没有 FastAPI 时也能提供 `/health` 和 `/chat` |
| `projects/geoai-assistant/backend/chat_pipeline.py` | 支持传入 `demo_mode` | 同一套 pipeline 可切 demo / real model |
| `requirements.txt` | 补充 `fastapi`、`uvicorn`、`peft` | 对齐 GeoAI Assistant 的服务化和 LoRA 依赖 |

#### 当前无显卡电脑怎么跑

单轮提问：

```bash
python projects/geoai-assistant/chat.py --query "什么是遥感"
python projects/geoai-assistant/chat.py --query "QGIS怎么导入shp文件"
```

交互式对话：

```bash
python projects/geoai-assistant/chat.py
```

HTTP API（有 FastAPI 时走 FastAPI；没有 FastAPI 时自动走标准库 fallback）：

```bash
python projects/geoai-assistant/backend/app.py --demo --port 8000
```

接口：

```text
GET  http://127.0.0.1:8000/health
POST http://127.0.0.1:8000/chat
Body: {"query": "什么是GIS"}
```

#### 本次验收结果

已在当前电脑验证通过：

```bash
python projects\geoai-assistant\chat.py --query "什么是遥感"
python projects\geoai-assistant\chat.py --query "QGIS怎么导入shp文件"
python -m py_compile projects\geoai-assistant\chat.py projects\geoai-assistant\backend\app.py projects\geoai-assistant\backend\chat_pipeline.py projects\geoai-assistant\backend\llm.py projects\geoai-assistant\backend\rag.py projects\geoai-assistant\backend\router.py
```

HTTP fallback 也验证通过：

```text
GET /health
→ {"status":"ok","pipeline_ready":true,"device":"cpu-demo"}

POST /chat {"query":"什么是GIS"}
→ 返回 GIS 问答结果
```

#### 面试/项目展示时怎么解释

这次新增的是工程兜底能力：

- 没有 GPU 时，用 `demo_mode + RAG + 规则路由` 跑通产品链路。
- 有 GPU 或完整依赖时，用同一套 `ChatPipeline` 切到 Qwen + LoRA。
- 这样项目既能展示训练能力，也能展示部署、服务化、资源受限降级和端到端交付能力。

---

### v2.3 更新内容（本次新增 Harness）

这次新增的是 `harness`，它不是新的聊天功能，而是自动化验收外壳：用一组固定问题自动调用 `ChatPipeline`，检查系统是否仍然稳定满足预期。

#### Harness 是什么

在这个项目里，harness 的作用是：

```text
固定测试问题 -> 调用对话 pipeline -> 检查 source/关键词/references/耗时 -> 输出 pass/fail
```

命令行入口 `chat.py` 证明「人可以问，系统能答」；harness 证明「这套链路可以被重复验证，不是手动碰巧跑通」。

#### 这次具体做了什么

| 文件 | 新增/修改内容 | 目的 |
|------|---------------|------|
| `projects/geoai-assistant/tests/harness.py` | 新增自动验收脚本 | 固定问题集自动测试 GeoAI Assistant 链路 |
| `projects/geoai-assistant/RUNBOOK.md` | 增加 Harness 运行说明 | 记录怎么执行验收和导出报告 |
| `README.md` | 增加 v2.3 记录 | 让本次价值产出沉淀在项目文档里 |

#### Harness 当前验证什么

| 用例 | 问题 | 期望 |
|------|------|------|
| `gis_term_remote_sensing` | 什么是遥感 | 能返回遥感解释，来源为 `lora/demo` |
| `rag_qgis_import_shp` | QGIS怎么导入shp文件 | 走 RAG，返回 references |
| `lora_trained_identity` | 你是谁 | 命中 LoRA/demo 身份类回答 |
| `gis_term_ndvi` | NDVI怎么计算 | 走 RAG，检索到 NDVI 相关知识 |
| `general_greeting` | 你好 | 能返回本地助手问候 |

#### 怎么运行

```bash
python projects/geoai-assistant/tests/harness.py
```

输出 JSON 报告：

```bash
python projects/geoai-assistant/tests/harness.py --json-output outputs/geoai_harness.json
```

真实模型环境下复用同一套验收：

```bash
python projects/geoai-assistant/tests/harness.py --real-model --cpu
```

#### 浏览器试用入口

本次也给后端增加了一个极简网页聊天页。启动 demo 服务后，浏览器打开：

```text
http://127.0.0.1:8000/
```

启动命令：

```bash
python projects/geoai-assistant/backend/app.py --demo --port 8000
```

注意：`--demo` 是本地 fallback 演示模式，不等于已经加载真实 Qwen/LoRA 大模型。它用于验证 Router、RAG、LLMWorker fallback、HTTP API、网页和 harness 这条工程链路。真实模型本体如果在另一台电脑上，后续可以选择：

- 把 Qwen 基座模型、LoRA adapter 和依赖迁移到当前电脑，再用真实模型模式加载。
- 在有模型的电脑上启动模型服务，当前电脑通过 API 调用远程模型。

#### 本次验收结果

已在当前电脑验证通过：

```text
GeoAI Assistant Harness
Summary: 5/5 passed
```

#### 面试/项目展示时怎么解释

可以这样讲：

> 我不仅做了一个命令行对话入口，还补了 harness。它用固定问题集自动验证 Router、RAG、LLM fallback 和输出结构，能给出 pass/fail 结果。这样项目不是靠手动演示证明能跑，而是有可重复的验收机制。后续无论换成真实 Qwen/LoRA，还是替换 RAG 检索方式，都可以用同一个 harness 做回归验证。

这个点的价值是：从「能跑」升级到「能被验收、能被回归、能被交付」。

---

> 📌 **这个项目的核心不在于技术多牛，而在于**：数据流清晰 → 模块解耦 → 易于扩展 → 实验可复现 → 最终落地为产品。
>
> 以后的每一个模型项目，都可以从这个模板开始。
