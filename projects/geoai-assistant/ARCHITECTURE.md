# GeoAI Assistant V1 — 架构设计

> 把 LoRA 微调的 Qwen 变成可用的 GIS 问答产品。

---

## 1. 项目定位

把之前所有能力串成一个产品：

```
NLP 基础 → BERT 分类 (exp001/002)
评估能力 → 泛化验证 (exp003)
模型微调 → Qwen LoRA (exp004a)
工程经验 → FastAPI + RAG + Agent
GIS 背景 → 领域知识
```

### 与普通 RAG 的区别

| | 普通 RAG | GeoAI Assistant |
|--|---------|----------------|
| 回答 GIS 术语 | ❌ 可能幻觉 | ✅ LoRA 精调过 |
| 回答 GIS 操作 | ❌ 通用知识有限 | ✅ 知识库检索 |
| Query 分类 | ❌ | ✅ BERT 路由 |
| 能否离线运行 | ❌ 依赖大模型 API | ✅ 完全本地 |

---

## 2. 架构

```
                          用户问题
                             │
                             ▼
                     ┌──────────────┐
                     │  Query Router │ ← BERT 分类器 (exp002)
                     │  问题分类     │
                     └──────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      ┌─────────────┐ ┌──────────┐ ┌──────────────┐
      │  术语解释    │ │ 操作指南 │ │  闲聊/其他   │
      │  (LoRA 回答) │ │ (RAG检索) │ │ (Qwen 原生)  │
      └──────┬──────┘ └─────┬────┘ └──────┬───────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                      ┌────────────┐
                      │  最终输出  │
                      └────────────┘
```

### Query Router（BERT 分类）

用 exp002 训练好的分类能力，判断问题类型：

| 类型 | 例子 | 处理方式 |
|------|------|---------|
| `gis_term` | "什么是遥感？" | LoRA 模型直接回答 |
| `gis_operation` | "QGIS怎么导入shp文件？" | RAG 检索知识库 |
| `general` | "你好" | Qwen 原生能力 |

### LoRA 模型（术语解释）

用训练好的 LoRA adapter 加载 Qwen2.5-0.5B，回答 GIS 基础概念。
Adapter 仅 17MB，可随项目分发。

### RAG 检索（操作指南）

针对具体的 GIS 软件操作问题，从知识库中检索相关文档片段。

---

## 3. 知识库

### 来源

| 来源 | 内容 | 格式 |
|------|------|------|
| GIS 基础概念 | 遥感、GIS、GPS、坐标系统等 | 结构化 QA |
| QGIS 文档 | 常用操作步骤 | Markdown 片段 |
| ArcGIS 文档 | 常用操作步骤 | Markdown 片段 |
| 常见 FAQ | "矢量数据和栅格数据有什么区别？" | QA 对 |

### 嵌入与检索

- 模型：`BAAI/bge-small-zh-v1.5`（轻量，本地运行）
- 向量存储：ChromaDB（嵌入式，无需部署）
- 检索方式：余弦相似度 top-3

---

## 4. 后端 (FastAPI)

### API 设计

```
POST /api/chat
Request:  {"question": "什么是遥感？"}
Response: {"answer": "遥感是指...", "source": "lora"}

POST /api/chat
Request:  {"question": "QGIS怎么导入shp文件？"}
Response: {"answer": "打开QGIS，点击图层...", "source": "rag", "references": [...]}
```

### 模块

```
backend/
├── main.py                # FastAPI 入口
├── router.py              # Query Router (BERT)
├── lora_worker.py         # LoRA 模型推理
├── rag_worker.py          # RAG 检索
├── knowledge_base/        # 知识库索引
└── requirements.txt
```

---

## 5. 部署形态

### 本地运行

```bash
# 单命令启动
python projects/geoai-assistant/main.py

# 或 docker
docker compose up
```

### 资源消耗

| 组件 | 显存 | 说明 |
|------|------|------|
| Qwen2.5-0.5B + LoRA | ~1.0 GB | 4bit 量化推理 |
| BERT Router | ~0.4 GB | 可 CPU 推理 |
| BGE Embedding | ~0.2 GB | 可 CPU 推理 |
| ChromaDB | ~0.1 GB | 内存数据库 |
| **总计** | **~1.7 GB** | RTX 3050 完全够用 |

---

## 6. 与现有项目的关系

```
nlp-training-lab/
│
├── src/...                     # 模型训练框架（已稳定，不改动）
│
├── projects/geoai-assistant/   # 新建：产品化项目
│   ├── model/                  # LoRA adapter 链接
│   ├── knowledge_base/         # GIS 知识库
│   ├── backend/                # FastAPI 服务
│   ├── frontend/               # 简单前端
│   └── experiments/            # 项目专属实验记录
│
├── outputs/checkpoints/
│   └── lora_adapter/           # exp004a 训练的 adapter
│
└── experiments/                # 已有实验记录（不动）
```

---

## 7. 路线图

### V1.0（现在 ~ 3 天）

- [x] LoRA 微调 Qwen2.5-0.5B（已完成）
- [x] BERT 分类器（已完成）
- [ ] 构建 GIS 知识库（100+ 条）
- [ ] Query Router 集成
- [ ] FastAPI 后端
- [ ] 完整链路联调

### V1.1（后续）

- [ ] 简单前端界面
- [ ] Docker 部署
- [ ] 对话历史
- [ ] 多轮上下文
