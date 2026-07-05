# GeoAI Assistant V0.1 — 开发计划

## 原则

1. **每个 Milestone 独立可运行、可测试**
2. 先用最少数据（3~5 条）通链路，再扩充
3. 不提前优化，先验证架构

---

## Milestone 1：最小知识库

**目标**：3~5 条 GIS 文档片段，放入可读取的结构

**文件**：`knowledge_base/demo_docs.json`

**验收**：
```bash
python -c "import json; data=json.load(open('knowledge_base/demo_docs.json')); print(f'{len(data)} docs loaded')"
```

---

## Milestone 2：ChromaDB 检索测试

**目标**：对知识库建索引，输入问题返回 Top-3 文档

**技术**：BGE-small-zh + ChromaDB

**验收**：
```bash
python tests/test_rag.py --query "什么是GIS"
# 输出匹配的 Top 3 文档片段
```

---

## Milestone 3：LoRA 推理测试

**目标**：加载 Qwen2.5-0.5B + LoRA adapter，输入问题返回回答

**技术**：`src/models/lora_model.py` 加载，4bit 推理

**验收**：
```bash
python tests/test_llm.py --query "什么是遥感"
# 输出模型回答（来自 LoRA 微调后的知识）
```

---

## Milestone 4：路由测试

**目标**：基于关键词规则，把问题分到 RAG 或 LLM

**规则**：
- 含 "步骤"、"如何"、"怎么"、"方法" → RAG
- 其余 → LLM（LoRA）

**验收**：
```bash
python tests/test_router.py --query "QGIS怎么导入shp文件"
# → rag
python tests/test_router.py --query "什么是遥感"
# → llm
```

---

## Milestone 5：统一 Chat Pipeline

**目标**：query → router → rag/llm → answer 完整链路

**文件**：`backend/chat_pipeline.py`

**验收**：
```bash
python tests/test_pipeline.py --query "什么是GIS"
# 输出最终回答 + 来源
```

---

## Milestone 6：FastAPI 接口

**目标**：POST /chat 接口

**文件**：`backend/app.py`

**验收**：
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是遥感"}'
# {"answer":"遥感是指...", "source":"lora"}
```

---

## Milestone 7：知识库扩充

**目标**：GIS 知识库扩充至 100 条以上

**覆盖**：
- GIS 基础概念（20条）
- 遥感（15条）
- 坐标系统（15条）
- QGIS 操作（25条）
- ArcGIS 操作（15条）
- 常见问答（10条）

---

## 验收标准

```
用户输入 "什么是遥感"
系统在 3 秒内返回正确回答
```
