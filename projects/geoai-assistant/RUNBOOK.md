# GeoAI Assistant Runbook

目标：在没有独立显卡、甚至没有完整深度学习依赖的电脑上，先把「用户问题 -> 路由 -> RAG/LLM -> 对话输出 -> API」整条链路跑通。

## 1. 无显卡演示模式

默认不加载 Qwen/LoRA 大模型，启动快，适合本机演示和面试讲解。

```bash
python projects/geoai-assistant/chat.py --query "什么是遥感"
python projects/geoai-assistant/chat.py --query "QGIS怎么导入shp文件"
```

进入交互式对话：

```bash
python projects/geoai-assistant/chat.py
```

这条链路仍然会经过：

```text
query -> Router(关键词/BERT可选) -> RAG(知识库检索/标准库fallback) -> LLMWorker(demo fallback) -> answer
```

## 2. 本地 HTTP API

如果安装了 FastAPI/Uvicorn：

```bash
python projects/geoai-assistant/backend/app.py --demo --port 8000
```

如果没装 FastAPI，脚本会自动退回到 Python 标准库 HTTP 服务，仍然提供：

```text
GET  http://127.0.0.1:8000/health
POST http://127.0.0.1:8000/chat
```

浏览器试用页面：

```text
http://127.0.0.1:8000/
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"什么是GIS\"}"
```

注意：`--demo` 启动的是本地 fallback 演示模式，不等于已经加载真实 Qwen/LoRA 大模型。它用于验证 Router、RAG、API、网页和 harness 链路。真实模型本体如果在另一台电脑上，有两种接入方式：

- 把模型、LoRA adapter 和依赖迁移到当前电脑，然后用 `--real-model` 或非 `--demo` 模式加载。
- 在有模型的电脑上启动模型服务，当前电脑只通过 HTTP/API 调用远程模型。

## 3. 真实模型模式

换到有完整依赖和更好硬件的环境后，可以尝试加载 Qwen/LoRA：

```bash
python projects/geoai-assistant/chat.py --real-model --cpu --query "你是谁"
python projects/geoai-assistant/backend/app.py --cpu --port 8000
```

如果有 CUDA，去掉 `--cpu`，代码会优先走 4bit GPU 加载。

## 4. Harness 自动验收

Harness 是一层自动化验收外壳，用固定问题集验证链路是否稳定可用。它不会训练模型，而是检查：

- 概念问题是否能得到回答。
- 操作问题是否走 RAG 并返回 references。
- LoRA/demo 身份类问题是否能命中预期答案。
- 每个问题是否在限定时间内返回。
- 最终是否给出 pass/fail 汇总。

运行：

```bash
python projects/geoai-assistant/tests/harness.py
```

输出 JSON 报告：

```bash
python projects/geoai-assistant/tests/harness.py --json-output outputs/geoai_harness.json
```

真实模型环境下也可以复用同一个 harness：

```bash
python projects/geoai-assistant/tests/harness.py --real-model --cpu
```

## 5. 面试讲法

这个项目的重点不是本机跑出最强模型，而是展示完整工程能力：

- 数据构造：分类、匹配、LoRA 指令数据都有脚本和实验记录。
- 训练链路：BERT 分类、文本匹配、LoRA 微调入口都保留。
- 评估链路：有验证集、独立测试集、实验结果和训练曲线。
- 产品链路：GeoAI Assistant 提供 Router、RAG、LLMWorker、CLI、HTTP API 和 harness。
- 资源约束处理：无 GPU 时使用 demo/fallback 跑通链路；有 GPU 时切换到 LoRA/Qwen。

这比「只写一个训练脚本」更接近市面上大模型/NLP 岗位常问的闭环：训练、评估、推理、检索增强、服务化和工程降级。
