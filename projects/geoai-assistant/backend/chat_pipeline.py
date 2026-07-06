"""
统一 Chat Pipeline (v2.0)

流程：
1. Router (BERT/关键词) 判断问题类型
2. 操作类 → RAG 检索知识库 → LLM 增强回答
3. 概念/闲聊类 → LoRA 模型直接回答
4. 返回最终回答 + 来源标记 + 耗时

支持 GPU (CUDA 4bit) 和 CPU 模式。

用法：
    pipeline = ChatPipeline()
    result = pipeline.chat("什么是GIS")
    # {"answer": "GIS是...", "source": "lora", "time_ms": 1234}
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import Router
from rag import RAGWorker


class ChatPipeline:
    """统一问答管道 (v2.0)"""

    def __init__(self, force_cpu=False, demo_mode=None):
        t0 = time.time()

        print("[Pipeline] 初始化 Router (BERT + 关键词回退)...")
        self.router = Router()

        print("[Pipeline] 初始化 RAG (TF-IDF, 104 篇文档)...")
        self.rag = RAGWorker()

        print("[Pipeline] 初始化 LLM...")
        from llm import LLMWorker
        self.llm = LLMWorker(force_cpu=force_cpu, demo_mode=demo_mode)

        elapsed = time.time() - t0
        print(f"[Pipeline] 就绪 (总耗时 {elapsed:.1f}s)")

    def chat(self, query):
        """
        处理用户查询

        返回：
            {"answer": str, "source": "lora"|"rag", "time_ms": int, "references": [...]|None}
        """
        t0 = time.time()

        # 1. 路由判断
        route = self.router.route(query)

        if route["type"] == "rag":
            # 2a. RAG 检索 + LLM 增强
            docs = self.rag.search(query, top_k=3)
            if docs:
                context = "\n\n".join([
                    f"[{d['title']}] {d['content']}" for d in docs
                ])
                rag_prompt = (
                    f"### Instruction:\n"
                    f"基于以下参考资料回答问题。如果资料中没有相关内容，请如实说明。\n\n"
                    f"参考资料：\n{context}\n\n"
                    f"问题：{query}\n\n"
                    f"### Response:\n"
                )
                answer = self.llm.generate_with_prompt(rag_prompt, max_new_tokens=200)

                # 提取 Response 部分
                if "### Response:\n" in answer:
                    answer = answer.split("### Response:\n")[-1].strip()

                elapsed = int((time.time() - t0) * 1000)
                return {
                    "answer": answer,
                    "source": "rag",
                    "time_ms": elapsed,
                    "references": [
                        {"title": d["title"], "score": d["score"]} for d in docs
                    ],
                }

            # 没检索到相关文档，回退到 LLM
            route["type"] = "llm"

        # 2b. LLM 直接回答
        answer = self.llm.generate(query, max_new_tokens=150)
        elapsed = int((time.time() - t0) * 1000)

        return {
            "answer": answer,
            "source": "lora",
            "time_ms": elapsed,
            "references": None,
        }


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是遥感"
    print(f"\n[Pipeline Test] Query: {query}\n")

    pipeline = ChatPipeline()
    result = pipeline.chat(query)

    print(f"\n{'='*60}")
    print(f"Answer:   {result['answer'][:300]}")
    print(f"Source:   {result['source']}")
    print(f"Time:     {result['time_ms']}ms")
    if result.get("references"):
        print(f"Refs:     {[r['title'] for r in result['references']]}")
    print(f"{'='*60}")
