"""
统一 Chat Pipeline

流程：
1. Router 判断问题类型
2. 如果是操作类问题 → RAG 检索知识库
3. 如果是概念类问题 → LoRA 模型直接回答
4. 返回最终回答 + 来源标记

用法：
    pipeline = ChatPipeline()
    result = pipeline.chat("什么是GIS")
    # {"answer": "GIS是...", "source": "lora"}
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import Router
from rag import RAGWorker


class ChatPipeline:
    """统一问答管道"""

    def __init__(self):
        print("[Pipeline] 初始化 Router...")
        self.router = Router()

        print("[Pipeline] 初始化 RAG...")
        self.rag = RAGWorker()

        print("[Pipeline] 初始化 LLM...")
        from llm import LLMWorker
        self.llm = LLMWorker()

        print("[Pipeline] 就绪 ✓")

    def chat(self, query):
        """
        处理用户查询

        返回：
            {"answer": str, "source": "lora"|"rag", "references": [...]|None}
        """
        route = self.router.route(query)

        if route["type"] == "rag":
            # RAG 检索 + LLM 增强
            docs = self.rag.search(query, top_k=2)
            if docs:
                # 用检索到的文档作为上下文，让 LLM 生成回答
                context = "\n\n".join([d["content"] for d in docs])
                rag_prompt = (
                    f"### Instruction:\n基于以下资料回答问题。\n\n"
                    f"资料：\n{context}\n\n"
                    f"问题：{query}\n\n### Response:\n"
                )
                answer = self.llm.generate_with_prompt(rag_prompt, max_new_tokens=150)
                # 解析回答（去掉 prompt 部分）
                if "### Response:\n" in answer:
                    answer = answer.split("### Response:\n")[-1].strip()
                return {
                    "answer": answer,
                    "source": "rag",
                    "references": [{"title": d["title"], "score": d["score"]} for d in docs],
                }
            # 没检索到，回退到 LLM
            route["type"] = "llm"

        # LLM 直接回答
        answer = self.llm.generate(query, max_new_tokens=150)
        return {
            "answer": answer,
            "source": "lora",
            "references": None,
        }


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是遥感"
    print(f"\n[Pipeline Test] Query: {query}\n")

    pipeline = ChatPipeline()
    result = pipeline.chat(query)

    print(f"\nResult:")
    print(f"  Answer: {result['answer'][:200]}")
    print(f"  Source: {result['source']}")
    if result["references"]:
        print(f"  References: {result['references']}")
