"""
RAG 检索模块（轻量版，纯 Python）

知识库：JSON 文档
检索方式：TF-IDF 余弦相似度（无需 GPU、无需下载模型）

用法：
    rag = RAGWorker()
    docs = rag.search("什么是GIS")
"""

import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGWorker:
    """RAG 检索器（TF-IDF，零依赖）"""

    def __init__(self, docs_path=None):
        self.docs_path = docs_path or os.path.join(
            os.path.dirname(__file__), "../knowledge_base/demo_docs.json"
        )
        self.docs = []
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize, max_features=5000)
        self.tfidf_matrix = None
        self._load()

    def _tokenize(self, text):
        """简单中文分词（按字 + 按词拆分）"""
        # 单字 + 双字组合，覆盖中文关键词
        chars = list(text)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        return chars + bigrams

    def _load(self):
        if not os.path.exists(self.docs_path):
            print(f"[RAG] 知识库文件不存在: {self.docs_path}")
            return

        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        contents = [d["content"] + " " + d["title"] + " " + ", ".join(d["keywords"])
                    for d in self.docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        print(f"[RAG] 已加载 {len(self.docs)} 篇文档 (TF-IDF)")

    def search(self, query, top_k=3):
        """检索 Top-K 相关文档"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:min(top_k, len(scores))]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append({
                    "id": self.docs[idx]["id"],
                    "content": self.docs[idx]["content"],
                    "title": self.docs[idx]["title"],
                    "category": self.docs[idx]["category"],
                    "score": round(float(scores[idx]), 4),
                })
        return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是GIS"
    print(f"\n[RAG Test] Query: {query}\n")
    rag = RAGWorker()
    results = rag.search(query)
    print(f"Top {len(results)} results:")
    for r in results:
        print(f"\n  [{r['title']}] (score: {r['score']})")
        print(f"  {r['content'][:120]}...")
