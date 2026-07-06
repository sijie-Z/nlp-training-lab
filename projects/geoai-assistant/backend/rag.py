"""
RAG 检索模块（轻量版）

知识库：JSON 文档
检索方式：优先 sklearn TF-IDF；未安装依赖时自动退回纯标准库词项匹配。

用法：
    rag = RAGWorker()
    docs = rag.search("什么是GIS")
"""

import os
import json
import math
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


class RAGWorker:
    """RAG 检索器（TF-IDF，零依赖）"""

    def __init__(self, docs_path=None):
        self.docs_path = docs_path or os.path.join(
            os.path.dirname(__file__), "../knowledge_base/demo_docs.json"
        )
        self.docs = []
        self.use_sklearn = TfidfVectorizer is not None and cosine_similarity is not None
        self.vectorizer = (
            TfidfVectorizer(tokenizer=self._tokenize, max_features=5000)
            if self.use_sklearn
            else None
        )
        self.tfidf_matrix = None
        self.doc_tokens = []
        self.doc_freq = Counter()
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
        if self.use_sklearn:
            self.tfidf_matrix = self.vectorizer.fit_transform(contents)
            print(f"[RAG] 已加载 {len(self.docs)} 篇文档 (sklearn TF-IDF)")
        else:
            self.doc_tokens = [Counter(self._tokenize(text)) for text in contents]
            for tokens in self.doc_tokens:
                self.doc_freq.update(tokens.keys())
            print(f"[RAG] 已加载 {len(self.docs)} 篇文档 (标准库检索)")

    def search(self, query, top_k=3):
        """检索 Top-K 相关文档"""
        if not self.docs:
            return []

        if self.use_sklearn:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            ranked = [(idx, float(scores[idx])) for idx in scores.argsort()[::-1]]
        else:
            query_tokens = Counter(self._tokenize(query))
            ranked = sorted(
                enumerate(self._fallback_scores(query_tokens)),
                key=lambda item: item[1],
                reverse=True,
            )

        results = []
        for idx, score in ranked[:min(top_k, len(ranked))]:
            if score > 0:
                results.append({
                    "id": self.docs[idx]["id"],
                    "content": self.docs[idx]["content"],
                    "title": self.docs[idx]["title"],
                    "category": self.docs[idx]["category"],
                    "score": round(float(score), 4),
                })
        return results

    def _fallback_scores(self, query_tokens):
        """纯标准库 BM25-like 打分，保证无 sklearn 时仍可演示链路。"""
        scores = []
        total_docs = max(len(self.doc_tokens), 1)
        for doc_counter in self.doc_tokens:
            score = 0.0
            doc_len = sum(doc_counter.values()) or 1
            for token, query_count in query_tokens.items():
                tf = doc_counter.get(token, 0)
                if not tf:
                    continue
                idf = math.log((total_docs + 1) / (self.doc_freq[token] + 1)) + 1
                score += query_count * (tf / doc_len) * idf
            scores.append(score)
        return scores


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
