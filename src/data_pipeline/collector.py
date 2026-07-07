"""
数据管线 — 语料采集模块
从知识库提取纯文本，输出干净的训练语料
"""
import json
import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KB_PATH = PROJECT_ROOT / "projects/geoai-assistant/knowledge_base/demo_docs.json"
RAW_CORPUS_PATH = PROJECT_ROOT / "data/raw/domain_corpus.txt"
CLEANED_CORPUS_PATH = PROJECT_ROOT / "data/raw/domain_corpus_cleaned.txt"


def load_knowledge_base(path=None):
    """加载知识库 JSON，提取所有纯文本"""
    path = path or KB_PATH
    if not os.path.exists(path):
        print(f"[Collector] 知识库不存在: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc in data:
        title = doc.get("title", "")
        content = doc.get("content", "")
        category = doc.get("category", "")
        keywords = doc.get("keywords", [])

        # 拼接为完整文档
        text = f"{title}\n{content}"
        documents.append({
            "id": doc.get("id", ""),
            "category": category,
            "text": text,
            "keywords": keywords,
        })

    print(f"[Collector] 加载 {len(documents)} 篇文档")
    return documents


def extract_raw_corpus(documents, output_path=None):
    """提取纯文本语料，一行一篇"""
    output_path = output_path or RAW_CORPUS_PATH
    lines = []
    for doc in documents:
        lines.append(doc["text"].strip())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))

    total_chars = sum(len(l) for l in lines)
    print(f"[Collector] 原始语料已保存: {output_path}")
    print(f"[Collector] {len(lines)} 行, {total_chars} 字符")
    return output_path


def clean_text(text):
    """清洗单段文本"""
    # 去除多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除纯符号行
    lines = [l for l in text.split("\n") if re.search(r"[一-鿿\w]", l)]
    return "\n".join(lines).strip()


def clean_corpus(input_path=None, output_path=None, min_chars=20, max_chars=5000):
    """清洗语料：去重、长度过滤、质量过滤"""
    input_path = input_path or RAW_CORPUS_PATH
    output_path = output_path or CLEANED_CORPUS_PATH

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 按双换行分割文档
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]

    cleaned = []
    seen = set()
    stats = {"too_short": 0, "too_long": 0, "duplicate": 0, "kept": 0}

    for doc in docs:
        text = clean_text(doc)

        # 去重（基于文本前100字符的hash）
        fingerprint = text[:100]
        if fingerprint in seen:
            stats["duplicate"] += 1
            continue
        seen.add(fingerprint)

        # 长度过滤
        if len(text) < min_chars:
            stats["too_short"] += 1
            continue
        if len(text) > max_chars:
            stats["too_long"] += 1
            continue

        cleaned.append(text)
        stats["kept"] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned))

    total_chars = sum(len(t) for t in cleaned)
    print(f"[Cleaner] 清洗完成: {output_path}")
    print(f"[Cleaner] 原始: {len(docs)} → 保留: {stats['kept']}")
    print(f"[Cleaner] 过滤: 太短{stats['too_short']}, 太长{stats['too_long']}, 重复{stats['duplicate']}")
    print(f"[Cleaner] 总字符数: {total_chars}")
    return output_path


def analyze_corpus(path=None):
    """分析语料统计信息"""
    path = path or CLEANED_CORPUS_PATH
    if not os.path.exists(path):
        print(f"[Analyzer] 文件不存在: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split("\n\n") if d.strip()]

    lengths = [len(d) for d in docs]
    lengths.sort()

    stats = {
        "total_docs": len(docs),
        "total_chars": sum(lengths),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": sum(lengths) / len(lengths) if lengths else 0,
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "p10_chars": lengths[len(lengths) // 10] if len(lengths) >= 10 else lengths[0] if lengths else 0,
        "p90_chars": lengths[len(lengths) * 9 // 10] if len(lengths) >= 10 else lengths[-1] if lengths else 0,
    }

    print(f"[Analyzer] 语料分析:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    return stats


# ============================================================
# 语料扩充：把知识库文档拆成句子级别的训练数据
# 用于 tokenizer 训练时可以学到更多中文词汇
# ============================================================
def extract_sentences(documents, output_path=None):
    """从文档中拆出句子，用于 tokenizer 训练"""
    output_path = output_path or str(PROJECT_ROOT / "data/raw/sentences.txt")

    all_sentences = []
    for doc in documents:
        text = doc["text"]
        # 中英文分句
        sentences = re.split(r"[。！？\n;；]", text)
        for s in sentences:
            s = s.strip()
            # 至少包含2个中文字符或5个英文字符
            cn_chars = len(re.findall(r"[一-鿿]", s))
            en_words = len(re.findall(r"[a-zA-Z]+", s))
            if cn_chars >= 2 or en_words >= 3:
                all_sentences.append(s)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_sentences))

    print(f"[Collector] 句子级语料: {output_path} ({len(all_sentences)} 句)")
    return output_path


if __name__ == "__main__":
    # 完整管线
    docs = load_knowledge_base()
    extract_raw_corpus(docs)
    clean_corpus()
    analyze_corpus()
    extract_sentences(docs)
