"""
生成式模型评测体系 (v3.0)

覆盖:
  1. BLEU (n-gram 精度)
  2. ROUGE-L (最长公共子序列)
  3. 关键词匹配分数
  4. 字符覆盖率
  5. 多模型对比框架

用法:
    python src/evaluators/gen_evaluator.py --checkpoint outputs/checkpoints/pretrain_tiny/best_model.pt
    python src/evaluators/gen_evaluator.py --compare_all  # 对比所有训练阶段模型
"""
import json
import os
import sys
import time
import math
from pathlib import Path
from collections import Counter
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from tokenizers import Tokenizer
from src.models.tiny_gpt import TinyGPT


def compute_bleu(reference, candidate, max_n=4):
    """字符级 BLEU（适配中文）"""
    def get_ngrams(text, n):
        chars = list(text)
        ngrams = [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]
        return Counter(ngrams)

    ref_chars = len(reference)
    cand_chars = len(candidate)
    if cand_chars == 0:
        return 0.0

    bp = 1.0 if cand_chars >= ref_chars else math.exp(1 - ref_chars / max(cand_chars, 1))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)
        matches = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(matches / total)

    if all(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions if p > 0) / len(precisions)
    return bp * math.exp(log_avg)


def compute_rouge_l(reference, candidate):
    """ROUGE-L: 基于最长公共子序列的 F1"""
    def lcs_len(s1, s2):
        if not s1 or not s2:
            return 0
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if s1[i-1] == s2[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[len(s1)][len(s2)]

    if not reference or not candidate:
        return 0.0
    lcs_len_val = lcs_len(reference, candidate)
    recall = lcs_len_val / max(len(reference), 1)
    precision = lcs_len_val / max(len(candidate), 1)
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_keyword_score(keywords, text):
    if not keywords:
        return 0.0
    return sum(1 for kw in keywords if kw.lower() in text.lower()) / len(keywords)


def compute_coverage(reference, candidate):
    ref_set = set(reference)
    if not ref_set:
        return 0.0
    return len(ref_set & set(candidate)) / len(ref_set)


def evaluate_single(model, tokenizer, item, max_new_tokens=80, device="cpu"):
    question = item["question"]
    reference = item.get("reference_answer", "")
    keywords = item.get("keywords", [])

    enc = tokenizer.encode(question)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
        candidate = tokenizer.decode(output_ids[0].tolist())

    # 去掉 prompt
    if question in candidate:
        candidate = candidate[candidate.index(question) + len(question):]

    bleu = compute_bleu(reference, candidate)
    rouge_l = compute_rouge_l(reference, candidate)
    kw_score = compute_keyword_score(keywords, candidate)
    coverage = compute_coverage(reference, candidate)
    composite = 0.25 * bleu + 0.30 * rouge_l + 0.30 * kw_score + 0.15 * coverage

    return {
        "question_id": item["id"],
        "question": question[:60],
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "candidate": candidate[:200],
        "reference": reference[:150],
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge_l, 4),
        "keyword_score": round(kw_score, 4),
        "coverage": round(coverage, 4),
        "composite": round(composite, 4),
    }


def evaluate_benchmark(model, tokenizer, benchmark_path, max_new_tokens=80, device="cpu", sample_size=None):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    questions = benchmark["questions"]
    if sample_size:
        questions = questions[:sample_size]

    print(f"[Eval] {len(questions)} 题, 评测中...")
    t0 = time.time()
    results = [evaluate_single(model, tokenizer, q, max_new_tokens, device) for q in questions]

    composites = [r["composite"] for r in results]
    bleus = [r["bleu"] for r in results]
    rouges = [r["rouge_l"] for r in results]
    kw = [r["keyword_score"] for r in results]

    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r["composite"])

    by_diff = {}
    for r in results:
        diff = r["difficulty"]
        by_diff.setdefault(diff, []).append(r["composite"])

    summary = {
        "total_questions": len(results),
        "time_s": round(time.time() - t0, 1),
        "avg_composite": round(sum(composites) / len(composites), 4),
        "avg_bleu": round(sum(bleus) / len(bleus), 4),
        "avg_rouge_l": round(sum(rouges) / len(rouges), 4),
        "avg_keyword_score": round(sum(kw) / len(kw), 4),
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in by_cat.items()},
        "by_difficulty": {k: round(sum(v) / len(v), 4) for k, v in by_diff.items()},
        "per_question": results,
    }

    print(f"[Eval] 综合={summary['avg_composite']:.4f} BLEU={summary['avg_bleu']:.4f} "
          f"ROUGE={summary['avg_rouge_l']:.4f} KW={summary['avg_keyword_score']:.4f}")
    return summary


def compare_models(model_configs, tokenizer_path, benchmark_path, output_path, device="cpu", sample_size=None):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    all_results = {}

    for mc in model_configs:
        print(f"\n{'='*60}")
        print(f"[Compare] {mc['label']}")
        ckpt = torch.load(mc["checkpoint"], map_location=device, weights_only=False)
        config = ckpt["config"]
        model = TinyGPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        summary = evaluate_benchmark(model, tokenizer, benchmark_path, device=device, sample_size=sample_size)
        all_results[mc["name"]] = summary

    print(f"\n{'='*80}")
    print(f"{'模型':<30s} {'综合':>8s} {'BLEU':>8s} {'ROUGE':>8s} {'KW':>8s}")
    print("-" * 65)

    comparison = {}
    for name, s in all_results.items():
        label = next((mc["label"] for mc in model_configs if mc["name"] == name), name)
        print(f"{label:<30s} {s['avg_composite']:>8.4f} {s['avg_bleu']:>8.4f} {s['avg_rouge_l']:>8.4f} {s['avg_keyword_score']:>8.4f}")
        comparison[name] = {"label": label, "avg_composite": s["avg_composite"],
                            "avg_bleu": s["avg_bleu"], "avg_rouge_l": s["avg_rouge_l"],
                            "avg_keyword_score": s["avg_keyword_score"]}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark": Path(benchmark_path).stem, "comparison": comparison, "details": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[Compare] 保存: {output_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成式模型评测")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizers/bpe_domain/tokenizer.json")
    parser.add_argument("--benchmark", type=str, default="data/benchmark/gis_benchmark.json")
    parser.add_argument("--output", type=str, default="outputs/benchmarks/eval_result.json")
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--sample", type=int, default=None, help="只评测前N题")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = str(PROJECT_ROOT / args.tokenizer)
    bench_path = str(PROJECT_ROOT / args.benchmark)

    if args.compare_all:
        pretrain_dir = PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny"
        models = [
            {"name": "epoch00", "checkpoint": str(pretrain_dir / "checkpoint_epoch000.pt"), "label": "预训练 Epoch 0 (随机)"},
            {"name": "epoch10", "checkpoint": str(pretrain_dir / "checkpoint_epoch010.pt"), "label": "预训练 Epoch 10 (PPL 80)"},
            {"name": "epoch20", "checkpoint": str(pretrain_dir / "checkpoint_epoch020.pt"), "label": "预训练 Epoch 20 (PPL 4)"},
            {"name": "epoch30", "checkpoint": str(pretrain_dir / "checkpoint_epoch030.pt"), "label": "预训练 Epoch 30 (PPL 1.3)"},
            {"name": "epoch49", "checkpoint": str(pretrain_dir / "best_model.pt"), "label": "预训练 Epoch 49 (PPL 1.1)"},
        ]
        dpo_path = str(PROJECT_ROOT / "outputs/checkpoints/dpo_aligned/dpo_model.pt")
        if os.path.exists(dpo_path):
            models.append({"name": "dpo", "checkpoint": dpo_path, "label": "DPO 对齐 (β=0.5)"})

        out = str(PROJECT_ROOT / args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        compare_models(models, tok_path, bench_path, out, device=device, sample_size=args.sample)

    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = TinyGPT(config).to(device)
        model.load_state_dict(ckpt["model"])
        tokenizer = Tokenizer.from_file(tok_path)
        summary = evaluate_benchmark(model, tokenizer, bench_path, device=device, sample_size=args.sample)
        out = str(PROJECT_ROOT / args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Eval] 保存: {out}")
