"""
训练自己的 BPE Tokenizer
对比 Qwen 官方 tokenizer 的编码效率
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data/raw/domain_corpus_cleaned.txt"
SENTENCES_PATH = PROJECT_ROOT / "data/raw/sentences.txt"


def train_bpe_tokenizer(
    corpus_path=None,
    vocab_size=8000,
    output_dir=None,
    min_frequency=2,
):
    """使用 tokenizers 库训练 BPE tokenizer"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    corpus_path = corpus_path or CORPUS_PATH
    output_dir = output_dir or str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain")
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 预分词器：按 Unicode 字符边界拆分
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 解码器: ByteLevel
    tokenizer.decoder = decoders.ByteLevel()

    # 后处理: 添加特殊 token
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 准备训练语料
    files = [str(corpus_path)]
    if os.path.exists(str(SENTENCES_PATH)):
        files.append(str(SENTENCES_PATH))

    print(f"[Tokenizer] 开始训练 BPE (vocab_size={vocab_size})")
    print(f"[Tokenizer] 训练文件: {files}")
    t0 = time.time()

    tokenizer.train(files, trainer)

    elapsed = time.time() - t0
    print(f"[Tokenizer] 训练完成，耗时 {elapsed:.1f}s")
    print(f"[Tokenizer] 词表大小: {tokenizer.get_vocab_size()}")

    # 保存
    model_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(model_path)
    print(f"[Tokenizer] 已保存: {model_path}")

    # 输出词表统计
    vocab = tokenizer.get_vocab()
    top_items = sorted(vocab.items(), key=lambda x: x[1])[:20]
    top_display = [(t.replace("\n","\\n"), i) for t, i in top_items]
    print(f"[Tokenizer] 前20个token: {top_display}")

    return tokenizer, model_path


def train_unigram_tokenizer(
    corpus_path=None,
    vocab_size=8000,
    output_dir=None,
):
    """训练 Unigram tokenizer（SentencePiece 风格）作为对比"""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    corpus_path = corpus_path or CORPUS_PATH
    output_dir = output_dir or str(PROJECT_ROOT / "outputs/tokenizers/unigram_domain")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"],
        unk_token="[UNK]",
        show_progress=True,
    )

    files = [str(corpus_path)]
    if os.path.exists(str(SENTENCES_PATH)):
        files.append(str(SENTENCES_PATH))

    print(f"[Tokenizer] 开始训练 Unigram (vocab_size={vocab_size})")
    t0 = time.time()
    tokenizer.train(files, trainer)
    elapsed = time.time() - t0
    print(f"[Tokenizer] Unigram 训练完成，耗时 {elapsed:.1f}s")
    print(f"[Tokenizer] 词表大小: {tokenizer.get_vocab_size()}")

    model_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(model_path)
    print(f"[Tokenizer] 已保存: {model_path}")
    return tokenizer, model_path


def load_qwen_tokenizer():
    """加载 Qwen2.5 的 tokenizer 用于对比"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        print(f"[Tokenizer] Qwen tokenizer 加载成功，词表: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        print(f"[Tokenizer] Qwen 加载失败（可能无网络/未安装）: {e}")
        return None


def compare_encoding_efficiency(my_tokenizer, qwen_tokenizer, texts):
    """对比编码效率：token 数量越少 = 效率越高"""
    results = []
    for i, text in enumerate(texts):
        # 用自己的 tokenizer 编码
        my_tokens = my_tokenizer.encode(text).ids
        my_len = len(my_tokens)

        # 用 Qwen 编码
        if qwen_tokenizer:
            qwen_tokens = qwen_tokenizer.encode(text)
            qwen_len = len(qwen_tokens)
            ratio = my_len / qwen_len if qwen_len > 0 else 0
        else:
            qwen_len = 0
            ratio = 0

        results.append({
            "text_preview": text[:80],
            "my_tokens": my_len,
            "qwen_tokens": qwen_len,
            "ratio": ratio,
            "my_efficient": ratio < 1.0,
        })

    # 汇总
    avg_ratio = sum(r["ratio"] for r in results if r["qwen_tokens"] > 0) / max(
        sum(1 for r in results if r["qwen_tokens"] > 0), 1
    )
    my_efficient_count = sum(1 for r in results if r["my_efficient"])

    print(f"\n[Compare] 平均编码比 (my/qwen): {avg_ratio:.2f} ({'<' if avg_ratio < 1 else '>'}1 = {'更高效' if avg_ratio < 1 else '更低效'})")
    print(f"[Compare] 自己训练的更高效: {my_efficient_count}/{len(results)}")

    return results


def run_full_pipeline(vocab_size=8000, compare_with_qwen=True):
    """完整流程：数据准备 → 训练 → 对比"""
    # 1. 准备数据
    print("=" * 60)
    print("Step 1: 准备语料")
    print("=" * 60)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data_pipeline import collector

    docs = collector.load_knowledge_base()
    collector.extract_raw_corpus(docs)
    collector.clean_corpus()
    collector.analyze_corpus()
    collector.extract_sentences(docs)

    # 2. 训练 BPE
    print("\n" + "=" * 60)
    print("Step 2: 训练 BPE Tokenizer")
    print("=" * 60)
    bpe_tokenizer, bpe_path = train_bpe_tokenizer(vocab_size=vocab_size)

    # 3. 训练 Unigram
    print("\n" + "=" * 60)
    print("Step 3: 训练 Unigram Tokenizer（对比）")
    print("=" * 60)
    uni_tokenizer, uni_path = train_unigram_tokenizer(vocab_size=vocab_size)

    # 4. 对比 Qwen
    if compare_with_qwen:
        print("\n" + "=" * 60)
        print("Step 4: 编码效率对比")
        print("=" * 60)
        qwen_tok = load_qwen_tokenizer()
        if qwen_tok:
            test_texts = [
                "什么是遥感技术？遥感是指通过卫星或航空器远距离感知地物信息的技术。",
                "QGIS是一个开源的地理信息系统软件，支持shp文件导入和空间分析。",
                "NDVI归一化植被指数计算公式为(NIR-Red)/(NIR+Red)，用于评估植被覆盖度。",
                "坐标系转换是GIS中的常见操作，WGS84和CGCS2000是不同的参考椭球。",
                "地理信息系统GIS的空间数据包括矢量数据和栅格数据两种类型。",
                "ArcGIS中可以使用缓冲区分析工具进行空间邻近性分析。",
                "The Geographic Information System (GIS) integrates hardware, software, and data.",
                "pyproj是一个Python库，用于坐标系统转换和投影。",
                "GeoJSON是一种基于JSON的地理空间数据交换格式。",
                "DEM数字高程模型是地形分析的基础数据，可以用于水文分析和视域分析。",
            ]
            print(f"\n[Compare] 测试文本: {len(test_texts)} 条")
            results = compare_encoding_efficiency(bpe_tokenizer, qwen_tok, test_texts)

            # 打印详细结果
            for r in results:
                flag = "✓ 更高效" if r["my_efficient"] else "✗"
                print(f"  {flag} my={r['my_tokens']:3d} qwen={r['qwen_tokens']:3d} ratio={r['ratio']:.2f} | {r['text_preview']}")

    # 5. 保存结果
    summary = {
        "vocab_size": vocab_size,
        "bpe_vocab_size": bpe_tokenizer.get_vocab_size(),
        "bpe_model": bpe_path,
        "unigram_vocab_size": uni_tokenizer.get_vocab_size(),
        "unigram_model": uni_path,
        "corpus": str(CORPUS_PATH),
    }
    summary_path = str(PROJECT_ROOT / "outputs/tokenizers/summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Summary] {summary_path}")

    return bpe_tokenizer, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 BPE/Unigram Tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000, help="词表大小")
    parser.add_argument("--no_compare", action="store_true", help="跳过 Qwen 对比")
    args = parser.parse_args()

    # 确保 src 在 path 中
    sys.path.insert(0, str(PROJECT_ROOT))

    run_full_pipeline(vocab_size=args.vocab_size, compare_with_qwen=not args.no_compare)
