"""
Tokenizer 辅助：向现有 tokenizer 添加新 token
查看/分析词表内容
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def add_tokens_to_tokenizer(tokenizer_path, new_tokens, output_path=None):
    """向已训练的 tokenizer 添加领域专用 token"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    old_size = tokenizer.get_vocab_size()
    added = tokenizer.add_tokens(new_tokens)
    new_size = tokenizer.get_vocab_size()
    print(f"[AddTokens] 词表: {old_size} → {new_size} (新增 {added})")

    output_path = output_path or tokenizer_path
    tokenizer.save(output_path)
    print(f"[AddTokens] 已保存: {output_path}")
    return tokenizer


def save_vocab_list(tokenizer_path, output_path):
    """导出词表为可读文本"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()
    # 按 id 排序
    items = sorted(vocab.items(), key=lambda x: x[1])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 词表大小: {len(items)}\n")
        for token, idx in items:
            # 把不可见字符转成可读形式
            display = token.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␣")
            f.write(f"{idx}\t{display}\n")

    print(f"[Vocab] 词表已导出: {output_path} ({len(items)} tokens)")
    return output_path


def analyze_vocab(tokenizer_path):
    """分析词表组成"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = tokenizer.get_vocab()

    cn_tokens = []
    en_tokens = []
    digit_tokens = []
    special_tokens = []
    other_tokens = []

    for token, idx in vocab.items():
        if token.startswith("[") and token.endswith("]"):
            special_tokens.append(token)
        elif token.startswith("<") and token.endswith(">"):
            special_tokens.append(token)
        elif any("一" <= c <= "鿿" for c in token):
            cn_tokens.append(token)
        elif any(c.isalpha() for c in token):
            en_tokens.append(token)
        elif any(c.isdigit() for c in token):
            digit_tokens.append(token)
        else:
            other_tokens.append(token)

    print(f"[VocabAnalysis] 总 token: {len(vocab)}")
    print(f"  中文字符: {len(cn_tokens)}")
    print(f"  英文字符: {len(en_tokens)}")
    print(f"  数字: {len(digit_tokens)}")
    print(f"  特殊token: {len(special_tokens)}")
    print(f"  其他: {len(other_tokens)}")

    # 列出所有特殊 token
    print(f"\n  特殊token: {special_tokens}")

    return {
        "total": len(vocab),
        "cn": len(cn_tokens),
        "en": len(en_tokens),
        "digit": len(digit_tokens),
        "special": len(special_tokens),
        "other": len(other_tokens),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer.json 路径")
    parser.add_argument("--export_vocab", type=str, default=None, help="导出词表文件路径")
    parser.add_argument("--analyze", action="store_true", help="分析词表组成")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")

    if args.analyze:
        analyze_vocab(tokenizer_path)

    if args.export_vocab:
        save_vocab_list(tokenizer_path, args.export_vocab)
