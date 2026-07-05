"""
测试集评估脚本

作用：用训练好的模型在完全独立的测试集上评估，
衡量模型的"泛化能力"——即对没见过的数据的预测能力。

用法：
    python evaluate.py --checkpoint outputs/checkpoints/best_model --test_data data/splits/test_independent.csv
"""

import os
import sys
import argparse
import json
import csv

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestDataset(Dataset):
    """测试数据集——加载 CSV/TSV，自动判断格式"""

    def __init__(self, data_path, tokenizer, max_length=128, label2id=None, task_type="classification"):
        self.pairs = []  # for matching: list of (text_a, text_b)
        self.texts = []  # for classification: list of text
        self.labels = []
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            has_header = "text" in first_line and "label" in first_line
            f.seek(0)

            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", "")
                    label = row["label"]
                    if task_type == "match":
                        # CSV 格式不适用匹配数据，走 TSV 分支
                        pass
                    else:
                        self.texts.append(text)
                        self._add_label(label, label2id)
            else:
                # TSV 格式: text_a \t text_b \t label (匹配任务)
                # 或: text \t label (分类任务)
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        # 匹配任务: three columns
                        self.pairs.append((parts[0].strip(), parts[1].strip()))
                        label = parts[2].strip()
                        self._add_label(label, label2id)
                    elif len(parts) >= 2:
                        # 分类任务: two columns
                        self.texts.append(parts[0].strip())
                        label = parts[1].strip()
                        self._add_label(label, label2id)

    def _add_label(self, label_str, label2id):
        if label2id:
            # 如果 label2id 有对应关系，用映射
            if label_str in label2id:
                self.labels.append(label2id[label_str])
            else:
                # 否则尝试直接转数字
                try:
                    self.labels.append(int(label_str))
                except ValueError:
                    # 最后尝试反转 label2id 查找
                    rev = {v: k for k, v in label2id.items()}
                    self.labels.append(rev.get(label_str, 0))
        else:
            self.labels.append(int(label_str))

    def __len__(self):
        if self.task_type == "match" and self.pairs:
            return len(self.pairs)
        return len(self.texts)

    def __getitem__(self, idx):
        if self.task_type == "match" and self.pairs:
            text_a, text_b = self.pairs[idx]
            encoded = self.tokenizer(
                text_a, text_b,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }
        else:
            encoded = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }


def main():
    parser = argparse.ArgumentParser(description="评估模型在独立测试集上的表现")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model",
                        help="模型 checkpoint 目录")
    parser.add_argument("--test_data", type=str, default="data/splits/test_independent.csv",
                        help="测试集 CSV/TSV 路径")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--task_type", type=str, default=None,
                        help="任务类型: classification | match (默认自动检测)")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型 + tokenizer
    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # 加载 label 映射
    mapping_path = f"{args.checkpoint}/label_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
            id2label = {v: k for k, v in label2id.items()}
    else:
        label2id = {"0": 0, "1": 1, "2": 2, "3": 3}
        id2label = {0: "0", 1: "1", 2: "2", 3: "3"}

    print(f"Label mapping: {label2id}")
    print(f"Test data: {args.test_data}")

    # 自动检测任务类型
    task_type = args.task_type
    if task_type is None:
        with open(args.test_data, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            parts = first_line.split("\t")
            is_tsv_match = len(parts) >= 3 and parts[2] in ["0", "1"]
        task_type = "match" if is_tsv_match else "classification"
    print(f"Task type: {task_type}")

    # 加载测试集
    test_dataset = TestDataset(args.test_data, tokenizer, args.max_length, label2id, task_type)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # 评估
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)

            preds = outputs.logits.argmax(dim=-1)
            confidence, preds_max = torch.max(probs, dim=-1)

            all_preds.extend(preds_max.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(confidence.cpu().numpy())

    # ====== 按类别统计 ======
    per_class_correct = {label: {"correct": 0, "total": 0} for label in label2id.keys()}

    for pred, label, prob in zip(all_preds, all_labels, all_probs):
        # 找到这个 label 对应的中文标签名
        ch_label = id2label[label]
        per_class_correct[ch_label]["total"] += 1
        if pred == label:
            per_class_correct[ch_label]["correct"] += 1

    # ====== 整体指标 ======
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    avg_confidence = sum(all_probs) / len(all_probs) if all_probs else 0

    # ====== 输出结果 ======
    print("\n" + "=" * 60)
    print("   Test Set Evaluation Report")
    print("=" * 60)

    print(f"\n  Overall:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    Avg Conf:  {avg_confidence:.4f}")

    print(f"\n  Per-Class Breakdown:")
    print(f"    {'Label':<8s} {'Correct':>8s} {'Total':>6s} {'Acc':>8s}")
    print(f"    {'-'*30}")
    for label in label2id.keys():
        c = per_class_correct[label]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"    {label:<8s} {c['correct']:>4d}/{c['total']:<2d}        {acc:.2%}")

    # 错误样本
    print(f"\n  Misclassified Samples:")
    errors = 0
    with open(args.test_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # 跳过 CSV header
        start = 1 if "text" in lines[0] and "label" in lines[0] else 0
        for i in range(start, len(lines)):
            line_idx = i - start
            if line_idx >= len(all_preds):
                break
            if all_preds[line_idx] != all_labels[line_idx]:
                parts = lines[i].strip().split("\t")
                if task_type == "match" and len(parts) >= 2:
                    print(f"    [{line_idx}] A: {parts[0][:30]}...")
                    print(f"              B: {parts[1][:30]}...")
                else:
                    print(f"    [{line_idx}] text: {parts[0][:40]}...")
                true_label = id2label[all_labels[line_idx]]
                pred_label = id2label[int(all_preds[line_idx])]
                print(f"         true: {true_label}, pred: {pred_label}, conf: {all_probs[line_idx]:.4f}")
                errors += 1
                if errors >= 10:
                    print(f"    ... and {len(all_preds) - line_idx - 1} more misclassified")
                    break

    print("=" * 60)


if __name__ == "__main__":
    main()
