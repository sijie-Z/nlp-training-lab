"""
推理脚本

作用：加载训练好的最佳模型，对输入文本进行预测。

用法：
    # 单条预测
    python src/inference/predict.py --text "国足今晚迎战日本队"

    # 输出：
    # 文本：国足今晚迎战日本队
    # 类别：体育 (0.94)

流程：
1. 加载 best_model checkpoint（模型 + tokenizer + label 映射）
2. Tokenize 输入文本
3. 模型推理 → logits → softmax → argmax
4. 返回类别名称 + 置信度
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Predictor:
    """
    推理器（V0.1 极简版）

    没有 BertClassifier 包装，直接加载 HuggingFace 原生模型。
    """

    def __init__(self, checkpoint_dir, device=None):
        """
        参数：
            checkpoint_dir: checkpoint 目录
                            （如 "outputs/checkpoints/best_model"）
            device: "cuda" 或 "cpu"（默认自动选择）
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                        else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model from {checkpoint_dir}...")
        print(f"Using device: {self.device}")

        # 加载模型（直接用 HuggingFace 的 AutoModel）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir
        )
        self.model.to(self.device)
        self.model.eval()

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # 加载 label 映射
        import json
        import os
        mapping_path = f"{checkpoint_dir}/label_mapping.json"
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.id2label = {v: k for k, v in json.load(f).items()}
        else:
            # 如果没有 label_mapping，用数字作为标签
            num_labels = self.model.config.num_labels
            self.id2label = {i: str(i) for i in range(num_labels)}

    def predict(self, text: str, max_length: int = 128):
        """
        单条文本预测

        参数：
            text: 输入文本
            max_length: 最大长度

        返回：
            {"label": "体育", "confidence": 0.94}
        """
        # 1. Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. 推理（关掉梯度计算）
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. 概率 → argmax
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, pred_id = torch.max(probs, dim=-1)

        label = self.id2label[pred_id.item()]

        return {
            "label": label,
            "confidence": round(confidence.item(), 4)
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="BERT 新闻分类推理")
    parser.add_argument("--text", type=str, required=True,
                        help="输入文本")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/best_model",
                        help="模型 checkpoint 目录")
    parser.add_argument("--device", type=str, default=None,
                        help="设备（cuda/cpu）")
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, args.device)
    result = predictor.predict(args.text)

    print(f"\n文本：{args.text}")
    print(f"类别：{result['label']} ({result['confidence']})")


if __name__ == "__main__":
    main()
