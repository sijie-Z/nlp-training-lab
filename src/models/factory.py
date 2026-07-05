"""
模型工厂

作用：返回一个 HuggingFace 模型，够用就行。

V0.1 不封装，直接返回 AutoModelForSequenceClassification。
以后需要抽象（RoBERTa、LoRA、Qwen）时再逐步加封装。

为什么不用 BertClassifier？
- V0.1 的目标是理解训练流程，不是设计模型抽象层
- 提前封装容易：花 2 小时设计架构，花 10 分钟训练——本末倒置
"""

from transformers import AutoModelForSequenceClassification


def build_model(model_name: str, num_classes: int, device: str):
    """
    构建模型

    就一行：AutoModelForSequenceClassification

    参数：
        model_name: 预训练模型名称
            - "bert-base-chinese"（默认）
            - "hfl/chinese-roberta-wwm-ext"
            - "hfl/chinese-macbert-base"
        num_classes: 分类数
        device: "cuda" 或 "cpu"

    返回：
        model（已移到 device 上）
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    model = model.to(device)
    return model
