"""
评估器

从 trainer.py 中独立提取的评估模块。
Phase 1（新闻分类）只需要 Accuracy，所以验证写在 trainer 内部。
Phase 2（文本匹配）需要 Precision/Recall/F1，独立出来更合理。

用法：
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    # → {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1": 0.84}
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    """评估器：支持 Accuracy、Precision、Recall、F1"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, dataloader, metrics=None):
        """
        在 DataLoader 上评估模型

        参数：
            dataloader: 评估数据
            metrics: 要计算的指标列表，默认 ["accuracy", "f1"]

        返回：
            {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1": 0.84}
        """
        if metrics is None:
            metrics = ["accuracy", "f1"]

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()
                         if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # 计算指标
        self.model.train()  # 恢复训练模式
        return compute_classification_metrics(all_labels, all_preds, metrics)


def compute_classification_metrics(y_true, y_pred, metrics_list=None):
    """
    纯函数：计算分类指标

    用法：
        compute_classification_metrics([0,1,0], [0,1,1], ["accuracy", "f1"])
    """
    if metrics_list is None:
        metrics_list = ["accuracy", "f1"]

    results = {}

    if "accuracy" in metrics_list:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    if "precision" in metrics_list:
        results["precision"] = precision_score(y_true, y_pred, average="binary")

    if "recall" in metrics_list:
        results["recall"] = recall_score(y_true, y_pred, average="binary")

    if "f1" in metrics_list:
        results["f1"] = f1_score(y_true, y_pred, average="binary")

    if "confusion" in metrics_list:
        from sklearn.metrics import confusion_matrix as cm
        results["confusion_matrix"] = cm(y_true, y_pred).tolist()

    return results
