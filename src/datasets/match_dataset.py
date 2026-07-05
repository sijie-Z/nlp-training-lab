"""
文本匹配数据集

处理句子对输入，用于 LCQMC 等匹配任务。

数据格式（每行一条）：
    今天苏州下雨了吗\t苏州现在有降水吗\t1
    （sentence_a、sentence_b、label）

返回格式：
    {
        "input_ids": Tensor,        # [CLS] text_a [SEP] text_b [SEP]
        "attention_mask": Tensor,
        "token_type_ids": Tensor,   # 区分句子A和句子B
        "labels": Tensor            # 0=不匹配, 1=匹配
    }
"""

import torch
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    """文本匹配数据集（双句输入）"""

    def __init__(self, data_path, tokenizer, max_length=128):
        """
        参数：
            data_path: 数据文件路径（TSV 格式，三列：text_a, text_b, label）
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.pairs = []
        self.labels = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    self.pairs.append((parts[0].strip(), parts[1].strip()))
                    self.labels.append(int(parts[2].strip()))

        self.tokenizer = tokenizer
        self.max_length = max_length

        assert len(self.pairs) == len(self.labels), \
            f"数据量不匹配: {len(self.pairs)} pairs vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_a, text_b = self.pairs[idx]
        label = self.labels[idx]

        # 双句编码: [CLS] text_a [SEP] text_b [SEP]
        encoded = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros(self.max_length)).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
