"""
新闻分类数据集

作用：把 CSV 数据变成模型能吃的 Tensor。

流程：
1. __init__: 读取 CSV，加载 Tokenizer，建立 label↔id 映射
2. __getitem__: 取出一条文本 → tokenize → 返回 Tensor 字典

返回格式：
    {
        "input_ids": Tensor,       # (seq_len,) 文本转成的数字编号
        "attention_mask": Tensor,  # (seq_len,) 1=真实文本，0=填充
        "labels": Tensor           # (1,) 类别数字标签
    }

数据格式（CSV）：
    text,label
    "国足今晚迎战日本队",体育
    "美国大选最新进展",政治
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """新闻分类数据集"""

    def __init__(self, csv_path, tokenizer, max_length,
                 text_col="text", label_col="label",
                 label2id=None):
        """
        参数：
            csv_path: CSV 文件路径
            tokenizer: HuggingFace tokenizer 实例
            max_length: 最大序列长度
            text_col: CSV 中文本列名
            label_col: CSV 中标签列名
            label2id: 标签→数字映射
                      （首次传入 None，会自动从数据中构建）
        """
        # 1. 读取 CSV
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

        # 2. 构建或复用 label→id 映射
        if label2id is None:
            unique_labels = self.df[label_col].unique()
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label2id = label2id

        # 3. 保存 id→label 映射（推理时用）
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # 4. 将所有标签转为数字
        self.labels = [self.label2id[l] for l in self.df[label_col]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        取出一条数据

        流程：
        1. 取文本和标签
        2. Tokenizer 编码（自动 padding + truncation）
        3. 返回 Tensor 字典
        """
        text = str(self.df.iloc[idx][self.text_col])
        label = self.labels[idx]

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (seq_len,)
            "labels": torch.tensor(label, dtype=torch.long)     # (1,)
        }
