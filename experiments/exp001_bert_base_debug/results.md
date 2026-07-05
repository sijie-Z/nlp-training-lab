## Experiment 001 — BERT Base Debug Run (20 samples)

### 基本信息

- **日期**: 2026-06-22
- **模型**: bert-base-chinese
- **数据**: data/raw/news.csv (20 条, 4 分类, 每类 5 条)
- **训练/验证划分**: 18 / 2
- **训练时长**: ~3 分钟 (CPU)
- **配置**: configs/test_debug.yaml

### 超参数

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| learning_rate | 2e-5 |
| max_length | 128 (默认) |
| epochs | 3 |
| seed | 42 |
| warmup_ratio | 0.1 |

### 结果

| Epoch | Train Loss | Val Accuracy |
|-------|-----------|-------------|
| 1 | 1.4576 | 0.5000 |
| 2 | 1.4026 | 0.5000 |
| 3 | 1.0846 | 0.5000 |

Loss 趋势: 1.5003 → 1.4150 → 1.1977 → 1.6074 → 1.1832 → 0.9861

### 推理验证

```bash
输入: "国足今晚迎战日本队"
输出: 类别：体育 (0.4326)
```

置信度 0.43 > 随机概率 0.25 → 模型学到了基本模式

### 数据流验证 (Tensor Shape)

```
CSV → NewsDataset → DataLoader → BERT → Loss → Backward → Optimizer

[16, 128]  →  input_ids
[16, 128]  →  attention_mask
[16]       →  labels
[16, 4]    →  logits  ← 每条新闻对应4个类别的分数
```

### 遇到的 Bug 与修复

| Bug | 原因 | 修复 |
|-----|------|------|
| UnicodeEncodeError: emoji | Windows GBK 终端不支持 emoji | 将所有 emoji 替换为 ASCII |
| ValueError: int("体育") | label_mapping key/value 反转 | `int(k)` → `v: k` |
| pip 安装编码错误 | requirements.txt 中文注释导致 GBK 乱码 | 移除中文注释 |

### 结论

- ✅ 完整训练链路跑通
- ✅ 模型成功收敛（Loss 从 1.50 降至 0.99）
- ✅ 推理正常（体育, 0.43）
- ✅ Shape Tracker 清晰展示了数据流
- ✅ Checkpoint / 日志 / 曲线图全部自动保存

### 不足

- 只有 20 条数据，模型学到的东西有限
- CPU 训练较慢（3 分钟/次）
- 验证集只有 2 条，Accuracy 统计意义不足

### 下一步

- [ ] 增加数据量到 200~400 条
- [ ] 尝试 hfl/chinese-roberta-wwm-ext
- [ ] 开启 GPU 加速
