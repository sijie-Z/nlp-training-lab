"""
Tensor Shape 追踪器

作用：打印数据流过模型每一步的 tensor shape。
让新手能亲眼看到数据在每一层之后变成了什么形状。

用法：
    tracker = ShapeTracker()

    # 在训练循环中插桩
    tracker.track("输入", input_ids=batch["input_ids"])

    # 模型 forward 之后
    tracker.track("Logits", logits=outputs.logits)

    # 打印总结
    tracker.print_summary()

输出示例：
    ============================
    📐 Shape Track Summary
    ============================
      [输入]        input_ids:       [16, 128]
      [Attention]   attention_mask:  [16, 128]
      [Logits]      logits:          [16, 4]
    ============================
"""


class ShapeTracker:
    """Tensor Shape 追踪器"""

    def __init__(self, enabled: bool = True):
        """
        参数：
            enabled: 是否启用追踪（设为 False 可完全关闭，不影响性能）
        """
        self.enabled = enabled
        self.records = []

    def track(self, stage_name: str, **tensors):
        """
        记录一个阶段的 tensor shape

        参数：
            stage_name: 阶段名称（如 "输入", "Embedding后", "Logits"）
            tensors: 要追踪的 tensor（传入关键字参数）
                tracker.track("输入", input_ids=tensor, attention_mask=tensor)
        """
        if not self.enabled:
            return
        shapes = {k: list(v.shape) for k, v in tensors.items()}
        self.records.append({"stage": stage_name, "shapes": shapes})

    def print_summary(self):
        """打印所有记录的 shape 总结"""
        if not self.enabled or not self.records:
            return

        print("\n" + "=" * 60)
        print("--- Shape Track Summary ---")
        print("=" * 60)
        for rec in self.records:
            print(f"\n  [{rec['stage']}]")
            for name, shape in rec["shapes"].items():
                print(f"    {name:20s} {str(shape):20s}")
        print("=" * 60 + "\n")

    def reset(self):
        """清空所有记录"""
        self.records = []
