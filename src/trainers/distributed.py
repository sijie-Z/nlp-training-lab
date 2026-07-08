"""
分布式训练核心技巧（单卡可验证版）

在只有一张 GPU 的情况下，把分布式训练的核心概念全做一遍：
  1. 混合精度训练 (AMP) — FP16/BF16 加速
  2. 梯度累积 — 模拟大 batch size
  3. 梯度检查点 (Gradient Checkpointing) — 用计算换显存
  4. DeepSpeed ZeRO 配置 — 单卡也能用 ZeRO-2

面试常问：
  - "训练时 OOM 了怎么办？"
  - "混合精度训练的原理？为什么需要 loss scaling？"
  - "梯度累积和增大 batch size 等价吗？"（不等价，BatchNorm 层不一样）
  - "ZeRO-1/2/3 的区别？"
"""
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 把 scripts/ 也加入 path（pretrain_tiny 里的数据集类在 scripts/）
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.models.tiny_gpt import TinyGPT, create_model
from src.data_pipeline.collector import load_knowledge_base, extract_raw_corpus, clean_corpus
from tokenizers import Tokenizer


# ============================================================
# 1. 混合精度训练 (Automatic Mixed Precision)
# ============================================================
class AMPTrainer:
    """
    FP16 混合精度训练器

    原理：
      - 前向/反向用 FP16（速度快，显存省一半）
      - 权重更新用 FP32（精度够）
      - loss scaling：FP16 动态范围小，小梯度会 underflow 到 0
        所以 forward 后把 loss 放大（scale），backward 后再缩小

    BF16 vs FP16:
      - BF16: 和 FP32 范围一样，不需要 loss scaling，但只有 Ampere+ (RTX 30系+) 支持
      - FP16: 范围小，需要 loss scaling，兼容性好
    """

    def __init__(self, model, use_bf16=False):
        self.model = model
        self.device = next(model.parameters()).device
        self.use_bf16 = use_bf16

        # GradScaler 只在 FP16 时需要
        self.scaler = None if use_bf16 else torch.amp.GradScaler('cuda')

        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.amp_enabled = True

    def training_step(self, batch, optimizer):
        """带混合精度的训练步骤"""
        self.model.train()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # 混合精度上下文
        with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.amp_enabled):
            logits, loss = self.model(x, labels=y)

        # 反向传播（FP16 需要 loss scaling）
        optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        return loss.item()


# ============================================================
# 2. 梯度累积 — 模拟大 batch size
# ============================================================
class GradientAccumulationTrainer:
    """
    梯度累积：每 accumulation_steps 步才更新一次权重

    等效于 batch_size × accumulation_steps 的大 batch，但：
      - BatchNorm 层：每步 stats 不同，不等价
      - Dropout：每步 mask 不同，也不完全等价
      - 对于纯 Transformer（无 BN），基本等价

    面试要点：
      "为什么不用大 batch 而用梯度累积？"
      → 因为显存放不下大 batch，但需要大 batch 来稳定训练
    """

    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.device = next(model.parameters()).device
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def training_step(self, batch, optimizer):
        self.model.train()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.amp.autocast('cuda', enabled=True):
            logits, loss = self.model(x, labels=y)
            # 归一化 loss：每个 micro-batch 贡献相同
            loss = loss / self.accumulation_steps

        loss.backward()
        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            updated = True
        else:
            updated = False

        return loss.item() * self.accumulation_steps, updated


# ============================================================
# 3. 梯度检查点 (Gradient Checkpointing / Activation Checkpointing)
# ============================================================
def enable_checkpointing(model):
    """
    用 torch.utils.checkpoint 重新包装每个 DecoderBlock 的 forward

    原理：
      正常训练：forward 时保存所有中间激活值，backward 时直接用
      检查点：forward 时不保存中间激活，backward 时重新计算
      → 省显存，多计算（~20% 慢）

    ZeRO 层级对比（面试高频）：
      ZeRO-1: 优化器状态分片（省 4x 优化器显存）
      ZeRO-2: 优化器 + 梯度分片
      ZeRO-3: 优化器 + 梯度 + 参数分片（省最多，通信最多）
    """
    from torch.utils.checkpoint import checkpoint

    enabled = False
    for i, layer in enumerate(model.layers):
        original_forward = layer.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(x):
                # 需要 requires_grad=True 才能 checkpoint
                return checkpoint(orig_fn, x, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)
        enabled = True

    return enabled


def disable_checkpointing(model, original_forwards=None):
    """恢复原始的 forward（如果需要）"""
    pass  # 简化处理：不恢复了


# ============================================================
# 4. 显存剖析工具
# ============================================================
class MemoryProfiler:
    """记录训练过程中的显存使用"""

    def __init__(self):
        self.records = []
        self._baseline = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    def snapshot(self, label=""):
        if not torch.cuda.is_available():
            return {"label": label, "allocated_mb": 0, "reserved_mb": 0}

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        record = {
            "label": label,
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
        self.records.append(record)
        return record

    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def report(self):
        print("\n" + "=" * 60)
        print("[Memory] 显存使用报告")
        print("=" * 60)
        for r in self.records:
            print(f"  {r['label']:<30s} alloc={r['allocated_mb']:6.1f} MB  "
                  f"reserved={r['reserved_mb']:6.1f} MB  peak={r['peak_mb']:6.1f} MB")
        return self.records


# ============================================================
# 5. 模拟 DeepSpeed ZeRO-2 配置（概念演示）
# ============================================================
DEEPSPEED_CONFIG_TEMPLATE = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: 优化器状态 + 梯度分片
        "offload_optimizer": {
            "device": "none",  # 或 "cpu" 把优化器状态 offload 到 CPU
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,        # 动态 loss scaling
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-4,
            "warmup_num_steps": 100,
        }
    },
    "gradient_clipping": 1.0,
}

DEEPSPEED_ZERO_COMPARISON = {
    "ZeRO-1": {
        "shards": "优化器状态 (optimizer states)",
        "显存节省": "~4x (Adam 有 m, v 两个状态)",
        "通信量": "和普通 DDP 一样",
        "适用场景": "模型能放进单卡但优化器放不下",
    },
    "ZeRO-2": {
        "shards": "优化器状态 + 梯度",
        "显存节省": "~8x",
        "通信量": "和普通 DDP 一样",
        "适用场景": "梯度和优化器都放不下",
    },
    "ZeRO-3": {
        "shards": "优化器状态 + 梯度 + 模型参数",
        "显存节省": "和 GPU 数量线性相关 (N 卡 → 省 N 倍)",
        "通信量": "增加 50%（需要 all-to-all 通信）",
        "适用场景": "模型本身太大，单卡放不下参数",
    },
}


# ============================================================
# 6. 完整训练对比实验
# ============================================================
def load_data(tokenizer_path, corpus_path, block_size=256, batch_size=8):
    """加载预训练数据"""
    from scripts.pretrain_tiny import PretrainDataset, DataCollator

    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = PretrainDataset(corpus_path, tokenizer_path,
                              block_size=block_size, stride=128)
    pad_id = tokenizer.token_to_id("[PAD]") or 0
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=DataCollator(pad_token_id=pad_id),
    )
    return dataloader, tokenizer


def run_benchmark():
    """
    对比四种训练模式的显存和速度：
      1. FP32 (baseline)
      2. FP16 AMP
      3. BF16 AMP
      4. FP16 + 梯度检查点
      5. FP16 + 梯度累积 (×4)
      6. FP16 + 梯度检查点 + 梯度累积 (全开)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] 设备: {device}")
    if device.type == "cuda":
        print(f"[Benchmark] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Benchmark] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 用 medium 模型来压测
    model, config = create_model(preset="small", vocab_size=8000, block_size=256)
    model = model.to(device)
    print(f"[Benchmark] 模型参数量: {model.get_num_params():,}")

    # 加载数据
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")
    corpus_path = str(PROJECT_ROOT / "data/raw/domain_corpus_cleaned.txt")
    dataloader, _ = load_data(tokenizer_path, corpus_path, batch_size=8)

    # 取一个 batch 做 benchmark
    batch = next(iter(dataloader))
    print(f"[Benchmark] Batch: x={batch[0].shape}, y={batch[1].shape}")

    results = []

    # --- 模式 1: FP32 (baseline) ---
    print("\n" + "=" * 60)
    print("[Benchmark] 模式 1: FP32 (baseline)")
    print("=" * 60)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_fp32, _ = create_model(preset="small", vocab_size=8000, block_size=256)
    model_fp32 = model_fp32.to(device)
    opt = AdamW(model_fp32.parameters(), lr=5e-4)

    t0 = time.time()
    model_fp32.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits, loss = model_fp32(x, labels=y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    fp32_time = time.time() - t0
    fp32_mem = torch.cuda.max_memory_allocated() / 1024**2

    results.append({
        "mode": "FP32 (baseline)",
        "time_ms": round(fp32_time * 1000, 1),
        "peak_memory_mb": round(fp32_mem, 1),
    })
    print(f"  耗时: {fp32_time*1000:.1f} ms, 峰值显存: {fp32_mem:.1f} MB")

    del model_fp32
    torch.cuda.empty_cache()

    # --- 模式 2: FP16 AMP ---
    print("\n[Benchmark] 模式 2: FP16 AMP")
    torch.cuda.reset_peak_memory_stats()

    model_fp16, _ = create_model(preset="small", vocab_size=8000, block_size=256)
    model_fp16 = model_fp16.to(device)
    opt = AdamW(model_fp16.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda')

    t0 = time.time()
    model_fp16.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model_fp16(x, labels=y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    fp16_time = time.time() - t0
    fp16_mem = torch.cuda.max_memory_allocated() / 1024**2

    speedup = (fp32_time / fp16_time - 1) * 100
    mem_saved = (fp32_mem / fp16_mem - 1) * 100

    results.append({
        "mode": "FP16 AMP",
        "time_ms": round(fp16_time * 1000, 1),
        "peak_memory_mb": round(fp16_mem, 1),
        "speedup": f"{speedup:+.0f}%",
        "mem_saved": f"{mem_saved:+.0f}%",
    })
    print(f"  耗时: {fp16_time*1000:.1f} ms (加速 {speedup:+.0f}%), "
          f"峰值显存: {fp16_mem:.1f} MB (节省 {mem_saved:+.0f}%)")

    del model_fp16
    torch.cuda.empty_cache()

    # --- 模式 3: BF16 AMP ---
    if torch.cuda.is_bf16_supported():
        print("\n[Benchmark] 模式 3: BF16 AMP")
        torch.cuda.reset_peak_memory_stats()

        model_bf16, _ = create_model(preset="small", vocab_size=8000, block_size=256)
        model_bf16 = model_bf16.to(device)
        opt = AdamW(model_bf16.parameters(), lr=5e-4)

        t0 = time.time()
        model_bf16.train()
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model_bf16(x, labels=y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        bf16_time = time.time() - t0
        bf16_mem = torch.cuda.max_memory_allocated() / 1024**2

        speedup = (fp32_time / bf16_time - 1) * 100
        mem_saved = (fp32_mem / bf16_mem - 1) * 100

        results.append({
            "mode": "BF16 AMP",
            "time_ms": round(bf16_time * 1000, 1),
            "peak_memory_mb": round(bf16_mem, 1),
            "speedup": f"{speedup:+.0f}%",
            "mem_saved": f"{mem_saved:+.0f}%",
        })
        print(f"  耗时: {bf16_time*1000:.1f} ms (加速 {speedup:+.0f}%), "
              f"峰值显存: {bf16_mem:.1f} MB (节省 {mem_saved:+.0f}%)")

        del model_bf16
        torch.cuda.empty_cache()
    else:
        print("\n[Benchmark] 模式 3: BF16 AMP — 不支持，跳过")

    # --- 模式 4: FP16 + 梯度检查点 ---
    print("\n[Benchmark] 模式 4: FP16 + 梯度检查点")
    torch.cuda.reset_peak_memory_stats()

    model_ckpt, _ = create_model(preset="small", vocab_size=8000, block_size=256)
    model_ckpt = model_ckpt.to(device)
    enable_checkpointing(model_ckpt)
    opt = AdamW(model_ckpt.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda')

    try:
        t0 = time.time()
        model_ckpt.train()
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, loss = model_ckpt(x, labels=y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        ckpt_time = time.time() - t0
        ckpt_mem = torch.cuda.max_memory_allocated() / 1024**2

        speedup = (fp32_time / ckpt_time - 1) * 100
        mem_saved = (fp32_mem / ckpt_mem - 1) * 100

        results.append({
            "mode": "FP16 + Grad CKPT",
            "time_ms": round(ckpt_time * 1000, 1),
            "peak_memory_mb": round(ckpt_mem, 1),
            "speedup": f"{speedup:+.0f}%",
            "mem_saved": f"{mem_saved:+.0f}%",
        })
        print(f"  耗时: {ckpt_time*1000:.1f} ms (vs FP32: {speedup:+.0f}%), "
              f"峰值显存: {ckpt_mem:.1f} MB (vs FP32: {mem_saved:+.0f}%)")
    except Exception as e:
        print(f"  梯度检查点模式出错: {e}")
        results.append({
            "mode": "FP16 + Grad CKPT",
            "time_ms": 0,
            "peak_memory_mb": 0,
            "error": str(e)[:100],
        })

    del model_ckpt
    torch.cuda.empty_cache()

    # --- 模式 5: FP16 + 梯度累积 (×4) ---
    print("\n[Benchmark] 模式 5: FP16 + 梯度累积 (×4)")
    torch.cuda.reset_peak_memory_stats()

    model_ga, _ = create_model(preset="small", vocab_size=8000, block_size=256)
    model_ga = model_ga.to(device)
    opt = AdamW(model_ga.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler('cuda')
    accum_steps = 4

    t0 = time.time()
    model_ga.train()
    total_loss = 0
    for _ in range(accum_steps):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, loss = model_ga(x, labels=y)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    ga_time = time.time() - t0
    ga_mem = torch.cuda.max_memory_allocated() / 1024**2

    results.append({
        "mode": f"FP16 + GradAccum ×{accum_steps}",
        "time_ms": round(ga_time * 1000, 1),
        "peak_memory_mb": round(ga_mem, 1),
        "effective_batch": batch[0].shape[0] * accum_steps,
    })
    print(f"  耗时: {ga_time*1000:.1f} ms, 峰值显存: {ga_mem:.1f} MB, "
          f"等效 batch_size: {batch[0].shape[0] * accum_steps}")

    del model_ga
    torch.cuda.empty_cache()

    # --- 汇总 ---
    print("\n" + "=" * 70)
    print("[Benchmark] 汇总对比")
    print("=" * 70)
    print(f"{'模式':<30s} {'耗时(ms)':>8s} {'峰值显存(MB)':>12s} {'加速':>8s} {'省显存':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['mode']:<30s} {r['time_ms']:>8.1f} {r['peak_memory_mb']:>12.1f} "
              f"{r.get('speedup', '-'):>8s} {r.get('mem_saved', '-'):>8s}")

    # 保存结果
    output_path = str(PROJECT_ROOT / "outputs/benchmarks/distributed_benchmark.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model_params": model.get_num_params(),
            "batch_shape": list(batch[0].shape),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Benchmark] 结果已保存: {output_path}")

    return results


def run_deepspeed_config_demo():
    """生成 DeepSpeed 配置并解释"""
    output_path = str(PROJECT_ROOT / "configs/deepspeed_zero2.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEEPSPEED_CONFIG_TEMPLATE, f, ensure_ascii=False, indent=2)
    print(f"[DeepSpeed] 示例配置已保存: {output_path}")
    print("[DeepSpeed] 如果要真正用 DeepSpeed 启动训练（需要安装 deepspeed）：")
    print(f"  deepspeed scripts/pretrain_tiny.py --deepspeed configs/deepspeed_zero2.json")

    print("\n[DeepSpeed] ZeRO 层级对比：")
    for level, info in DEEPSPEED_ZERO_COMPARISON.items():
        print(f"  {level}: 分片 {info['shards']}, 省 {info['显存节省']}, "
              f"通信量 {info['通信量']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练核心技巧演示")
    parser.add_argument("--benchmark", action="store_true", help="运行显存基准测试")
    parser.add_argument("--deepspeed_config", action="store_true", help="生成 DeepSpeed 配置")
    args = parser.parse_args()

    if args.benchmark or not args.deepspeed_config:
        run_benchmark()

    if args.deepspeed_config:
        run_deepspeed_config_demo()
