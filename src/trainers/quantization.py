"""
模型量化 — 从 FP32 到 INT8/INT4 的完整流程

技术路线：
  1. 动态量化 (PTQ) — torch.quantization（最简单）
  2. 静态量化 (PTQ) — 需要校准数据
  3. 半精度 (FP16/BF16) — 作为对比基线
  4. 保存量化模型 + 大小对比 + 推理速度 benchmark

面试核心：
  - 对称量化 vs 非对称量化
  - 逐层量化 vs 逐通道量化
  - 量化感知训练 (QAT) vs 训练后量化 (PTQ)
  - GPTQ 为什么需要 Hessian 矩阵？
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from src.models.tiny_gpt import TinyGPT, create_model


# ============================================================
# 1. 模型大小计算工具
# ============================================================
def get_model_size_mb(model_or_state_dict):
    """计算模型参数占用的存储空间"""
    if isinstance(model_or_state_dict, dict):
        total = sum(p.numel() * p.element_size() for p in model_or_state_dict.values())
    else:
        total = sum(p.numel() * p.element_size() for p in model_or_state_dict.parameters())
    return total / (1024 * 1024)


def get_model_size_by_dtype(model, dtype_map=None):
    """估算不同精度下的模型大小"""
    total_params = sum(p.numel() for p in model.parameters())
    dtype_sizes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    if dtype_map:
        dtype_sizes.update(dtype_map)

    result = {}
    for name, bytes_per in dtype_sizes.items():
        result[name] = total_params * bytes_per / (1024 * 1024)
    return result, total_params


# ============================================================
# 2. 动态量化（最简单的方式）
# ============================================================
def apply_dynamic_quantization(model):
    """
    PyTorch 动态量化：将 Linear 层权重量化为 INT8

    原理：
      - 权重：训练后静态量化（离线算好 scale 和 zero_point）
      - 激活：运行时动态量化（每次 forward 时算 min/max）

    注意：动态量化对 Transformer 的加速不如静态量化好
          因为每次 forward 都要重新算激活的量化参数
    """
    # torch.quantization.quantize_dynamic 只支持 CPU
    model_cpu = model.cpu()

    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},         # 量化哪些层类型
        dtype=torch.qint8,   # 量化精度
    )
    return quantized


# ============================================================
# 3. 手动 INT8 量化（演示原理，不用 torch 的量化 API）
# ============================================================
class ManualLinearINT8(nn.Module):
    """
    手动实现的 INT8 量化 Linear 层
    用于演示对称量化的原理

    对称量化: x_q = round(x / scale) * scale
               scale = max(|x|) / 127  (INT8 范围是 -128 ~ 127)

    非对称: x_q = round((x - zero_point) / scale)
            zero_point 补偿数据分布偏移
    """

    def __init__(self, weight_fp32, bias_fp32=None, bits=8):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1  # 127 for int8

        # 计算 scale（对称量化）
        w_max = weight_fp32.abs().max().item()
        self.scale = w_max / self.qmax

        # 量化权重
        w_int = torch.round(weight_fp32 / self.scale).clamp(-self.qmax, self.qmax).to(torch.int8)
        self.register_buffer("weight_int", w_int)
        self.register_buffer("scale_tensor", torch.tensor([self.scale]))

        if bias_fp32 is not None:
            self.register_buffer("bias_fp32", bias_fp32.float())
        else:
            self.bias_fp32 = None

    def forward(self, x):
        # 反量化 → FP32 计算 → 输出
        w_fp = self.weight_int.float() * self.scale
        out = x @ w_fp.T
        if self.bias_fp32 is not None:
            out += self.bias_fp32
        return out


def manual_quantize_model(model):
    """将模型中所有 Linear 层替换为手动 INT8 量化版"""
    quantized = model
    named_modules = dict(quantized.named_modules())
    replaced = 0

    for name, module in list(named_modules.items()):
        if isinstance(module, nn.Linear) and not name.endswith('.lm_head'):
            # 获取父子关系
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = dict(quantized.named_modules())[parent_name]
            else:
                parent_name = ''
                child_name = name
                parent = quantized

            quant_linear = ManualLinearINT8(module.weight.data, module.bias.data if module.bias is not None else None)
            setattr(parent, child_name, quant_linear)
            replaced += 1

    print(f"[Quantize] 手动量化了 {replaced} 个 Linear 层")
    return quantized


# ============================================================
# 4. 保存为 FP16/BF16（不需要量化，直接转换精度）
# ============================================================
def convert_to_half(model, dtype=torch.float16):
    """将模型转换为半精度"""
    model_half = model.half() if dtype == torch.float16 else model.to(dtype=torch.bfloat16)
    return model_half


# ============================================================
# 5. 基准测试
# ============================================================
def benchmark_inference(model, tokenizer, test_questions, device="cpu", num_runs=3):
    """测试推理延迟和吞吐量"""
    model.eval()
    model = model.to(device)

    latencies = []
    for question in test_questions:
        enc = tokenizer.encode(question)
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

        t0 = time.time()
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=40)
        latencies.append(time.time() - t0)

    avg_latency = sum(latencies) / len(latencies)
    return {
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "total_time_s": round(sum(latencies), 2),
        "num_questions": len(test_questions),
    }


def run_quantization_benchmark():
    """完整量化对比实验"""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[Quant Benchmark] 设备: {device}")
    print(f"[Quant Benchmark] GPU: {torch.cuda.get_device_name(0) if device_str == 'cuda' else 'CPU'}")

    # 加载模型
    checkpoint_path = str(PROJECT_ROOT / "outputs/checkpoints/pretrain_tiny/checkpoint_epoch020.pt")
    tokenizer_path = str(PROJECT_ROOT / "outputs/tokenizers/bpe_domain/tokenizer.json")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    tokenizer = Tokenizer.from_file(tokenizer_path)

    test_questions = [
        "什么是GIS", "遥感技术是什么", "NDVI怎么计算",
        "QGIS如何导入shp文件", "坐标系转换方法",
    ]

    results = []

    # --- FP32 Baseline ---
    print("\n[Quant] 1. FP32 Baseline")
    model_fp32 = TinyGPT(config)
    model_fp32.load_state_dict(ckpt["model"])
    fp32_size = get_model_size_mb(model_fp32)
    _, total_params = get_model_size_by_dtype(model_fp32)

    bench = benchmark_inference(model_fp32, tokenizer, test_questions, device_str)
    results.append({
        "mode": "FP32 (baseline)",
        "size_mb": round(fp32_size, 1),
        "compression_ratio": "1.0x",
        **bench,
    })
    print(f"  大小: {fp32_size:.1f} MB, 延迟: {bench['avg_latency_ms']:.1f} ms")
    del model_fp32
    torch.cuda.empty_cache()

    # --- FP16 ---
    print("\n[Quant] 2. FP16 半精度")
    model_fp16 = TinyGPT(config)
    model_fp16.load_state_dict(ckpt["model"])
    model_fp16 = model_fp16.to(device)
    model_fp16 = model_fp16.half()
    fp16_size = get_model_size_mb(model_fp16)

    bench = benchmark_inference(model_fp16, tokenizer, test_questions, device_str)
    ratio = fp32_size / fp16_size
    results.append({
        "mode": "FP16 (半精度)",
        "size_mb": round(fp16_size, 1),
        "compression_ratio": f"{ratio:.1f}x",
        **bench,
    })
    print(f"  大小: {fp16_size:.1f} MB, 压缩比: {ratio:.1f}x, 延迟: {bench['avg_latency_ms']:.1f} ms")
    del model_fp16
    torch.cuda.empty_cache()

    # --- BF16 ---
    if device_str == "cuda" and torch.cuda.is_bf16_supported():
        print("\n[Quant] 3. BF16 半精度")
        model_bf16 = TinyGPT(config)
        model_bf16.load_state_dict(ckpt["model"])
        model_bf16 = model_bf16.to(dtype=torch.bfloat16, device=device)
        bf16_size = get_model_size_mb(model_bf16)

        bench = benchmark_inference(model_bf16, tokenizer, test_questions, device_str)
        ratio = fp32_size / bf16_size
        results.append({
            "mode": "BF16 (半精度)",
            "size_mb": round(bf16_size, 1),
            "compression_ratio": f"{ratio:.1f}x",
            **bench,
        })
        print(f"  大小: {bf16_size:.1f} MB, 压缩比: {ratio:.1f}x, 延迟: {bench['avg_latency_ms']:.1f} ms")
        del model_bf16
        torch.cuda.empty_cache()

    # --- 手动 INT8 演示 ---
    print("\n[Quant] 4. 手动 INT8 量化（演示原理）")
    model_int8 = TinyGPT(config)
    model_int8.load_state_dict(ckpt["model"])
    model_int8 = manual_quantize_model(model_int8).to(device)

    # 估算 INT8 大小
    int8_size = total_params * 1 / (1024 * 1024)  # 1 byte per param
    ratio = fp32_size / int8_size

    bench = benchmark_inference(model_int8, tokenizer, test_questions, device_str)
    results.append({
        "mode": "INT8 (手动量化)",
        "size_mb": round(int8_size, 1),
        "compression_ratio": f"{ratio:.1f}x",
        **bench,
    })
    print(f"  大小: {int8_size:.1f} MB, 压缩比: {ratio:.1f}x, 延迟: {bench['avg_latency_ms']:.1f} ms")
    del model_int8
    torch.cuda.empty_cache()

    # --- INT4 估算 ---
    print("\n[Quant] 5. INT4 量化估算（理论值）")
    int4_size = total_params * 0.5 / (1024 * 1024)  # 0.5 byte per param
    ratio = fp32_size / int4_size
    results.append({
        "mode": "INT4 (理论值)",
        "size_mb": round(int4_size, 1),
        "compression_ratio": f"{ratio:.1f}x",
        "avg_latency_ms": "N/A（未实现推理）",
        "note": "真正的INT4需要specialized kernel(GPTQ/AWQ)",
    })
    print(f"  大小: {int4_size:.1f} MB, 压缩比: {ratio:.1f}x")
    print(f"  注: 真正的 INT4 推理需要 GPTQ/AWQ 等专用框架")

    # --- 汇总 ---
    print(f"\n{'='*80}")
    print(f"[Quant] 量化对比汇总 (模型参数: {total_params:,})")
    print(f"{'='*80}")
    print(f"{'模式':<25s} {'大小(MB)':>10s} {'压缩比':>8s} {'延迟(ms)':>10s}")
    print("-" * 60)
    for r in results:
        lat = f"{r['avg_latency_ms']:>10.1f}" if isinstance(r.get('avg_latency_ms'), (int, float)) else f"{'N/A':>10s}"
        print(f"{r['mode']:<25s} {r['size_mb']:>10.1f} {r['compression_ratio']:>8s} {lat}")

    # 保存
    output_path = str(PROJECT_ROOT / "outputs/benchmarks/quantization_benchmark.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": device_str,
            "gpu": torch.cuda.get_device_name(0) if device_str == "cuda" else "CPU",
            "model_params": total_params,
            "fp32_size_mb": round(fp32_size, 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Quant] 结果已保存: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型量化基准测试")
    parser.add_argument("--all", action="store_true", help="运行完整对比")
    args = parser.parse_args()

    if args.all or True:  # 默认全跑
        run_quantization_benchmark()
