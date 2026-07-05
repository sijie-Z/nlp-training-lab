"""
Qwen2.5-0.5B 基座模型检查脚本

作用：
1. 下载并加载模型
2. 打印参数量、显存占用
3. 测试推理速度和回答
4. 保存训练前的 baseline 回答
"""

import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 强制所有缓存到 D 盘
os.environ["HF_HOME"] = "D:/packge/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/packge/huggingface/hub"

print("=" * 60)
print("Qwen2.5-0.5B Baseline Check")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# ====== 1. 加载模型 ======
print("\n[1/5] Loading model...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,      # 半精度，省显存
    device_map="auto",               # 自动分配到 GPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

load_time = time.time() - start
print(f"  Load time: {load_time:.1f}s")

# ====== 2. 参数量 ======
print("\n[2/5] Model parameters:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
print(f"  Parameter dtype:      {model.dtype}")

# ====== 3. 显存占用 ======
print("\n[3/5] GPU Memory:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")

# ====== 4. 推理测试 ======
print("\n[4/5] Inference test:")

test_questions = [
    "什么是遥感？",
    "GPS和北斗有什么区别？",
    "苏州科技大学在哪？",
    "你是谁？",
    "今天天气怎么样？",
]

for q in test_questions:
    inputs = tokenizer(q, return_tensors="pt").to(device)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,          # 确定性生成
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_per_sec = (outputs.shape[1] - inputs.input_ids.shape[1]) / elapsed

    print(f"\n  Q: {q}")
    print(f"  A: {answer}")
    print(f"  ({elapsed:.2f}s, {tokens_per_sec:.1f} tok/s)")

# ====== 5. 保存 baseline ======
print("\n[5/5] Saving baseline answers...")

os.makedirs("experiments/exp004_lora_prep", exist_ok=True)

baseline = {"model": "Qwen/Qwen2.5-0.5B", "answers": []}
for q in test_questions:
    inputs = tokenizer(q, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    baseline["answers"].append({"question": q, "answer_before_lora": answer})

with open("experiments/exp004_lora_prep/baseline_before_lora.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, ensure_ascii=False, indent=2)

print("  Saved to experiments/exp004_lora_prep/baseline_before_lora.json")
print("\n" + "=" * 60)
print("Baseline check complete!")
print("=" * 60)
