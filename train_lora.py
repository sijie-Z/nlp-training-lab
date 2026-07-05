"""
LoRA 微调入口

用法：
    python train_lora.py --config configs/exp_lora_qwen.yaml

流程：
1. 加载 Qwen2.5-0.5B + 4bit 量化
2. 应用 LoRA（冻结 99.6% 参数）
3. 加载 40 条简单 QA 数据
4. 训练 20 个 epoch
5. 训练后推理，对比 Before/After
6. 保存 adapter（~10MB）
"""

import os
import sys
import json
import argparse
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.lora_model import build_lora_model
from src.trainers.lora_trainer import LoRATrainer, InstructionDataset


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/exp_lora_qwen.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # 创建输出目录
    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.log_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoint_dir, exist_ok=True)

    # 日志
    logger = setup_logger("lora", log_file=f"{cfg.system.log_dir}/lora_train.log")
    logger.info("=" * 50)
    logger.info("LoRA Fine-tuning — Qwen2.5-0.5B")
    logger.info("=" * 50)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 随机种子
    set_seed(cfg.experiment.seed)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.system.device == "cpu":
        device = "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_vram:.1f}GB")

    # 构建 LoRA 模型
    logger.info(f"Building LoRA model: {cfg.model.model_name}")
    logger.info(f"LoRA: r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}, "
                f"targets={cfg.lora.target_modules}")
    model, tokenizer = build_lora_model(
        model_name=cfg.model.model_name,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        use_4bit=cfg.model.get("use_4bit", True),
    )

    # ====== Baseline（训练前回答）======
    test_questions = [
        "苏州科技大学在哪",
        "你是谁",
        "一加一等于几",
        "苏州靠海吗",
    ]
    logger.info("\nBefore training — baseline answers:")
    baselines = {}
    for q in test_questions:
        # 用原始模型生成（还没有 LoRA）
        prompt = f"### Instruction:\n{q}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0], skip_special_tokens=True)
        if "### Response:\n" in answer:
            answer = answer.split("### Response:\n")[-1].strip()
        baselines[q] = answer
        logger.info(f"  Q: {q}")
        logger.info(f"  A: {answer[:60]}...")

    # 保存 baseline
    os.makedirs("experiments/exp004_lora_prep", exist_ok=True)
    with open("experiments/exp004_lora_prep/baseline_before_lora.json",
              "w", encoding="utf-8") as f:
        json.dump({"model": "Qwen2.5-0.5B+LoRA", "answers": baselines},
                  f, ensure_ascii=False, indent=2)

    # 加载数据
    logger.info(f"\nLoading data...")
    train_dataset = InstructionDataset(
        cfg.data.train_path, tokenizer, cfg.data.max_length
    )
    val_dataset = InstructionDataset(
        cfg.data.val_path, tokenizer, cfg.data.max_length
    )
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")

    # 训练
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        logger=logger,
    )
    results = trainer.train()

    # ====== After training — 对比 ======
    logger.info("\n" + "=" * 50)
    logger.info("After training — comparison:")
    logger.info("=" * 50)

    for q in test_questions:
        new_answer = trainer.generate(q)
        old_answer = baselines[q]

        changed = "✅ CHANGED" if new_answer != old_answer else "❌ SAME"
        logger.info(f"\n  Q: {q}")
        logger.info(f"  Before: {old_answer[:60]}")
        logger.info(f"  After:  {new_answer[:60]}")
        logger.info(f"  Status: {changed}")

    # 保存训练后回答
    after = {}
    for q in test_questions:
        after[q] = trainer.generate(q)
    with open("experiments/exp004_lora_prep/answers_after_lora.json",
              "w", encoding="utf-8") as f:
        json.dump({"model": "Qwen2.5-0.5B+LoRA (trained)", "answers": after},
                  f, ensure_ascii=False, indent=2)

    # 显存统计
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        logger.info(f"\nMax GPU memory during training: {max_mem:.2f} GB")

    logger.info(f"\nExperiment 004a complete!")
    logger.info(f"Adapter: outputs/checkpoints/lora_adapter/")


if __name__ == "__main__":
    main()
