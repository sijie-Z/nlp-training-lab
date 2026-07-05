"""
训练入口脚本

用法：
    python train.py --config configs/exp002_400.yaml
    python train.py --config configs/exp002_400.yaml training.epochs=5
"""

import os
import sys
import argparse
import torch

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.datasets.news_dataset import NewsDataset
from src.datasets.match_dataset import MatchDataset
from src.models.factory import build_model
from src.trainers.trainer import Trainer


def main():
    # ====== 1. 解析命令行参数 ======
    parser = argparse.ArgumentParser(description="NLP Training Lab")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="配置文件路径")
    parser.add_argument("overrides", nargs="*",
                        help="命令行覆盖参数, 如 training.epochs=5")
    args = parser.parse_args()

    # ====== 2. 加载配置 + 命令行覆盖 ======
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        overrides = OmegaConf.from_cli(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    # 创建输出目录
    os.makedirs(cfg.system.output_dir, exist_ok=True)
    os.makedirs(cfg.system.log_dir, exist_ok=True)
    os.makedirs(cfg.system.figure_dir, exist_ok=True)
    os.makedirs(cfg.system.checkpoint_dir, exist_ok=True)

    # ====== 2. 设置日志 ======
    logger = setup_logger(
        "train",
        log_file=f"{cfg.system.log_dir}/train.log",
    )
    logger.info("=" * 50)
    logger.info("NLP Training Lab — BERT News Classification")
    logger.info("=" * 50)
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # ====== 3. 固定随机种子 ======
    set_seed(cfg.experiment.seed)
    logger.info(f"Random seed set to {cfg.experiment.seed}")

    # ====== 4. 自动选择设备 ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.system.device == "cpu":
        device = "cpu"
    cfg.system.device = device
    logger.info(f"Device: {device}")

    # ====== 5. 准备数据（按 task_type 路由）======
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    task_type = cfg.data.get("task_type", "classification")
    logger.info(f"Task type: {task_type}")

    if task_type == "match":
        # ---- 文本匹配: 双句输入 ----
        logger.info(f"Loading match data from {cfg.data.train_path}...")
        train_dataset = MatchDataset(
            data_path=cfg.data.train_path,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )
        val_dataset = MatchDataset(
            data_path=cfg.data.val_path,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )
        label2id = {"不匹配": 0, "匹配": 1}
        logger.info(f"Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

    else:
        # ---- 新闻分类: 单句输入（原有逻辑）----
        logger.info(f"Loading dataset from {cfg.data.csv_path}...")
        full_dataset = NewsDataset(
            csv_path=cfg.data.csv_path,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
            text_col=cfg.data.text_col,
            label_col=cfg.data.label_col,
        )
        label2id = full_dataset.label2id
        logger.info(f"Labels: {label2id}")

        # 划分训练/验证
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * cfg.data.val_size)
        train_size = dataset_size - val_size
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.experiment.seed)
        )
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.system.num_workers,
    )

    # ====== 6. 创建模型 ======
    logger.info(f"Building model: {cfg.model.model_name}")
    model = build_model(
        model_name=cfg.model.model_name,
        num_classes=cfg.data.num_labels,
        device=device,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ====== 7. 训练 ======
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        logger=logger,
        label2id=label2id,
    )

    results = trainer.train()

    # ====== 8. 输出结果 ======
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {results['best_accuracy']:.4f}")
    logger.info(f"{'='*50}")
    logger.info(f"Run inference: python src/inference/predict.py --text \"国足今晚迎战日本队\"")


if __name__ == "__main__":
    main()
