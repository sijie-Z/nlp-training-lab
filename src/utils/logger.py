"""
日志工具

作用：同时输出日志到控制台和文件

用法：
    logger = setup_logger("train", "outputs/logs/train.log")
    logger.info("Epoch 1 | Loss: 1.23")
"""

import logging
import os


def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    设置同时输出到控制台和文件的日志

    参数：
        name: 日志名字（通常用 "train"）
        log_file: 日志文件路径
        level: 日志级别（默认 INFO）

    返回：
        logger 实例

    输出格式：
        [2026-06-22 14:30:00] INFO | Epoch 1 | Loss: 1.21
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件 Handler（追加模式）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
