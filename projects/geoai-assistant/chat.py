"""
GeoAI Assistant 命令行对话入口。

默认使用 demo 模式，不加载大模型，适合无显卡或缺少深度学习依赖的电脑演示完整链路。

用法：
    python projects/geoai-assistant/chat.py
    python projects/geoai-assistant/chat.py --query "什么是遥感"
    python projects/geoai-assistant/chat.py --real-model --cpu --query "你是谁"
"""

import argparse
import os
import sys


BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from chat_pipeline import ChatPipeline


def print_result(result):
    print("\nAnswer:")
    print(result["answer"])
    print(f"\nsource={result['source']}  time_ms={result['time_ms']}")
    if result.get("references"):
        print("references:")
        for ref in result["references"]:
            print(f"- {ref['title']} (score={ref['score']})")


def main():
    parser = argparse.ArgumentParser(description="GeoAI Assistant CLI")
    parser.add_argument("--query", "-q", help="单轮问题；不传则进入交互模式")
    parser.add_argument("--cpu", action="store_true", help="真实模型模式下强制 CPU")
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="尝试加载 Qwen/LoRA；默认 demo 模式用于无显卡演示",
    )
    args = parser.parse_args()

    pipeline = ChatPipeline(force_cpu=args.cpu, demo_mode=not args.real_model)

    if args.query:
        print_result(pipeline.chat(args.query))
        return

    print("\nGeoAI Assistant CLI")
    print("输入问题开始对话；输入 exit/quit 退出。")
    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        print_result(pipeline.chat(query))


if __name__ == "__main__":
    main()
