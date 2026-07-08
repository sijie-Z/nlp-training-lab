"""
构造 DPO 偏好数据集

策略：从知识库文档生成 (prompt, chosen, rejected) 三元组

- prompt: 一个 GIS 问题
- chosen: 知识库中的正确答案/专业解释
- rejected: 同一 prompt 的劣质回答（混淆、错误、太笼统）

每对数据来自同一篇文档，分为 chosen 和 rejected 两个版本。
"""
import json
import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)


def load_kb():
    kb_path = PROJECT_ROOT / "projects/geoai-assistant/knowledge_base/demo_docs.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_dpo_pairs(kb_docs):
    """
    以知识库为基础，为 DPO 生成偏好数据。

    思路：知识库文档是"正确答案"（chosen），
    然后人工构造"劣质回答"（rejected）
    """
    pairs = []

    # 策略1: 知识库原文作为 chosen，截断/简化版作为 rejected
    for doc in kb_docs:
        title = doc.get("title", "")
        content = doc.get("content", "")

        if len(content) < 50:
            continue

        prompt = f"请解释：{title}"
        chosen = content

        # rejected: 取前1/3的内容，制造"不完整"的回答
        third = len(content) // 3
        if third > 30:
            rejected = content[:third] + "（以上为不完整回答，缺少关键细节）"
        else:
            rejected = content[:30] + "...\n（回答不完整）"

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "category": "truncation",
            "source_doc_id": doc.get("id", ""),
        })

    # 策略2: 泛泛而谈 vs 专业知识（针对前20篇概念类文档）
    generic_answers = {
        "gis": "GIS就是一个做地图的软件系统，可以用来画地图和做空间分析。大概就是这样，具体我也不太清楚。",
        "遥感": "遥感就是用卫星拍照，就是从天上看地面。拍完照片可以用来做各种分析。",
        "坐标": "坐标就是经纬度，表示一个地方在地球上的位置。用经度和纬度两个数字就可以了。",
        "投影": "投影就是把球面变成平面，因为地球是圆的，地图是平的。",
        "数据": "数据就是存储在电脑里的一些数字和文字，可以用来做分析和展示。",
        "空间": "空间就是三维的物理范围，有长度、宽度和高度。",
        "矢量": "矢量数据就是用点线面来表示地理要素，每个点都有坐标。",
        "栅格": "栅格数据就是像照片一样，由一个个像素组成的。",
        "分析": "分析就是对数据做各种计算和处理，得出一些结论。",
        "模型": "模型就是对现实世界的简化表示，用来模拟和预测。",
    }

    for doc in kb_docs[:25]:
        title = doc.get("title", "")
        content = doc.get("content", "")

        if len(content) < 80:
            continue

        prompt = f"请解释：{title}"
        chosen = content

        # 匹配一个泛泛的回答
        matched = None
        for keyword, generic in generic_answers.items():
            if keyword in title or keyword in content[:50]:
                matched = generic
                break

        if matched:
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": matched,
                "category": "generic_vs_professional",
                "source_doc_id": doc.get("id", ""),
            })

    # 策略3: 拒绝"一问三不知"
    i_dont_know = "抱歉，我不太了解这个。建议您查阅相关资料或咨询专业人士。"

    for doc in kb_docs[:15]:
        title = doc.get("title", "")
        content = doc.get("content", "")
        if len(content) < 100:
            continue

        pairs.append({
            "prompt": f"请解释：{title}",
            "chosen": content,
            "rejected": i_dont_know,
            "category": "refuse_vs_answer",
            "source_doc_id": doc.get("id", ""),
        })

    print(f"[DPO Data] 生成了 {len(pairs)} 对偏好数据")
    cats = {}
    for p in pairs:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    for k, v in cats.items():
        print(f"  {k}: {v}")

    return pairs


def save_pairs(pairs, output_path=None):
    output_path = output_path or str(PROJECT_ROOT / "data/dpo/dpo_pairs.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"[DPO Data] 已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    kb = load_kb()
    pairs = generate_dpo_pairs(kb)
    save_pairs(pairs)

    # 打印几个样本
    print("\n=== 样本 ===")
    for i, p in enumerate(pairs[:3]):
        print(f"\n[{p['category']}] {p['prompt']}")
        print(f"  Chosen ({len(p['chosen'])} chars): {p['chosen'][:120]}...")
        print(f"  Rejected ({len(p['rejected'])} chars): {p['rejected'][:120]}...")
