"""
Query Router — BERT 分类版

用 exp005 训练的 BERT 分类器判断问题类型：
- gis_term:      GIS 概念/术语 → LLM (LoRA) 直接回答
- gis_operation: GIS 操作/如何做 → RAG 检索知识库
- general:       闲聊/其他 → LLM 原生回答

如果 BERT checkpoint 不存在，自动回退到关键词规则。
"""

import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class Router:
    """查询路由器（BERT 分类 + 关键词回退）"""

    # 关键词规则（BERT 不可用时的回退方案）
    RAG_KEYWORDS = [
        "步骤", "如何", "怎么", "方法", "操作", "导入", "导出",
        "设置", "安装", "配置", "运行", "使用", "创建", "制作",
        "加载", "添加", "删除", "修改", "处理", "新建", "打开",
        "编辑", "裁剪", "合并", "生成", "导出为",
    ]

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            PROJECT_ROOT, "outputs/checkpoints/best_model"
        )
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bert = False
        self.id2label = {0: "gis_term", 1: "gis_operation", 2: "general"}

        self._try_load_bert()

    def _try_load_bert(self):
        """尝试加载 BERT 分类器"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                print(f"[Router] BERT checkpoint 未找到 ({self.checkpoint_dir})，使用关键词规则")
                return

            label_mapping = os.path.join(self.checkpoint_dir, "label_mapping.json")
            if not os.path.exists(label_mapping):
                print(f"[Router] label_mapping 缺失，使用关键词规则")
                return

            import json
            with open(label_mapping, "r", encoding="utf-8") as f:
                label2id = json.load(f)
                self.id2label = {v: k for k, v in label2id.items()}

            print(f"[Router] 加载 BERT 分类器 from {self.checkpoint_dir}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint_dir
            )
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)
            self.use_bert = True
            print(f"[Router] BERT 分类器就绪 ({self.device})")
        except Exception as e:
            print(f"[Router] BERT 加载失败: {e}，回退到关键词规则")

    def route(self, query):
        """
        判断查询应该走 RAG 还是 LLM

        返回：{"type": "rag", "query": query}  或  {"type": "llm", "query": query}
        """
        if self.use_bert:
            return self._route_bert(query)
        return self._route_keywords(query)

    def _route_bert(self, query):
        """BERT 分类"""
        inputs = self.tokenizer(
            query,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_id = outputs.logits.argmax(dim=-1).item()

        label = self.id2label.get(pred_id, "general")

        if label == "gis_operation":
            return {"type": "rag", "query": query, "subtype": label}
        else:
            return {"type": "llm", "query": query, "subtype": label}

    def _route_keywords(self, query):
        """关键词规则"""
        for kw in self.RAG_KEYWORDS:
            if kw in query:
                return {"type": "rag", "query": query}
        return {"type": "llm", "query": query}


if __name__ == "__main__":
    router = Router()
    queries = [
        "什么是遥感",
        "什么是GIS",
        "NDVI怎么计算",
        "QGIS怎么导入shp文件",
        "如何做缓冲区分析",
        "你好",
        "今天天气怎么样",
        "坐标系统有哪些",
    ]
    for q in queries:
        result = router.route(q)
        subtype = result.get("subtype", "-")
        print(f"  {q:30s} → {result['type']:4s} ({subtype})")
