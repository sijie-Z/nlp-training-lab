"""
Query Router

根据简单关键词规则判断问题类型：
- "步骤"/"如何"/"怎么"/"方法"/"操作" → RAG 检索
- 其他 → LLM（LoRA）回答

用法：
    router = Router()
    result = router.route("什么是GIS")
    # {"type": "llm", "query": "什么是GIS"}
"""

class Router:
    """查询路由器（基于关键词）"""

    RAG_KEYWORDS = ["步骤", "如何", "怎么", "方法", "操作", "导入", "导出",
                    "设置", "安装", "配置", "运行", "使用", "创建"]

    def route(self, query):
        """
        判断查询应该走 RAG 还是 LLM

        返回：
            {"type": "rag", "query": query}  或
            {"type": "llm", "query": query}
        """
        for kw in self.RAG_KEYWORDS:
            if kw in query:
                return {"type": "rag", "query": query}

        return {"type": "llm", "query": query}


if __name__ == "__main__":
    import sys
    router = Router()
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "什么是GIS",
        "什么是遥感",
        "QGIS怎么导入shp文件",
        "坐标系统有哪些",
    ]
    for q in queries:
        result = router.route(q)
        print(f"  {q:30s} → {result['type']}")
