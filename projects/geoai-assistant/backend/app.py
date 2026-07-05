"""
GeoAI Assistant — FastAPI 服务 (v2.0)

启动（GPU）：
    cd projects/geoai-assistant
    python backend/app.py

启动（CPU / 核显）：
    python backend/app.py --cpu

接口：
    POST /chat  {"query": "什么是遥感"}
    → {"answer": "...", "source": "lora", "time_ms": 1234}

    GET  /health
    → {"status": "ok", "pipeline_ready": true}
"""

import sys
import os
import time
import argparse
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时释放"""
    global pipeline
    force_cpu = app.state.force_cpu
    print(f"[Server] 加载 ChatPipeline ({'CPU' if force_cpu else 'GPU/auto'} 模式)...")
    t0 = time.time()
    from chat_pipeline import ChatPipeline
    pipeline = ChatPipeline(force_cpu=force_cpu)
    print(f"[Server] 加载完成 ({time.time() - t0:.1f}s)")
    yield
    print("[Server] 关闭")


app = FastAPI(title="GeoAI Assistant", version="2.0.0")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    source: str
    time_ms: int
    references: list = None


@app.post("/chat")
def chat(req: ChatRequest):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again")
    t0 = time.time()
    try:
        result = pipeline.chat(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(
        answer=result["answer"],
        source=result["source"],
        time_ms=result["time_ms"],
        references=result.get("references"),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoAI Assistant Server")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 模式（核显机器）")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    app.state.force_cpu = args.cpu
    uvicorn.run(app, host="0.0.0.0", port=args.port)
