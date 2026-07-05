"""
GeoAI Assistant — FastAPI 服务

启动：
    cd projects/geoai-assistant
    python backend/app.py

接口：
    POST /chat
    {"query": "什么是遥感"}
    → {"answer": "...", "source": "lora", "time_ms": 1234}
"""

import sys
import os
import json
import time
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
    print("[Server] 加载 ChatPipeline...")
    t0 = time.time()
    from chat_pipeline import ChatPipeline
    pipeline = ChatPipeline()
    print(f"[Server] 加载完成 ({time.time()-t0:.1f}s)")
    yield
    print("[Server] 关闭")


app = FastAPI(title="GeoAI Assistant", version="0.1.0", lifespan=lifespan)


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
        raise HTTPException(status_code=503, detail="Model loading, try again")
    t0 = time.time()
    try:
        result = pipeline.chat(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = int((time.time() - t0) * 1000)
    return ChatResponse(
        answer=result["answer"],
        source=result["source"],
        time_ms=elapsed,
        references=result.get("references"),
    )


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
