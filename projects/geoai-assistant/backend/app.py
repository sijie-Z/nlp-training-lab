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
import json
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = object
    uvicorn = None


pipeline = None
force_demo_mode = False


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GeoAI Assistant</title>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }
    main { max-width: 860px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { margin: 0 0 24px; color: #52606d; }
    .panel { background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; }
    .messages { min-height: 320px; display: grid; gap: 12px; align-content: start; }
    .msg { padding: 12px 14px; border-radius: 8px; line-height: 1.6; white-space: pre-wrap; }
    .user { background: #dbeafe; justify-self: end; max-width: 80%; }
    .bot { background: #eef2f7; justify-self: start; max-width: 86%; }
    form { display: flex; gap: 10px; margin-top: 14px; }
    input { flex: 1; padding: 12px; border: 1px solid #bcccdc; border-radius: 6px; font-size: 16px; }
    button { padding: 0 18px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-size: 16px; cursor: pointer; }
    button:disabled { background: #9fb3c8; cursor: wait; }
    .meta { margin-top: 6px; font-size: 13px; color: #627d98; }
  </style>
</head>
<body>
  <main>
    <h1>GeoAI Assistant</h1>
    <p>本地 demo 模式：可测试 GIS 概念问答、RAG 操作问答和 fallback 链路。</p>
    <section class="panel">
      <div id="messages" class="messages">
        <div class="msg bot">你好，可以试试：什么是遥感 / QGIS怎么导入shp文件 / NDVI怎么计算</div>
      </div>
      <form id="form">
        <input id="query" autocomplete="off" placeholder="输入一个 GIS 问题" />
        <button id="send" type="submit">发送</button>
      </form>
    </section>
  </main>
  <script>
    const form = document.querySelector("#form");
    const query = document.querySelector("#query");
    const send = document.querySelector("#send");
    const messages = document.querySelector("#messages");

    function addMessage(text, cls, meta) {
      const wrap = document.createElement("div");
      const msg = document.createElement("div");
      msg.className = "msg " + cls;
      msg.textContent = text;
      wrap.appendChild(msg);
      if (meta) {
        const m = document.createElement("div");
        m.className = "meta";
        m.textContent = meta;
        wrap.appendChild(m);
      }
      messages.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = query.value.trim();
      if (!text) return;
      query.value = "";
      send.disabled = true;
      addMessage(text, "user");
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: text})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "请求失败");
        const refs = data.references && data.references.length
          ? " refs=" + data.references.map(r => r.title).join(", ")
          : "";
        addMessage(data.answer, "bot", `source=${data.source} time=${data.time_ms}ms${refs}`);
      } catch (err) {
        addMessage("请求失败：" + err.message, "bot");
      } finally {
        send.disabled = false;
        query.focus();
      }
    });
  </script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时释放"""
    global pipeline
    force_cpu = getattr(app.state, "force_cpu", False)
    demo_mode = getattr(app.state, "demo_mode", None)
    print(f"[Server] 加载 ChatPipeline ({'CPU' if force_cpu else 'GPU/auto'} 模式)...")
    t0 = time.time()
    from chat_pipeline import ChatPipeline
    pipeline = ChatPipeline(force_cpu=force_cpu, demo_mode=demo_mode)
    print(f"[Server] 加载完成 ({time.time() - t0:.1f}s)")
    yield
    print("[Server] 关闭")


if FastAPI is not None:
    app = FastAPI(title="GeoAI Assistant", version="2.0.0", lifespan=lifespan)
else:
    app = None


class ChatRequest(BaseModel):
    if FastAPI is not None:
        query: str


class ChatResponse(BaseModel):
    if FastAPI is not None:
        answer: str
        source: str
        time_ms: int
        references: list = None


def handle_chat(query):
    global pipeline
    if pipeline is None:
        raise RuntimeError("Pipeline still loading, try again")
    return pipeline.chat(query)


def health_payload():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "device": _device_name(),
    }


def _device_name():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu-demo"


if app is not None:
    @app.get("/")
    def index():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(INDEX_HTML)

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            result = handle_chat(req.query)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
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
        return health_payload()


class FallbackHandler(BaseHTTPRequestHandler):
    """无 FastAPI 依赖时的最小 HTTP 服务。"""

    def do_GET(self):
        if self.path == "/":
            self._send_html(200, INDEX_HTML)
            return
        if self.path == "/health":
            self._send_json(200, health_payload())
            return
        self._send_json(404, {"detail": "Not Found"})

    def do_POST(self):
        if self.path != "/chat":
            self._send_json(404, {"detail": "Not Found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            query = payload.get("query") or payload.get("question")
            if not query:
                self._send_json(400, {"detail": "query is required"})
                return
            self._send_json(200, handle_chat(query))
        except Exception as e:
            self._send_json(500, {"detail": str(e)})

    def log_message(self, format, *args):
        return

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status, html):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_fallback_server(port, force_cpu=False, demo_mode=None):
    global pipeline
    from chat_pipeline import ChatPipeline

    print("[Server] FastAPI 未安装，启动标准库 HTTP fallback")
    pipeline = ChatPipeline(force_cpu=force_cpu, demo_mode=True if demo_mode is None else demo_mode)
    server = ThreadingHTTPServer(("127.0.0.1", port), FallbackHandler)
    print(f"[Server] http://127.0.0.1:{port}  GET /health  POST /chat")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoAI Assistant Server")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 模式（核显机器）")
    parser.add_argument("--demo", action="store_true", help="不加载大模型，使用本地 demo/fallback 模式")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    if app is None:
        run_fallback_server(args.port, force_cpu=args.cpu, demo_mode=True)
    else:
        app.state.force_cpu = args.cpu
        app.state.demo_mode = True if args.demo else None
        uvicorn.run(app, host="0.0.0.0", port=args.port)
