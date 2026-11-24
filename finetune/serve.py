"""
VLM 动作服务（第5周）：在 WSL2（有 GPU）里加载基座 + LoRA，Windows 端 Agent 通过 HTTP 调用

    # WSL2，gui-agent 环境
    python -m finetune.serve --adapter outputs/lora_qwen2vl_2b/best --port 8765

    # Windows，gui-agent-win 环境（WSL2 的端口会自动转发到 Windows 的 localhost）
    python -m agent.cli --act vlm --vlm-url http://localhost:8765 "双击打开回收站"

接口：
  GET  /health → {"ok": true, "name": "lora", "prompt": "v1_schema"}
  POST /act    ← {"image_b64": "...", "instruction": "...", "sub_task": "...", "screen_size": [w, h]}
               → {"ok": bool, "actions": [...0~1000 坐标...], "raw": "...", "sec": 1.23, "error": null}

只用标准库 http.server；同一时刻只处理一个请求（GPU 推理本来也是串行的）。
"""
import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

MAX_BODY = 32 * 1024 * 1024


def decode_image(b64: str):
    from PIL import Image
    if not isinstance(b64, str) or not b64:
        raise ValueError("缺少 image_b64")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError(f"image_b64 不是合法 base64: {e}") from e
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法解码图片: {e}") from e


def make_handler(actor, lock: Optional[threading.Lock] = None):
    lock = lock or threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 安静一点
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self._send(200, {"ok": True, "name": getattr(actor, "name", "vlm"),
                                        "prompt": getattr(actor, "prompt", None)})
            return self._send(404, {"ok": False, "error": f"未知路径 {self.path}"})

        def do_POST(self):
            if self.path.rstrip("/") != "/act":
                return self._send(404, {"ok": False, "error": f"未知路径 {self.path}"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > MAX_BODY:
                return self._send(400, {"ok": False, "error": f"请求体大小不合法: {n}"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("请求体必须是 JSON 对象")
                image = decode_image(req.get("image_b64"))
                instruction = str(req.get("instruction") or "").strip()
                sub_task = str(req.get("sub_task") or "").strip()
                if not instruction and not sub_task:
                    raise ValueError("instruction 和 sub_task 不能都为空")
                size = req.get("screen_size") or list(image.size)
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
                return self._send(400, {"ok": False, "error": str(e)})
            try:
                with lock:
                    r = actor.predict(image, instruction or sub_task, sub_task or None, screen_size=tuple(size))
            except Exception as e:
                return self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return self._send(200, {"ok": bool(r.get("ok")), "actions": r.get("actions", []), "raw": r.get("raw", ""),
                                    "sec": r.get("sec"), "error": r.get("error")})

    return Handler


def make_server(actor, host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    return HTTPServer((host, port), make_handler(actor))


def main(argv=None):
    ap = argparse.ArgumentParser(description="VLM 动作 HTTP 服务")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；不填为基座")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--host", default="0.0.0.0", help="WSL2 里用 0.0.0.0，Windows 才能通过 localhost 访问")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args(argv)
    from .infer import VLMActor
    from .modeling import DEFAULT_MODEL
    from .prompts import DEFAULT_VARIANT
    actor = VLMActor.from_pretrained(a.model or DEFAULT_MODEL, adapter=a.adapter, prompt=a.prompt or DEFAULT_VARIANT,
                                     load_4bit=a.load_4bit)
    srv = make_server(actor, a.host, a.port)
    print(f"[服务] {actor.name} 已就绪：http://{a.host}:{a.port}  （Ctrl+C 退出）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
