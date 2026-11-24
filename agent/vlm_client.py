"""
VLM 决策客户端（第5周，Windows 端）

GUIAgent 的 --act vlm 模式：需要看屏幕的子任务不再走 “OCR → 文字匹配 → 坐标”，
而是把整张截图发给 WSL2 里的 finetune/serve.py（Qwen2-VL + LoRA），模型直接输出 0~1000 归一化坐标，
这里换算成鼠标逻辑坐标（与 DPI 缩放无关），交给 ControllerExecutor 执行。

好处：没有文字的图标按钮（第4周已知局限 3）也能点；不跑 OCR，感知更快。
代价：需要 GPU 服务；坐标精度取决于模型（见第5周对比报告）。
"""
import base64
import io
import json
from typing import Callable, Optional

try:
    from finetune.action_format import to_agent_action, parse_output
except ImportError:  # agent/ 在 sys.path 而仓库根目录不在时
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from finetune.action_format import to_agent_action, parse_output

DEFAULT_VLM_URL = "http://localhost:8765"


def _default_post(url: str, payload: dict, timeout: float) -> dict:
    import requests
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(f"VLM 服务返回的不是 JSON（HTTP {r.status_code}）")
    if r.status_code >= 400:
        raise RuntimeError(f"VLM 服务报错（HTTP {r.status_code}）: {data.get('error')}")
    return data


def encode_image(path: str, max_side: int = 1280, quality: int = 90) -> str:
    """截图压成 JPEG 再 base64：2560×1440 PNG 有好几 MB，缩到长边 1280 后约 200KB；坐标是归一化的，缩放不影响"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLMClient:
    def __init__(self, base_url: str = DEFAULT_VLM_URL, timeout: float = 120.0, post: Optional[Callable] = None,
                 max_side: int = 1280, encoder: Callable = encode_image):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.post = post or _default_post
        self.max_side = max_side
        self.encoder = encoder
        self.last = None

    def act(self, subtask: str, instruction: str, obs) -> dict:
        """obs: perception.screen_parser.Observation（需要 screenshot_path、image_size、logical_size）"""
        path = getattr(obs, "screenshot_path", None)
        if not path:
            raise ValueError("VLM 决策需要截图路径（Observation.screenshot_path 为空）")
        logical = tuple(getattr(obs, "logical_size", (0, 0)) or (0, 0))
        if not logical[0] or not logical[1]:
            logical = tuple(getattr(obs, "image_size", (0, 0)) or (0, 0))
        if not logical[0] or not logical[1]:
            raise ValueError("缺少屏幕尺寸，无法把归一化坐标换算成鼠标坐标")
        offset = tuple((getattr(obs, "extra", {}) or {}).get("offset", (0, 0)))
        payload = {"image_b64": self.encoder(path, self.max_side), "instruction": instruction or subtask,
                   "sub_task": subtask, "screen_size": list(getattr(obs, "image_size", logical) or logical)}
        resp = self.post(f"{self.base_url}/act", payload, self.timeout)
        if not isinstance(resp, dict):
            raise RuntimeError("VLM 服务返回格式错误")
        actions = resp.get("actions")
        if not actions and resp.get("raw"):  # 服务端没解析出来，客户端再宽容解析一次
            actions = parse_output(resp["raw"])["actions"]
        self.last = {"raw": resp.get("raw"), "sec": resp.get("sec"), "actions": actions}
        if not actions:
            return {"action_type": None, "_invalid": True, "_retryable": True,
                    "_reasons": [f"VLM 没有给出可执行动作: {resp.get('error') or (resp.get('raw') or '')[:80]}"],
                    "vlm_raw": resp.get("raw")}
        a = to_agent_action(actions, logical, offset)
        a["vlm_raw"] = resp.get("raw")
        a["vlm_actions"] = actions
        return a

    def health(self, get: Optional[Callable] = None) -> dict:
        if get is None:
            import requests
            return requests.get(f"{self.base_url}/health", timeout=5).json()
        return get(f"{self.base_url}/health")


def describe(actions) -> str:
    return json.dumps(actions, ensure_ascii=False)
