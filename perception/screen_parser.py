"""
屏幕感知管线（第4周新增）：截图 → OCR → 过滤/去重/排序 → 坐标换算 → 给 Agent 的屏幕描述

Observation.elements 中每个元素：
  {"text", "bbox"（鼠标逻辑坐标，可直接交给 pyautogui 点击）, "bbox_px"（截图像素坐标）, "confidence"}

DPI 缩放：Windows 显示缩放 ≠ 100% 且进程未声明 DPI 感知时，截图是物理像素、pyautogui 是逻辑像素，
两者相差 scale 倍（例如 150% 缩放时 2560 / 1707 ≈ 1.5）。每次感知都重新计算
scale = 截图尺寸 / pyautogui.size()，把 OCR 坐标除以 scale 再加上显示器偏移，保证点击位置正确。
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    from agent.grounding import normalize_text
except ImportError:  # 把 perception/ 单独加入 sys.path 时
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.grounding import normalize_text


@dataclass
class Observation:
    elements: List[dict]
    screenshot_path: Optional[str] = None
    image_size: Tuple[int, int] = (0, 0)
    logical_size: Tuple[int, int] = (0, 0)
    scale: Tuple[float, float] = (1.0, 1.0)
    elapsed_sec: float = 0.0
    raw_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_context(self, max_elements: int = 80) -> str:
        """与第3周相同的格式："识别到的UI元素: [{"text":..,"bbox":[..]}, ...]"，元素过多时截断"""
        items = [{"text": e["text"], "bbox": e["bbox"]} for e in self.elements[:max_elements]]
        ctx = "识别到的UI元素: " + json.dumps(items, ensure_ascii=False)
        if len(self.elements) > max_elements:
            ctx += f"\n（另有 {len(self.elements) - max_elements} 个元素未列出）"
        return ctx

    def texts(self) -> List[str]:
        return [e["text"] for e in self.elements]

    def summary(self) -> dict:
        return {"screenshot": self.screenshot_path, "elements": len(self.elements), "raw": self.raw_count,
                **({"filtered_out": self.extra["filtered_out"]} if "filtered_out" in self.extra else {}),
                "image_size": list(self.image_size), "logical_size": list(self.logical_size),
                "scale": [round(self.scale[0], 4), round(self.scale[1], 4)], "sec": round(self.elapsed_sec, 2)}


def _default_logical_size():
    import pyautogui
    w, h = pyautogui.size()
    return int(w), int(h)


class ScreenPerceiver:
    def __init__(self, capture=None, ocr=None, min_confidence: float = 0.3, monitor_index: int = 1,
                 logical_size_fn: Optional[Callable[[], Tuple[int, int]]] = None,
                 offset: Tuple[int, int] = (0, 0), max_elements: int = 300,
                 region_fn: Optional[Callable[[], dict]] = None):
        """
        capture: perception.screenshot.ScreenCapture（或有 capture_full_screen(monitor_index, save) 与 save_dir 的对象）
        ocr: perception.ocr_detector.OCRDetector（或有 detect(image_path) 的对象）
        logical_size_fn: 返回鼠标坐标系屏幕尺寸，默认 pyautogui.size()
        offset: 所截显示器在鼠标坐标系中的左上角（多显示器时非 0）
        region_fn: 返回 {"include": [l,t,r,b] 或 None, "exclude": [[l,t,r,b], ...]}（鼠标坐标）。
                   只保留中心点在 include 内、且不在任何 exclude 内的元素。通常 include = 前台窗口，
                   exclude = Agent 自己的命令行窗口（2026-11-24 真机 T4：OCR 读到了命令行里打印的子任务文字，
                   把它当成“消息内容”输入框点了上去，消息输进了命令行）
        """
        self.capture = capture
        self.ocr = ocr
        self.min_confidence = min_confidence
        self.monitor_index = monitor_index
        self.logical_size_fn = logical_size_fn or _default_logical_size
        self.offset = offset
        self.max_elements = max_elements
        self.region_fn = region_fn

    # ---- 纯函数部分（便于单元测试） ----
    @staticmethod
    def compute_scale(image_size, logical_size) -> Tuple[float, float]:
        iw, ih = image_size
        lw, lh = logical_size
        if min(iw, ih, lw, lh) <= 0:
            raise ValueError(f"尺寸无效: 截图 {image_size}, 逻辑屏幕 {logical_size}")
        return iw / lw, ih / lh

    def postprocess(self, raw: List[dict], scale=(1.0, 1.0), offset=None) -> List[dict]:
        """过滤低置信度/空文本/无效框 → 去重（同文本且框重叠）→ 按从上到下、从左到右排序 → 坐标换算"""
        ox, oy = self.offset if offset is None else offset
        sx, sy = scale
        kept = []
        for r in raw or []:
            if not isinstance(r, dict):
                continue
            text = str(r.get("text") or "").strip()
            conf = float(r.get("confidence", 1.0) or 0.0)
            bbox = r.get("bbox")
            if not text or not normalize_text(text) or conf < self.min_confidence:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            kept.append({"text": text, "bbox_px": [int(x1), int(y1), int(x2), int(y2)], "confidence": round(conf, 4)})

        deduped = []
        for e in sorted(kept, key=lambda e: -e["confidence"]):
            if any(normalize_text(d["text"]) == normalize_text(e["text"]) and _iou(d["bbox_px"], e["bbox_px"]) > 0.5
                   for d in deduped):
                continue
            deduped.append(e)
        # 同一行（纵向中心相差 < 行高一半）按 x 排序
        deduped.sort(key=lambda e: (round(((e["bbox_px"][1] + e["bbox_px"][3]) / 2) / 20), e["bbox_px"][0]))
        for e in deduped:
            x1, y1, x2, y2 = e["bbox_px"]
            e["bbox"] = [int(round(x1 / sx + ox)), int(round(y1 / sy + oy)),
                         int(round(x2 / sx + ox)), int(round(y2 / sy + oy))]
        return deduped[: self.max_elements]

    # ---- 真正截图 + OCR ----
    def perceive(self) -> Observation:
        if self.capture is None or self.ocr is None:
            raise RuntimeError("ScreenPerceiver 需要 capture（截图）和 ocr（文字识别）两个组件")
        t0 = time.time()
        img = self.capture.capture_full_screen(monitor_index=self.monitor_index, save=False)
        save_dir = Path(getattr(self.capture, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"step_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
        img.save(path)
        raw = self.ocr.detect(str(path))
        logical = tuple(self.logical_size_fn())
        scale = self.compute_scale(img.size, logical)
        elements = self.postprocess(raw, scale)
        extra = {}
        if self.region_fn is not None:
            try:
                region = self.region_fn() or {}
            except Exception as e:  # 取窗口位置失败时不过滤，照常返回全屏元素
                region = {}
                extra["region_error"] = f"{type(e).__name__}: {e}"
            before = len(elements)
            elements = filter_by_region(elements, region.get("include"), region.get("exclude") or [])
            extra.update(region={"include": region.get("include"), "exclude": region.get("exclude") or []},
                         filtered_out=before - len(elements))
        return Observation(elements=elements, screenshot_path=str(path), image_size=tuple(img.size),
                           logical_size=logical, scale=scale, elapsed_sec=time.time() - t0, raw_count=len(raw or []),
                           extra=extra)


def _inside(pt, rect) -> bool:
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


def filter_by_region(elements: List[dict], include=None, exclude=()) -> List[dict]:
    """按元素中心点过滤：在 include 矩形内（None 表示不限）且不在任何 exclude 矩形内"""
    valid = lambda r: r is not None and len(r) == 4 and r[2] > r[0] and r[3] > r[1]  # noqa: E731
    include = include if valid(include) else None
    exclude = [r for r in exclude if valid(r)]
    out = []
    for e in elements:
        b = e.get("bbox") or [0, 0, 0, 0]
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if include is not None and not _inside(c, include):
            continue
        if any(_inside(c, r) for r in exclude):
            continue
        out.append(e)
    return out


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StaticPerceiver:
    """返回固定元素列表的感知器：用于 --dry-run、单元测试，或接入第3周的 screen_context 字符串"""

    def __init__(self, elements: Optional[List[dict]] = None, frames: Optional[List[List[dict]]] = None):
        self.frames = [list(f) for f in frames] if frames else None
        self.elements = list(elements or [])
        self.calls = 0

    def perceive(self) -> Observation:
        self.calls += 1
        if self.frames:
            els = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        else:
            els = self.elements
        els = [{**e, "bbox": e.get("bbox", [0, 0, 0, 0])} for e in els]
        return Observation(elements=els, raw_count=len(els))
