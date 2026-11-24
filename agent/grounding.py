"""
文本 → OCR 元素 → 屏幕坐标 的 grounding（第4周新增）

第3周的 Acting 只输出 target_text（例如「发送」），没有把文字换算成可以点击的坐标。
本模块在 OCR 结果里为 target_text 找到最匹配的元素，返回其 bbox 中心点：

  1. 归一化：去空白、转小写、全角转半角、去掉常见标点（OCR 常把「发送」识别成「发 送」「发送。」）
  2. 打分：完全一致 1.0 > 一方包含另一方（按长度比例 0.7~0.95）> difflib 相似度（≥ min_score 才接受）
  3. 同分时优先 OCR 置信度高、面积小（更贴近按钮本身）、位置靠上的元素
  4. 元素坐标（屏幕截图的物理像素）在 perception/screen_parser.py 中已换算成 pyautogui 的逻辑坐标，
     这里也提供 scale_point() 供单独使用（DPI 缩放 ≠ 100% 时截图像素 ≠ 鼠标坐标）
"""
import difflib
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

_PUNCT_RE = re.compile(r"[\s​·•。.，,、:：;；!！?？()（）\[\]【】<>《》\"'“”‘’|_\-—~…]+")


def normalize_text(s) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    return _PUNCT_RE.sub("", s).lower()


def match_score(target: str, text: str) -> float:
    """target 与某个 OCR 文本的匹配分数，0~1"""
    t, e = normalize_text(target), normalize_text(text)
    if not t or not e:
        return 0.0
    if t == e:
        return 1.0
    if t in e or e in t:
        short, long_ = sorted((len(t), len(e)))
        return 0.7 + 0.25 * short / long_
    return difflib.SequenceMatcher(None, t, e).ratio() * 0.9


def bbox_center(bbox: Sequence) -> Optional[Tuple[int, int]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def scale_point(point: Tuple[float, float], scale: Tuple[float, float] = (1.0, 1.0),
                offset: Tuple[float, float] = (0, 0)) -> Tuple[int, int]:
    """截图像素坐标 -> 鼠标逻辑坐标：(x / sx + ox, y / sy + oy)"""
    sx, sy = scale
    if sx <= 0 or sy <= 0:
        raise ValueError(f"缩放比例必须为正数，实际 {scale}")
    return int(round(point[0] / sx + offset[0])), int(round(point[1] / sy + offset[1]))


class Grounder:
    def __init__(self, min_score: float = 0.6, min_confidence: float = 0.0):
        self.min_score = min_score
        self.min_confidence = min_confidence

    def rank(self, target: str, elements: List[dict]) -> List[Tuple[float, dict]]:
        scored = []
        for e in elements or []:
            if not isinstance(e, dict):
                continue
            if float(e.get("confidence", 1.0) or 0.0) < self.min_confidence:
                continue
            sc = match_score(target, e.get("text"))
            if sc >= self.min_score:
                scored.append((sc, e))

        def key(item):
            sc, e = item
            bbox = e.get("bbox") or [0, 0, 0, 0]
            try:
                area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                top = bbox[1]
            except (TypeError, IndexError):
                area, top = float("inf"), float("inf")
            return (-round(sc, 3), -float(e.get("confidence", 1.0) or 0.0), area, top)

        return sorted(scored, key=key)

    def locate(self, target: str, elements: List[dict]) -> Optional[dict]:
        """返回 {"element", "score", "point"}；找不到或该元素没有有效 bbox 时返回 None"""
        for sc, e in self.rank(target, elements):
            pt = bbox_center(e.get("bbox"))
            if pt is not None:
                return {"element": e, "score": round(sc, 3), "point": pt}
        return None
