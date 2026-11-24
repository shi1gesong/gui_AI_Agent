"""
无文字 UI 控件检测 + 标签关联（第6周新增，OpenCV）

OCR 只能找到“有字”的东西。第4周真机 T4 暴露了这个问题：输入框里是灰色占位符、OCR 读不出来，
Agent 找不到输入框，只好在模拟聊天程序里加了一个可点击的黑色“消息内容”标签来绕过。
真实软件不会配合我们改界面，所以第6周用 OpenCV 直接找矩形控件：

  1. 灰度 → Canny 边缘 → 轻微膨胀 → 轮廓 → 外接矩形
  2. 按尺寸 / 长宽比 / 内部颜色过滤，分成 input（扁长、内部浅色均匀）/ button（内部填充色与外部明显不同）/ box
  3. 标签关联：文字元素右侧同一行（或正下方）最近的 input 框，记为该文字的 input_bbox。
     执行“在‘消息内容’框中输入X”时，执行器点击 input_bbox 的中心（输入框本身），而不是标签文字。
  4. 文字落在某个控件内部时记下 widget / widget_bbox（例如“发送”在绿色按钮里）

坐标约定：detect_widgets 输入输出都是“截图像素”坐标；关联前由感知器换算成鼠标逻辑坐标。
"""
from typing import List, Optional


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _as_array(img):
    import numpy as np
    if hasattr(img, "convert"):
        return np.asarray(img.convert("RGB"))
    return np.asarray(img)


def detect_widgets(img, min_w: int = 24, min_h: int = 14, max_area_ratio: float = 0.25,
                   max_widgets: int = 200) -> List[dict]:
    """返回 [{"kind", "bbox"(截图像素), "inner_mean", "inner_std"}]，按从上到下、从左到右排序"""
    import cv2
    import numpy as np
    rgb = _as_array(img)
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape[:2]
    if H < min_h or W < min_w:
        return []
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h or w * h > max_area_ratio * W * H:
            continue
        if w >= W - 2 and h >= H - 2:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        rectish = 4 <= len(approx) <= 8 and cv2.contourArea(c) >= 0.55 * w * h
        # 只有 1px 细边框的输入框：轮廓面积可能很小（只是边），用“外接矩形周长 ≈ 轮廓周长”判断
        outline = abs(peri - 2 * (w + h)) <= 0.25 * 2 * (w + h)
        if not (rectish or outline):
            continue
        cands.append([x, y, x + w, y + h])
    # 去重：同一个控件的内外两条边会得到两个几乎重合的矩形，保留外面的
    cands.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept = []
    for b in cands:
        if any(_iou(b, k) > 0.75 for k in kept):
            continue
        kept.append(b)
    out = []
    for b in kept[:max_widgets]:
        x1, y1, x2, y2 = b
        pad = 3 if (x2 - x1) > 10 and (y2 - y1) > 10 else 0
        inner = gray[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
        if inner.size == 0:
            continue
        ring = np.concatenate([gray[max(0, y1 - 3):y1, x1:x2].ravel(), gray[y2:min(H, y2 + 3), x1:x2].ravel()])
        inner_mean, inner_std = float(inner.mean()), float(inner.std())
        outer_mean = float(ring.mean()) if ring.size else inner_mean
        w, h = x2 - x1, y2 - y1
        if w / max(h, 1) >= 3.0 and inner_mean >= 200 and inner_std < 40:
            kind = "input"
        elif abs(inner_mean - outer_mean) >= 25 and w / max(h, 1) < 8:
            kind = "button"
        else:
            kind = "box"
        out.append({"kind": kind, "bbox": [int(x1), int(y1), int(x2), int(y2)], "inner_mean": round(inner_mean, 1),
                    "inner_std": round(inner_std, 1)})
    out.sort(key=lambda d: (d["bbox"][1] // 10, d["bbox"][0]))
    return out


def _contains(outer, inner, tol: int = 2) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol
            and inner[3] <= outer[3] + tol)


def _v_overlap(a, b) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def _h_overlap(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def associate_labels(elements: List[dict], widgets: List[dict], key: str = "bbox",
                     max_gap_ratio: float = 3.0) -> List[dict]:
    """
    elements: OCR 文字元素（含 key 指定的坐标）；widgets: 同一坐标系的控件 [{"kind","bbox"}]。
    原地给文字元素加上 widget/widget_bbox（文字在控件内）或 input_bbox（标签右侧/下方的输入框），返回 elements。
    """
    inputs = [w for w in widgets if w.get("kind") == "input"]
    for e in elements:
        b = e.get(key)
        if not b or len(b) != 4:
            continue
        inside = [w for w in widgets if _contains(w["bbox"], b)]
        if inside:
            w = min(inside, key=lambda w: (w["bbox"][2] - w["bbox"][0]) * (w["bbox"][3] - w["bbox"][1]))
            e["widget"], e["widget_bbox"] = w["kind"], list(w["bbox"])
            if w["kind"] == "input":
                e["input_bbox"] = list(w["bbox"])
                continue
            b = w["bbox"]   # 文字在带底色的标签块里：用标签块的边界找旁边的输入框
        h = max(1, b[3] - b[1])
        best: Optional[tuple] = None
        for w in inputs:
            wb = w["bbox"]
            if _v_overlap(b, wb) >= 0.5 and wb[0] >= b[2] - 2:          # 同一行、在右侧
                gap = wb[0] - b[2]
            elif _h_overlap(b, wb) >= 0.3 and wb[1] >= b[3] - 2:        # 正下方
                gap = wb[1] - b[3]
            else:
                continue
            if gap <= max_gap_ratio * h and (best is None or gap < best[0]):
                best = (gap, wb)
        if best is not None:
            e["input_bbox"] = list(best[1])
    return elements


def empty_widgets(elements: List[dict], widgets: List[dict], key: str = "bbox") -> List[dict]:
    """没有任何文字落在里面的控件（图标按钮、空输入框）"""
    out = []
    for w in widgets:
        if not any(e.get(key) and _contains(w["bbox"], e[key]) for e in elements):
            out.append(w)
    return out
