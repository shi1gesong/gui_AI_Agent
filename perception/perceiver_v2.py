"""
屏幕感知 v2（第6周）：更快 + 更准

速度（第4周真机每步 OCR 4~6 秒，是端到端耗时的大头）：
  1. 前台窗口 ROI：v1 先对整屏 2560×1440 做 OCR，再按前台窗口过滤结果；v2 先按前台窗口裁剪再 OCR，
     窗口只占屏幕 15% 时 OCR 的像素量只有原来的 15%
  2. 帧缓存：画面（裁剪区域）与之前某一帧完全相同 → 直接复用那一帧的 OCR 结果（重试、等待加载时很常见）
  3. 增量识别：与上一帧相比只有一小块变化（输入了文字、选中了联系人）→ 只对变化区域做 OCR，
     其余元素沿用上一帧；变化超过 incr_max_ratio 或连续增量 full_every 次后做一次完整识别，防止误差累积

准确率：
  4. 碎片合并：OCR 常把字间距大的按钮拆成“发”“送”两个框 → 同一行、间距很小的短碎片合并
  5. 词表纠错：界面常用词（发送/搜索/确定/消息内容…）+ 本次任务里出现的目标词，
     长度 ≥ 3 且只错 1 个字的识别结果改成词表里的词（原文保留在 raw_text）
  6. 控件检测（perception/ui_detector.py）：找出输入框 / 按钮，给标签文字关联 input_bbox
  7. 可选预处理：灰度 + 自动对比度、小区域放大（preprocess="contrast" / "upscale"）

每次感知在 Observation.extra 中记录 timing（capture/ocr/post/widgets/total 秒）、cache（hit/incremental/miss/full）、
roi（截图像素坐标）、ocr_area_ratio（实际送进 OCR 的像素占整屏比例）。
"""
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

try:
    from agent.grounding import normalize_text
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.grounding import normalize_text

from perception.screen_parser import Observation, ScreenPerceiver, filter_by_region

UI_LEXICON = ["发送", "搜索", "确定", "取消", "保存", "打开", "关闭", "设置", "文件", "编辑", "查看", "帮助", "登录",
              "注册", "下一步", "上一步", "完成", "消息内容", "联系人", "请选择联系人", "当前聊天", "文件传输助手",
              "另存为", "新建", "全部", "返回", "刷新", "确认", "提交", "删除", "复制", "粘贴", "地址栏", "搜索框"]


def _rect_px(include, scale, offset, image_size) -> Optional[List[int]]:
    """鼠标逻辑坐标矩形 -> 截图像素矩形（裁剪到图像范围内）"""
    if not include or len(include) != 4:
        return None
    sx, sy = scale
    ox, oy = offset
    W, H = image_size
    x1 = max(0, int((include[0] - ox) * sx))
    y1 = max(0, int((include[1] - oy) * sy))
    x2 = min(W, int(round((include[2] - ox) * sx)))
    y2 = min(H, int(round((include[3] - oy) * sy)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def _frame_key(img, rect) -> str:
    small = img.convert("L").resize((96, 54))
    return hashlib.md5(small.tobytes() + str(rect).encode()).hexdigest()


def _is_ascii_word(s: str) -> bool:
    return bool(s) and all(ord(ch) < 128 for ch in s)


def merge_fragments(raw: List[dict], gap_ratio: float = 0.35, short: int = 2) -> List[dict]:
    """同一行、水平间距 ≤ gap_ratio × 行高、且至少一段不超过 short 个字的相邻碎片合并（截图像素坐标）"""
    items = sorted([dict(r) for r in raw if isinstance(r, dict) and r.get("bbox")],
                   key=lambda r: (round((r["bbox"][1] + r["bbox"][3]) / 2 / 8), r["bbox"][0]))
    out: List[dict] = []
    for r in items:
        if out:
            p = out[-1]
            a, b = p["bbox"], r["bbox"]
            ha, hb = a[3] - a[1], b[3] - b[1]
            inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            same_line = inter >= 0.6 * min(ha, hb) and max(ha, hb) <= 1.6 * max(1, min(ha, hb))
            gap = b[0] - a[2]
            ta, tb = str(p.get("text", "")).strip(), str(r.get("text", "")).strip()
            if (same_line and -2 <= gap <= gap_ratio * max(ha, hb)
                    and min(len(normalize_text(ta)), len(normalize_text(tb))) <= short):
                sep = " " if _is_ascii_word(ta[-1:]) and _is_ascii_word(tb[:1]) else ""
                p.setdefault("fragments", [ta]).append(tb)
                p["text"] = ta + sep + tb
                p["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                p["confidence"] = min(float(p.get("confidence", 1.0)), float(r.get("confidence", 1.0)))
                continue
        out.append(r)
    return out


def _one_sub(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def correct_text(text: str, lexicon: Iterable[str]) -> Optional[str]:
    """长度 ≥ 3、与词表某词只差 1 个字时返回该词；否则 None"""
    t = normalize_text(text)
    if len(t) < 3:
        return None
    for w in lexicon:
        nw = normalize_text(w)
        if t == nw:
            return None
        if _one_sub(t, nw):
            return w
    return None


def _preprocess(img, mode: Optional[str]):
    """返回 (处理后的图, 坐标缩放系数)"""
    if not mode:
        return img, 1.0
    from PIL import ImageOps
    if mode == "contrast":
        return ImageOps.autocontrast(img.convert("L")).convert("RGB"), 1.0
    if mode == "upscale":
        f = 2.0 if max(img.size) <= 1600 else 1.0
        if f == 1.0:
            return img, 1.0
        return img.resize((int(img.width * f), int(img.height * f))), f
    raise ValueError(f"未知预处理 {mode!r}，可选 contrast / upscale")


class PerceiverV2(ScreenPerceiver):
    def __init__(self, capture=None, ocr=None, min_confidence: float = 0.3, monitor_index: int = 1,
                 logical_size_fn: Optional[Callable[[], Tuple[int, int]]] = None, offset=(0, 0),
                 max_elements: int = 300, region_fn: Optional[Callable[[], dict]] = None,
                 roi: bool = True, cache: bool = True, cache_size: int = 8, incremental: bool = True,
                 incr_max_ratio: float = 0.35, full_every: int = 5, merge: bool = True,
                 lexicon: Optional[Iterable[str]] = None, widgets: bool = True, preprocess: Optional[str] = None,
                 save_frames: bool = True, clock: Callable[[], float] = time.perf_counter):
        super().__init__(capture=capture, ocr=ocr, min_confidence=min_confidence, monitor_index=monitor_index,
                         logical_size_fn=logical_size_fn, offset=offset, max_elements=max_elements, region_fn=region_fn)
        self.roi, self.use_cache, self.incremental = roi, cache, incremental
        self.cache_size, self.incr_max_ratio, self.full_every = cache_size, incr_max_ratio, full_every
        self.merge, self.widgets_on, self.preprocess = merge, widgets, preprocess
        self.lexicon = list(UI_LEXICON if lexicon is None else lexicon)
        self.extra_words: List[str] = []
        self.save_frames = save_frames
        self.clock = clock
        self._cache: "OrderedDict[str, List[dict]]" = OrderedDict()
        self._prev = None          # (crop_rect, crop_image, raw_px)
        self._incr_count = 0
        self.last_image = None
        self.stats = {"calls": 0, "hit": 0, "incremental": 0, "full": 0, "ocr_pixels": 0, "screen_pixels": 0}

    # ---- 词表 ----
    def add_words(self, words: Iterable[str]):
        """把本次任务里的目标词（联系人名、按钮名…）加入纠错词表"""
        for w in words or []:
            w = str(w or "").strip()
            if w and w not in self.extra_words:
                self.extra_words.append(w)

    def reset_cache(self):
        self._cache.clear()
        self._prev = None
        self._incr_count = 0

    # ---- 截图（供效果校验用，不跑 OCR） ----
    def grab(self):
        img = self.capture.capture_full_screen(monitor_index=self.monitor_index, save=False)
        self.last_image = img
        return img

    # ---- OCR 一块区域（截图像素坐标；返回的 bbox 已加上区域偏移） ----
    def _ocr_region(self, img, rect, tag: str) -> List[dict]:
        crop = img.crop(tuple(rect)) if rect != [0, 0, img.width, img.height] else img
        crop, f = _preprocess(crop, self.preprocess)
        save_dir = Path(getattr(self.capture, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"_ocr_{tag}.png"   # 每次覆盖同一个文件，不堆积
        crop.save(path)
        self.stats["ocr_pixels"] += crop.width * crop.height / (f * f)
        raw = self.ocr.detect(str(path)) or []
        out = []
        for r in raw:
            try:
                x1, y1, x2, y2 = (float(v) for v in r["bbox"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append({**r, "bbox": [x1 / f + rect[0], y1 / f + rect[1], x2 / f + rect[0], y2 / f + rect[1]]})
        return out

    def _ocr_frame(self, img, rect) -> Tuple[List[dict], str, Optional[List[int]]]:
        """返回 (raw_px, 缓存状态, 本次实际 OCR 的区域)"""
        from agent.verifier import cells_to_bbox, changed_cells
        key = _frame_key(img.crop(tuple(rect)), rect)
        if self.use_cache and key in self._cache:
            self._cache.move_to_end(key)
            raw = [dict(r) for r in self._cache[key]]
            self._prev = (rect, img.crop(tuple(rect)), raw)
            self.stats["hit"] += 1
            return raw, "hit", None
        crop_img = img.crop(tuple(rect))
        status, ocr_rect = "full", rect
        raw = None
        if (self.incremental and self._prev is not None and self._prev[0] == rect
                and self._incr_count < self.full_every):
            grid = (32, 18)
            cells = changed_cells(self._prev[1], crop_img, grid=grid)
            ratio = len(cells) / (grid[0] * grid[1])
            if 0 < ratio <= self.incr_max_ratio:
                cb = cells_to_bbox(cells, crop_img.size, grid)
                m = 8
                local = [max(0, cb[0] - m), max(0, cb[1] - m), min(crop_img.width, cb[2] + m),
                         min(crop_img.height, cb[3] + m)]
                ocr_rect = [local[0] + rect[0], local[1] + rect[1], local[2] + rect[0], local[3] + rect[1]]
                # 被变化区域切到一部分的旧元素：把它整个纳入重识别区域，否则只会识别出半截文字
                for _ in range(3):
                    grow = [r["bbox"] for r in self._prev[2] if _intersects(r["bbox"], ocr_rect)
                            and not _inside_rect(r["bbox"], ocr_rect)]
                    if not grow:
                        break
                    ocr_rect = [max(rect[0], int(min([ocr_rect[0]] + [b[0] - 2 for b in grow]))),
                                max(rect[1], int(min([ocr_rect[1]] + [b[1] - 2 for b in grow]))),
                                min(rect[2], int(max([ocr_rect[2]] + [b[2] + 2 for b in grow]))),
                                min(rect[3], int(max([ocr_rect[3]] + [b[3] + 2 for b in grow])))]
                fresh = self._ocr_region(img, ocr_rect, "inc")
                kept = [r for r in self._prev[2] if not _intersects(r["bbox"], ocr_rect)]
                raw = kept + fresh
                status = "incremental"
                self._incr_count += 1
                self.stats["incremental"] += 1
        if raw is None:
            raw = self._ocr_region(img, rect, "full")
            self._incr_count = 0
            self.stats["full"] += 1
        if self.use_cache:
            self._cache[key] = [dict(r) for r in raw]
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._prev = (rect, crop_img, raw)
        return raw, status, ocr_rect

    def perceive(self) -> Observation:
        if self.capture is None or self.ocr is None:
            raise RuntimeError("PerceiverV2 需要 capture（截图）和 ocr（文字识别）两个组件")
        t0 = self.clock()
        timing = {}
        img = self.capture.capture_full_screen(monitor_index=self.monitor_index, save=False)
        self.last_image = img
        save_dir = Path(getattr(self.capture, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        shot = save_dir / f"step_{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
        if self.save_frames:
            img.save(shot)
        logical = tuple(self.logical_size_fn())
        scale = self.compute_scale(img.size, logical)
        extra = {}
        region = {}
        if self.region_fn is not None:
            try:
                region = self.region_fn() or {}
            except Exception as e:
                extra["region_error"] = f"{type(e).__name__}: {e}"
        timing["capture"] = round(self.clock() - t0, 4)

        t1 = self.clock()
        full = [0, 0, img.width, img.height]
        rect = (_rect_px(region.get("include"), scale, self.offset, img.size) if self.roi else None) or full
        raw, status, ocr_rect = self._ocr_frame(img, rect)
        timing["ocr"] = round(self.clock() - t1, 4)

        t2 = self.clock()
        n_raw = len(raw)
        if self.merge:
            raw = merge_fragments(raw)
        elements = self.postprocess(raw, scale)
        corrected = 0
        lex = self.lexicon + self.extra_words
        for e in elements:
            fixed = correct_text(e["text"], lex)
            if fixed:
                e["raw_text"], e["text"] = e["text"], fixed
                corrected += 1
        if region:
            before = len(elements)
            elements = filter_by_region(elements, region.get("include"), region.get("exclude") or [])
            extra.update(region={"include": region.get("include"), "exclude": region.get("exclude") or []},
                         filtered_out=before - len(elements))
        timing["post"] = round(self.clock() - t2, 4)

        if self.widgets_on:
            t3 = self.clock()
            try:
                from perception.ui_detector import associate_labels, detect_widgets, empty_widgets
                sx, sy = scale
                ox, oy = self.offset
                ws = detect_widgets(img.crop(tuple(rect)))
                for w in ws:  # 裁剪坐标 -> 截图像素 -> 鼠标逻辑坐标
                    b = [w["bbox"][0] + rect[0], w["bbox"][1] + rect[1], w["bbox"][2] + rect[0], w["bbox"][3] + rect[1]]
                    w["bbox_px"] = b
                    w["bbox"] = [int(round(b[0] / sx + ox)), int(round(b[1] / sy + oy)),
                                 int(round(b[2] / sx + ox)), int(round(b[3] / sy + oy))]
                associate_labels(elements, ws)
                extra["widgets"] = ws
                extra["empty_widgets"] = len(empty_widgets(elements, ws))
            except Exception as e:  # 没装 opencv 等：不影响文字感知
                extra["widgets_error"] = f"{type(e).__name__}: {e}"
            timing["widgets"] = round(self.clock() - t3, 4)
        timing["total"] = round(self.clock() - t0, 4)

        area = img.width * img.height
        self.stats["calls"] += 1
        self.stats["screen_pixels"] += area
        ocr_area = 0 if ocr_rect is None else (ocr_rect[2] - ocr_rect[0]) * (ocr_rect[3] - ocr_rect[1])
        extra.update(timing=timing, cache=status, roi=rect, ocr_rect=ocr_rect, ocr_area_ratio=round(ocr_area / area, 4),
                     corrected=corrected, merged=n_raw - len(raw), offset=list(self.offset))
        return Observation(elements=elements, screenshot_path=str(shot) if self.save_frames else None,
                           image_size=tuple(img.size), logical_size=logical, scale=scale,
                           elapsed_sec=timing["total"], raw_count=n_raw, extra=extra)


def _inside_rect(a, r) -> bool:
    return a[0] >= r[0] and a[1] >= r[1] and a[2] <= r[2] and a[3] <= r[3]


def _intersects(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
