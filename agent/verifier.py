"""
动作效果校验与目标结果检查（第6周新增）

v1.0 判断一个动作“成功”的依据只有一条：执行器没抛异常。真机上常见的失败恰恰不抛异常：
点击被弹窗挡住、窗口还没响应、文字输到了别的窗口……执行器都报 ✓。v2.0 增加两层检查：

1. 动作级：截图前后对比（EffectVerifier）
   把前后两帧缩成小图逐像素比较（RGB 任一通道差超过阈值算变化），变化像素占比 < min_change 视为“界面没有变化”。
   只对“应当引起界面变化”的动作（click / double_click / type / drag）判失败；
   hotkey/key/scroll/wait 等可能没有可见变化的动作只记录、不判失败（例如 Ctrl+S 保存一个没改过的文件）。
   还会给出变化区域（网格单元），写入轨迹便于排查“点了哪里、哪里变了”。

2. 目标级：后置条件（Postcondition）
   每个目标（Goal）可以声明完成后应满足的状态，例如：
     window_exists  标题含“天气预报”的窗口存在（搜索完成）
     window_absent  标题含“计算器”的窗口不存在（关闭完成）
     text_present   屏幕上出现“我 → 张三：今晚7点开会”（消息已发出）
     file_contains  某文件内容包含指定文字（记事本保存完成）
   window_* / file_* 不需要 OCR，几乎零成本；text_* 需要一次感知。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# 应当引起可见变化的动作类型
EFFECT_REQUIRED = ("click", "double_click", "type", "drag")


def _to_small(img, size: Tuple[int, int] = (160, 90)):
    """PIL.Image -> 缩小的 RGB numpy 数组（对比用）。尺寸不同的两帧会缩放到同一大小。
    不转灰度：亮度相同、颜色不同的变化（例如选中项由灰变绿）转灰度后会被漏掉"""
    import numpy as np
    return np.asarray(img.convert("RGB").resize(size), dtype=np.int16)


def _diff_mask(a, b, size, threshold):
    import numpy as np
    return np.abs(_to_small(a, size) - _to_small(b, size)).max(axis=-1) > threshold


def image_diff_ratio(a, b, threshold: int = 12, size: Tuple[int, int] = (160, 90)) -> float:
    """两帧之间变化像素的比例（0~1）。threshold：任一颜色通道差超过多少算变化（过滤 JPEG/抗锯齿噪声）"""
    if a is None or b is None:
        return 1.0
    return float(_diff_mask(a, b, size, threshold).mean())


def changed_cells(a, b, grid: Tuple[int, int] = (16, 9), threshold: int = 12, cell_ratio: float = 0.0) -> List[Tuple[int, int]]:
    """把画面切成 grid 网格，返回发生变化的网格 (列, 行)"""
    if a is None or b is None:
        return []
    import numpy as np
    gw, gh = grid
    size = (gw * 10, gh * 10)
    diff = _diff_mask(a, b, size, threshold)
    out = []
    for r in range(gh):
        for c in range(gw):
            if diff[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10].mean() > cell_ratio:
                out.append((c, r))
    return out


def cells_to_bbox(cells, image_size, grid=(16, 9)) -> Optional[List[int]]:
    """变化网格 -> 图像像素坐标下的外接矩形"""
    if not cells:
        return None
    W, H = image_size
    gw, gh = grid
    cs = [c for c, _ in cells]
    rs = [r for _, r in cells]
    return [int(min(cs) * W / gw), int(min(rs) * H / gh), int((max(cs) + 1) * W / gw), int((max(rs) + 1) * H / gh)]


class EffectVerifier:
    def __init__(self, min_pixels: int = 4, threshold: int = 12, size: Tuple[int, int] = (320, 180)):
        """
        min_pixels：缩到 320×180 后至少多少个像素变化才算“界面有变化”。
        2560×1440 上一个 20×30 的汉字缩小后约占 2×4 个像素，所以输入一两个字也能检测到；
        单像素级噪声（抗锯齿、JPEG）由 threshold（任一颜色通道差 > 12）过滤。
        """
        self.min_pixels = min_pixels
        self.threshold = threshold
        self.size = size

    @property
    def min_change(self) -> float:
        return self.min_pixels / (self.size[0] * self.size[1])

    def verify(self, action: dict, before, after) -> dict:
        """返回 {"ok", "checked", "diff", "reason", "changed_bbox"}；ok=False 只在应当变化却没变时出现"""
        a_type = (action or {}).get("action_type")
        if before is None or after is None:
            return {"ok": True, "checked": False, "reason": "缺少截图，跳过效果校验"}
        diff = image_diff_ratio(before, after, self.threshold, self.size)
        cells = changed_cells(before, after, threshold=self.threshold)
        out = {"diff": round(diff, 5), "checked": a_type in EFFECT_REQUIRED,
               "changed_bbox": cells_to_bbox(cells, after.size)}
        if a_type in EFFECT_REQUIRED and diff < self.min_change:
            return {**out, "ok": False, "reason": f"界面没有变化（变化像素 {diff:.3%}），动作没有生效"}
        return {**out, "ok": True, "reason": "界面已变化" if diff >= self.min_change else "该动作不要求界面变化"}


# ---------------------------------------------------------------------- #
# 目标级后置条件
# ---------------------------------------------------------------------- #
def _norm(s) -> str:
    return "".join(str(s or "").split()).lower()


def text_matches(want, text, min_ratio: float = 0.85) -> bool:
    """want 是否出现在 text 中：去掉空白和标点后包含即可；want ≥ 6 个字时容忍 OCR 错 1~2 个字（局部相似度 ≥ min_ratio）。
    只在单个元素内匹配，不把相邻元素拼起来（否则“当前聊天：张三”+ 输入框里的“消息”会拼成假的“张三：消息”）"""
    import difflib
    from agent.grounding import normalize_text
    w, t = normalize_text(want), normalize_text(text)
    if not w or not t:
        return False
    if w in t:
        return True
    if len(w) < 6 or len(t) < len(w) - 1:
        return False
    best = 0.0
    for i in range(0, max(1, len(t) - len(w) + 1)):
        best = max(best, difflib.SequenceMatcher(None, w, t[i:i + len(w)]).ratio())
    return best >= min_ratio


@dataclass
class Postcondition:
    kind: str                     # window_exists / window_absent / active_window / text_present / text_absent / file_contains
    args: dict = field(default_factory=dict)
    required: bool = True         # False：不通过只记录警告，不判目标失败

    KINDS = ("window_exists", "window_absent", "active_window", "text_present", "text_absent", "file_contains")

    def __post_init__(self):
        if self.kind not in self.KINDS:
            raise ValueError(f"未知后置条件 {self.kind!r}，可选 {self.KINDS}")

    @property
    def needs_screen(self) -> bool:
        return self.kind in ("text_present", "text_absent")

    def describe(self) -> str:
        a = self.args
        return {"window_exists": f"存在标题含 {a.get('keywords')} 的窗口",
                "window_absent": f"不存在标题含 {a.get('keywords')} 的窗口",
                "active_window": f"前台窗口标题含 {a.get('keywords')}",
                "text_present": f"屏幕上出现「{a.get('text')}」",
                "text_absent": f"屏幕上没有「{a.get('text')}」",
                "file_contains": f"文件 {a.get('path')} 包含「{a.get('text')}」"}[self.kind]

    def to_dict(self):
        return {"kind": self.kind, "args": self.args, "required": self.required}

    def check(self, app_manager=None, texts_fn: Optional[Callable[[], List[str]]] = None) -> Tuple[bool, str]:
        k, a = self.kind, self.args
        if k in ("window_exists", "window_absent", "active_window"):
            if app_manager is None or getattr(app_manager, "windows", None) is None:
                return True, "无窗口信息，跳过"
            kws = a.get("keywords") or []
            if k == "active_window":
                title = app_manager.active_title()
                ok = any(_norm(x) in _norm(title) for x in kws if _norm(x))
                return ok, f"前台窗口「{title}」"
            wins = [getattr(w, "title", "") for w in app_manager.find_windows(kws)]
            ok = bool(wins) if k == "window_exists" else not wins
            return ok, f"匹配窗口 {wins}"
        if k in ("text_present", "text_absent"):
            if texts_fn is None:
                return True, "无感知器，跳过"
            try:
                texts = texts_fn()
            except Exception as e:
                return False, f"感知失败: {type(e).__name__}: {e}"
            hit = any(text_matches(a.get("text"), t) for t in texts)
            ok = hit if k == "text_present" else not hit
            return ok, ("找到" if hit else "没找到") + f"「{a.get('text')}」"
        if k == "file_contains":
            p = Path(os.path.expandvars(os.path.expanduser(str(a.get("path") or ""))))
            if not p.exists():
                return False, f"文件不存在: {p}"
            try:
                content = p.read_text(encoding="utf-8-sig", errors="replace")
            except OSError as e:
                return False, f"读文件失败: {e}"
            ok = _norm(a.get("text")) in _norm(content)
            return ok, f"文件内容{'包含' if ok else '不包含'}「{a.get('text')}」"
        return False, "未知条件"


def check_postconditions(conds: List[Postcondition], app_manager=None, texts_fn=None) -> dict:
    """依次检查；只要用到 text_* 就最多感知一次（结果缓存）"""
    cache = {}

    def texts():
        if "t" not in cache:
            cache["t"] = list(texts_fn()) if texts_fn else []
        return cache["t"]

    results, ok = [], True
    for c in conds or []:
        good, msg = c.check(app_manager, texts if texts_fn else None)
        results.append({**c.to_dict(), "ok": good, "message": msg, "desc": c.describe()})
        if not good and c.required:
            ok = False
    return {"ok": ok, "results": results}
