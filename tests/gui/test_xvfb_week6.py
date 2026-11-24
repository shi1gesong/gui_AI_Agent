"""第6周 v2.0 在 Xvfb 虚拟屏幕上的真实执行（非 mock）

真实 Tk 窗口（apps/mock_chat.py）+ 真实截图 + 真实鼠标键盘：
  - 慢加载（--load-delay）：v1 固定重试耗尽而失败，v2 按错误类型退避重试后成功；
  - 弹窗（--popup-on-select）：python-xlib 读取真实 X 窗口，v2 检测并关闭“系统通知”弹窗、重新聚焦后完成任务；
  - 真实截图上的控件检测（输入框）与“标签 → 输入框”关联；
  - 动作效果校验（截图前后对比）在真实帧上判断“输入生效/没有效果”；
  - 真实 EasyOCR（ocr_real）：v2 只识别前台窗口 + 帧缓存，比 v1 全屏识别更快。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT, xvfb_available
from gui.test_xvfb_e2e import LayoutOCR, sent
from gui.xvfb_utils import tk_available

pytestmark = [pytest.mark.xvfb, pytest.mark.skipif(not tk_available(), reason="需要 tkinter")]

CHAT = REPO_ROOT / "apps" / "mock_chat.py"
GEOM = (120, 90, 900, 600)


@pytest.fixture
def desktop(tmp_path):
    if not xvfb_available():
        pytest.skip("无 Xvfb")
    import pyautogui
    procs = []
    layout, log = tmp_path / "layout.json", tmp_path / "chat.jsonl"
    extra = []

    def spawn(cmd, **kw):
        x, y, w, h = GEOM
        cmd = list(cmd) + ["--layout", str(layout), "--borderless", "--geometry", f"{w}x{h}+{x}+{y}"] + extra
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for _ in range(100):
            if layout.exists():
                break
            time.sleep(0.05)
        time.sleep(0.3)

    yield {"spawn": spawn, "extra": extra, "layout": layout, "log": log, "size": pyautogui.size(),
           "tmp": tmp_path, "procs": procs}
    for p in procs:
        p.terminate()
        p.wait(timeout=5)


def _parts(desktop, ocr, windows=None):
    from agent.executor import ControllerExecutor
    from control.app_manager import AppManager
    from control.mouse_keyboard import MouseKeyboardController
    from perception.perceiver_v2 import PerceiverV2
    from perception.screen_parser import ScreenPerceiver
    from perception.screenshot import ScreenCapture
    capture = ScreenCapture(save_dir=str(desktop["tmp"] / "shots"))
    ctl = MouseKeyboardController()
    apps = AppManager(spawn=desktop["spawn"], windows=windows, controller=ctl, mock_chat_log=str(desktop["log"]),
                      search_paths=[], sleep=time.sleep)
    v1p = ScreenPerceiver(capture=capture, ocr=ocr)
    # LayoutOCR 给的是屏幕坐标：关闭 ROI 裁剪/增量识别，其余 v2 感知逻辑照常
    v2p = PerceiverV2(capture=capture, ocr=ocr, roi=False, cache=False, incremental=False, widgets=False)
    return ctl, apps, v1p, v2p, ControllerExecutor(ctl, app_manager=apps, settle_sec=0.3)


def build_v1(desktop, ocr):
    from agent.gui_agent import GUIAgent
    from agent.planner import HybridPlanner, RulePlanner
    ctl, apps, v1p, _, ex = _parts(desktop, ocr)
    return GUIAgent(v1p, ex, planner=HybridPlanner("rule", RulePlanner(open_wait=1)), act_mode="rule",
                    step_retries=2, retry_wait=0.5, trace_dir=str(desktop["tmp"] / "trace_v1"), log=print)


def build_v2(desktop, ocr, windows=None, **kw):
    from agent.agent_v2 import GUIAgentV2
    from agent.decomposer import Decomposer
    from agent.planner import RulePlanner
    ctl, apps, _, v2p, ex = _parts(desktop, ocr, windows)
    return GUIAgentV2(v2p, ex, decomposer=Decomposer(RulePlanner(open_wait=1)), act_mode="rule", app_manager=apps,
                      trace_dir=str(desktop["tmp"] / "trace_v2"), log=print, settle_sec=0.3, post_wait=0.5,
                      late_window_sec=3.0, **kw)


def grab():
    import mss
    from PIL import Image
    with mss.mss() as s:
        shot = s.grab(s.monitors[1])
        return Image.frombytes("RGB", shot.size, shot.rgb)


def layout_items(path):
    for _ in range(40):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
    return []


# ---------------- 慢加载：v1 vs v2 ----------------
@pytest.mark.functional
def test_v2_recovers_from_slow_loading(desktop):
    desktop["extra"] += ["--load-delay", "4"]
    agent = build_v2(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给张三发送消息：慢加载也能发出")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "慢加载也能发出")], \
        [(s["subtask"], s.get("error")) for s in r["steps"]]
    assert r["success"] and r["robustness"]["retries"] >= 1


@pytest.mark.functional
def test_v1_fails_on_same_slow_loading(desktop):
    """对照组：v1 固定 2 次重试、0.5s 间隔，联系人列表 4s 后才出现 → 失败"""
    desktop["extra"] += ["--load-delay", "4"]
    agent = build_v1(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给张三发送消息：v1对照")
    assert r["success"] is False and sent(desktop["log"]) == []


@pytest.mark.boundary
def test_v2_complex_two_goals_real_window(desktop, xwin):
    """复合任务拆成两个目标，第二个目标复用已打开的窗口（切换而非重复启动）"""
    agent = build_v2(desktop, LayoutOCR(desktop["layout"]), windows=xwin)
    r = agent.run("先给张三发送消息：第一条，然后给李四发送消息：第二条")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "第一条"), ("李四", "第二条")], \
        [(s["subtask"], s.get("error")) for s in r["steps"]]
    assert len(desktop["procs"]) == 1


@pytest.mark.boundary
def test_v2_postcondition_checked_on_real_screen(desktop):
    agent = build_v2(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给王五发送消息：检查后置条件")
    checks = [c for g in r["goals"] for c in g.get("post", {}).get("results", [])]
    assert r["success"] and checks and all(c["ok"] for c in checks)


@pytest.mark.exception
def test_v2_unknown_contact_fails_honestly(desktop):
    agent = build_v2(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给赵六发送消息：你好")
    assert r["success"] is False and sent(desktop["log"]) == []


# ---------------- 弹窗：真实 X 窗口 ----------------
@pytest.fixture
def xwin():
    pytest.importorskip("Xlib")
    from gui.x11_windows import XlibWindowProvider
    return XlibWindowProvider()


@pytest.mark.functional
def test_xlib_provider_lists_real_windows(desktop, xwin):
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    wins = [w for w in xwin.all_windows() if "模拟聊天" in w.title]
    assert wins and (wins[0].left, wins[0].top, wins[0].width, wins[0].height) == GEOM


@pytest.mark.functional
def test_xlib_popup_detected_and_closed(desktop, xwin):
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"]), "--popup-after", "0.5"])
    time.sleep(1.5)
    assert any(w.title == "系统通知" for w in xwin.all_windows())
    from agent.recovery import is_popup_title
    pop = next(w for w in xwin.all_windows() if is_popup_title(w.title))
    pop.close()
    time.sleep(0.5)
    assert not any(w.title == "系统通知" for w in xwin.all_windows())


@pytest.mark.functional
def test_xlib_activate_sets_foreground(desktop, xwin):
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    w = next(w for w in xwin.all_windows() if "模拟聊天" in w.title)
    w.activate()
    time.sleep(0.3)
    a = xwin.active_window()
    assert a is not None and "模拟聊天" in a.title


@pytest.mark.functional
def test_v2_dismisses_real_popup_and_finishes(desktop, xwin):
    """选中联系人后弹出模态“系统通知”（grab_set 挡住输入）：v2 关闭弹窗、重新聚焦后完成发送"""
    desktop["extra"] += ["--popup-on-select"]
    agent = build_v2(desktop, LayoutOCR(desktop["layout"]), windows=xwin)
    r = agent.run("给张三发送消息：弹窗之后")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "弹窗之后")], \
        [(s["subtask"], s.get("error")) for s in r["steps"]]
    assert not any(w.title == "系统通知" for w in xwin.all_windows())


# ---------------- 控件检测 / 效果校验（真实截图） ----------------
@pytest.mark.functional
def test_widget_detection_finds_message_input(desktop):
    from perception.ui_detector import associate_labels, detect_widgets
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    img = grab()
    widgets = detect_widgets(img)
    assert any(w["kind"] == "input" for w in widgets)
    items = layout_items(desktop["layout"])
    label = next(it for it in items if it["text"] == "消息内容")
    els = associate_labels([dict(label)], widgets)
    ib = els[0].get("input_bbox") or els[0].get("widget_bbox")
    assert ib is not None
    x1, y1, x2, y2 = ib
    gx, gy, gw, gh = GEOM
    assert gx <= x1 < x2 <= gx + gw and gy <= y1 < y2 <= gy + gh


@pytest.mark.functional
def test_effect_verifier_sees_real_typing(desktop):
    from agent.verifier import EffectVerifier
    from control.mouse_keyboard import MouseKeyboardController
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    items = layout_items(desktop["layout"])
    box = next(it for it in items if it["text"] == "张三")["bbox"]
    ctl = MouseKeyboardController()
    before = grab()
    ctl.click((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    time.sleep(0.4)
    after = grab()
    r = EffectVerifier().verify({"action_type": "click"}, before, after)
    assert r["ok"] and r["checked"] and r["changed_bbox"]


@pytest.mark.boundary
def test_effect_verifier_flags_click_on_empty_area(desktop):
    from agent.verifier import EffectVerifier
    from control.mouse_keyboard import MouseKeyboardController
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    ctl = MouseKeyboardController()
    w, h = desktop["size"]
    ctl.click(w - 20, h - 20)          # 窗口外的空白桌面
    time.sleep(0.2)
    before = grab()
    ctl.click(w - 20, h - 20)
    time.sleep(0.4)
    r = EffectVerifier().verify({"action_type": "click"}, before, grab())
    assert r["ok"] is False and "没有变化" in r["reason"]


@pytest.mark.functional
def test_frame_cache_hits_on_static_screen(desktop):
    from perception.perceiver_v2 import PerceiverV2
    from perception.screenshot import ScreenCapture
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    ocr = LayoutOCR(desktop["layout"])
    p = PerceiverV2(capture=ScreenCapture(save_dir=str(desktop["tmp"] / "s")), ocr=ocr, roi=False, widgets=False)
    p.perceive()
    p.perceive()
    assert p.stats["hit"] >= 1 and ocr.calls == 1


# ---------------- 真实 EasyOCR：v1 全屏 vs v2 前台窗口 ----------------
@pytest.mark.functional
@pytest.mark.ocr_real
def test_real_easyocr_roi_faster_than_full_screen(desktop):
    pytest.importorskip("easyocr")
    from perception.ocr_detector import OCRDetector
    from perception.perceiver_v2 import PerceiverV2
    from perception.screen_parser import ScreenPerceiver
    from perception.screenshot import ScreenCapture
    desktop["spawn"]([sys.executable, str(CHAT), "--log", str(desktop["log"])])
    ocr = OCRDetector(engine="easyocr")
    cap = ScreenCapture(save_dir=str(desktop["tmp"] / "s"))
    x, y, w, h = GEOM
    region = lambda: {"include": [x, y, x + w, y + h], "exclude": []}   # noqa: E731
    v1 = ScreenPerceiver(capture=cap, ocr=ocr)
    v2 = PerceiverV2(capture=cap, ocr=ocr, region_fn=region)
    v1.perceive()                       # 预热模型
    t = time.perf_counter()
    o1 = v1.perceive()
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    o2 = v2.perceive()
    t2 = time.perf_counter() - t
    t = time.perf_counter()
    v2.perceive()                       # 画面不变：缓存命中
    t3 = time.perf_counter() - t
    print(f"\nv1 全屏 {t1:.2f}s  v2 前台窗口 {t2:.2f}s  v2 缓存 {t3:.3f}s")
    texts2 = {e["text"] for e in o2.elements}
    assert "张三" in texts2 and "发送" in texts2
    assert t2 < t1 and t3 < 0.2 * t1
    assert len(o2.elements) <= len(o1.elements)
