"""第4周 端到端闭环 —— Xvfb 虚拟屏幕上的真实执行（非 mock）

真实启动 apps/mock_chat.py（Tk 窗口），真实 mss 截图、真实 pyautogui 鼠标键盘事件、真实剪贴板中文输入，
由 GUIAgent 完成“给张三发送消息”。OCR 两种：
  - LayoutOCR：读取模拟聊天程序导出的控件文字与屏幕坐标（等价于 100% 准确的 OCR），隔离 OCR 误差，
    专门验证 规划 → 决策 → grounding → 坐标换算 → 真实点击/输入 → 检查器 这条链路；
  - 真实 EasyOCR（ocr_real，需 GUI_AGENT_REAL_OCR=1 且安装 easyocr）：对虚拟屏幕截图做真实文字识别。
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT, xvfb_available
from gui.xvfb_utils import tk_available

pytestmark = [pytest.mark.xvfb, pytest.mark.skipif(not tk_available(), reason="需要 tkinter")]

CHAT = REPO_ROOT / "apps" / "mock_chat.py"


class LayoutOCR:
    """把 mock_chat --layout 导出的控件坐标当作 OCR 结果（坐标为屏幕物理像素）"""

    def __init__(self, layout_path, scale=1.0):
        self.layout_path = Path(layout_path)
        self.scale = scale
        self.calls = 0

    def detect(self, image_path):
        self.calls += 1
        for _ in range(40):
            if self.layout_path.exists():
                try:
                    items = json.loads(self.layout_path.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.05)
        else:
            return []
        s = self.scale
        return [{**it, "bbox": [v * s for v in it["bbox"]]} for it in items]


@pytest.fixture
def desktop(tmp_path):
    if not xvfb_available():
        pytest.skip("无 Xvfb")
    import pyautogui
    procs = []
    layout, log = tmp_path / "layout.json", tmp_path / "chat.jsonl"

    def spawn(cmd, **kw):
        cmd = list(cmd) + ["--layout", str(layout), "--borderless", "--geometry", "900x600+120+90"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for _ in range(100):  # 等窗口画出来（导出布局文件）
            if layout.exists():
                break
            time.sleep(0.05)
        time.sleep(0.3)

    yield {"spawn": spawn, "layout": layout, "log": log, "size": pyautogui.size(), "tmp": tmp_path}
    for p in procs:
        p.terminate()
        p.wait(timeout=5)


def build(desktop, ocr, logical_size_fn=None):
    from agent.executor import ControllerExecutor
    from agent.gui_agent import GUIAgent
    from agent.planner import HybridPlanner, RulePlanner
    from control.app_manager import AppManager
    from control.mouse_keyboard import MouseKeyboardController
    from perception.screen_parser import ScreenPerceiver
    from perception.screenshot import ScreenCapture

    capture = ScreenCapture(save_dir=str(desktop["tmp"] / "shots"))
    ctl = MouseKeyboardController()
    apps = AppManager(spawn=desktop["spawn"], windows=None, controller=ctl, mock_chat_log=str(desktop["log"]),
                      search_paths=[])
    perceiver = ScreenPerceiver(capture=capture, ocr=ocr, logical_size_fn=logical_size_fn)
    agent = GUIAgent(perceiver, ControllerExecutor(ctl, app_manager=apps, settle_sec=0.3),
                     planner=HybridPlanner("rule", RulePlanner(open_wait=1)), act_mode="rule",
                     step_retries=2, retry_wait=0.5, trace_dir=str(desktop["tmp"] / "trace"), log=print)
    return agent, ctl


def sent(log):
    if not Path(log).exists():
        return []
    return [json.loads(x) for x in Path(log).read_text(encoding="utf-8").splitlines() if '"send"' in x]


@pytest.mark.functional
def test_xvfb_e2e_send_message(desktop):
    agent, _ = build(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给张三发送消息：今晚7点开会")
    msgs = sent(desktop["log"])
    assert r["executed_all"], [s.get("error") for s in r["steps"]]
    assert [(m["to"], m["text"]) for m in msgs] == [("张三", "今晚7点开会")]


@pytest.mark.functional
def test_xvfb_e2e_click_points_inside_widgets(desktop):
    """真实点击坐标必须落在目标控件内部（grounding 正确）"""
    agent, _ = build(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给李四发送消息：收到")
    layout = {it["text"]: it["bbox"] for it in json.loads(desktop["layout"].read_text(encoding="utf-8"))}
    click = next(s for s in r["steps"] if s["subtask"].startswith("点击联系人"))
    x, y = click["attempts"][-1]["detail"]["grounding"][0]["point"]
    x1, y1, x2, y2 = layout["李四"]
    assert x1 <= x <= x2 and y1 <= y <= y2
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("李四", "收到")]


@pytest.mark.functional
def test_xvfb_e2e_ascii_message(desktop):
    agent, _ = build(desktop, LayoutOCR(desktop["layout"]))
    agent.run("给王五发送消息：hello GUI agent")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("王五", "hello GUI agent")]


@pytest.mark.boundary
def test_xvfb_e2e_two_messages_in_a_row(desktop):
    agent, _ = build(desktop, LayoutOCR(desktop["layout"]))
    agent.run("给张三发送消息：第一条")
    plan_from = __import__("agent.planner", fromlist=["Plan"]).Plan
    agent.run("x", plan=plan_from(["点击联系人“李四”", "在“消息内容”框中输入“第二条”", "点击“发送”"], "test"))
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "第一条"), ("李四", "第二条")]


@pytest.mark.exception
def test_xvfb_e2e_unknown_contact_fails_cleanly(desktop):
    """联系人不存在：重试后失败、停止后续步骤，不会把消息发给别人"""
    agent, _ = build(desktop, LayoutOCR(desktop["layout"]))
    r = agent.run("给赵六发送消息：你好")
    assert r["success"] is False and sent(desktop["log"]) == []
    assert r["steps"][-1]["subtask"] == "点击联系人“赵六”" and len(r["steps"][-1]["attempts"]) == 3


@pytest.mark.multi_resolution
def test_xvfb_e2e_simulated_dpi_150(desktop):
    """模拟 150% 缩放：OCR 给出物理像素坐标（×1.5），截图尺寸也按 ×1.5 计算，换算后仍能点中"""
    w, h = desktop["size"]
    agent, _ = build(desktop, LayoutOCR(desktop["layout"], scale=1.5),
                     logical_size_fn=lambda: (round(w / 1.5), round(h / 1.5)))
    r = agent.run("给张三发送消息：缩放测试")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "缩放测试")], \
        [s.get("error") for s in r["steps"]]


@pytest.mark.functional
@pytest.mark.ocr_real
def test_xvfb_e2e_real_easyocr(desktop):
    """真实 EasyOCR 识别虚拟屏幕上的中文界面，完成发送消息"""
    pytest.importorskip("easyocr")
    from perception.ocr_detector import OCRDetector
    agent, _ = build(desktop, OCRDetector(engine="easyocr"))
    r = agent.run("给张三发送消息：真实OCR测试")
    assert [(m["to"], m["text"]) for m in sent(desktop["log"])] == [("张三", "真实OCR测试")], \
        [(s["subtask"], s.get("error")) for s in r["steps"]]
