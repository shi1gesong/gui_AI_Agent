"""
pytest 全局配置

设计要点：
1. 大部分用例无需真实屏幕/LLM：pyautogui、mss、EasyOCR、PaddleOCR、Ollama、transformers 全部用替身(mock)，
   可在任何机器（含无图形界面的 CI）上运行。
2. 若本机有 Xvfb（Linux 虚拟屏幕），会自动起一个虚拟显示器，跑 `xvfb` 标记的“真实 GUI”用例：
   真实 mss 截图、真实 pyautogui 产生 X11 鼠标键盘事件，由事件记录窗口验证。
3. `real_device`（Windows 真机）、`llm`（真实 Ollama 服务）、`ocr_real`（真实 OCR 模型）默认跳过，
   需设置环境变量后显式运行，见 tests/README.md。
4. 每个用例必须且只能属于一个测试类别：functional / boundary / exception / multi_resolution，
   收集阶段自动检查；运行结束后按“类别 × 结果”输出统计表，测试报告中的数字直接来自这里。
"""
import atexit
import os
import shutil
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
# agent/base_agent.py 使用了 `from llm_interface import ...`（非包内导入），需要把 agent/ 加入 sys.path
for p in (REPO_ROOT, REPO_ROOT / "agent", TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CATEGORIES = ("functional", "boundary", "exception", "multi_resolution")
ENV_MARKERS = ("xvfb", "real_device", "llm", "ocr_real")


# --------------------------------------------------------------------------- #
# 会话级虚拟屏幕：必须在 import pyautogui 之前设置 DISPLAY
# --------------------------------------------------------------------------- #
XVFB_DISPLAY = None
XVFB_RES = os.environ.get("XVFB_RES", "1920x1080")


def _start_session_xvfb():
    global XVFB_DISPLAY
    if sys.platform.startswith("win") or os.environ.get("GUI_AGENT_NO_XVFB"):
        return
    if os.environ.get("DISPLAY") or not shutil.which("Xvfb"):
        return
    disp = ":%d" % (150 + os.getpid() % 100)
    proc = subprocess.Popen(
        ["Xvfb", disp, "-screen", "0", f"{XVFB_RES}x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sock = Path(f"/tmp/.X11-unix/X{disp[1:]}")
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)
    atexit.register(proc.terminate)
    xauth = Path.home() / ".Xauthority"
    if not xauth.exists():  # python-xlib 在缺少该文件时直接报错
        xauth.touch()
    os.environ["DISPLAY"] = disp
    XVFB_DISPLAY = disp


_start_session_xvfb()

# --------------------------------------------------------------------------- #
# 可选依赖：导入失败时注入最小桩模块，保证被测模块可以 import
# --------------------------------------------------------------------------- #
try:
    import pyautogui  # noqa: F401
    REAL_PYAUTOGUI = True
except Exception:  # 未安装 / 无显示器
    stub = types.ModuleType("pyautogui")
    stub.PAUSE, stub.FAILSAFE = 0.1, True

    class FailSafeException(Exception):
        pass

    stub.FailSafeException = FailSafeException
    stub.size = lambda: (1920, 1080)
    sys.modules["pyautogui"] = stub
    REAL_PYAUTOGUI = False

try:
    import cv2  # noqa: F401  ocr_detector 顶部 import 了 cv2（实际未使用）
except Exception:
    sys.modules["cv2"] = types.ModuleType("cv2")

try:
    import langchain_core  # noqa: F401
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False


def xvfb_available() -> bool:
    return XVFB_DISPLAY is not None and REAL_PYAUTOGUI


# --------------------------------------------------------------------------- #
# 标记与收集检查
# --------------------------------------------------------------------------- #
def pytest_configure(config):
    for name, desc in {
        "functional": "功能测试：正常输入下的核心功能是否正确",
        "boundary": "边界测试：极值、空值、临界条件",
        "exception": "异常测试：错误输入、依赖失败时的行为",
        "multi_resolution": "多分辨率测试：不同屏幕尺寸/DPI 下的适配",
        "xvfb": "需要 Xvfb 虚拟屏幕的真实 GUI 测试（Linux 自动启用）",
        "real_device": "需要 Windows 真机桌面（设置 GUI_AGENT_REAL_DEVICE=1）",
        "llm": "需要真实 Ollama 服务（设置 OLLAMA_BASE_URL）",
        "ocr_real": "需要真实安装 EasyOCR/PaddleOCR 及模型（设置 GUI_AGENT_REAL_OCR=1）",
    }.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_collection_modifyitems(config, items):
    bad = []
    for item in items:
        cats = [c for c in CATEGORIES if item.get_closest_marker(c)]
        if len(cats) != 1:
            bad.append(f"{item.nodeid} -> {cats}")
        if item.get_closest_marker("xvfb") and not xvfb_available():
            item.add_marker(pytest.mark.skip(reason="无 Xvfb 虚拟屏幕或 pyautogui 不可用"))
        if item.get_closest_marker("real_device") and not os.environ.get("GUI_AGENT_REAL_DEVICE"):
            item.add_marker(pytest.mark.skip(reason="真机用例：设置 GUI_AGENT_REAL_DEVICE=1 后在 Windows 桌面运行"))
        if item.get_closest_marker("llm") and not os.environ.get("OLLAMA_BASE_URL"):
            item.add_marker(pytest.mark.skip(reason="真实 LLM 用例：设置 OLLAMA_BASE_URL 后运行"))
        if item.get_closest_marker("ocr_real") and not os.environ.get("GUI_AGENT_REAL_OCR"):
            item.add_marker(pytest.mark.skip(reason="真实 OCR 用例：设置 GUI_AGENT_REAL_OCR=1 后运行"))
    if bad:
        raise pytest.UsageError("以下用例没有恰好一个类别标记：\n" + "\n".join(bad))


# --------------------------------------------------------------------------- #
# 结束时输出“类别 × 结果”统计
# --------------------------------------------------------------------------- #
_RESULTS = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    cat = next((c for c in CATEGORIES if item.get_closest_marker(c)), "?")
    env = next((m for m in ENV_MARKERS if item.get_closest_marker(m)), "mock")
    module = item.nodeid.split("::")[0].split("/")[-1]
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        if hasattr(rep, "wasxfail"):
            status = "xpassed" if rep.passed else "xfailed"
        else:
            status = rep.outcome
        _RESULTS[item.nodeid] = (module, cat, env, status)


def pytest_terminal_summary(terminalreporter):
    if not _RESULTS:
        return
    tr = terminalreporter
    statuses = ["passed", "failed", "xfailed", "xpassed", "skipped"]
    tr.section("用例统计（类别 × 结果）")
    by_cat = Counter((c, s) for _, c, _, s in _RESULTS.values())
    tr.write_line(f"{'category':<20}" + "".join(f"{s:>9}" for s in statuses) + f"{'total':>9}")
    for c in CATEGORIES:
        row = [by_cat[(c, s)] for s in statuses]
        tr.write_line(f"{c:<20}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    tr.section("用例统计（模块 × 结果）")
    by_mod = Counter((m, s) for m, _, _, s in _RESULTS.values())
    for m in sorted({m for m, *_ in _RESULTS.values()}):
        row = [by_mod[(m, s)] for s in statuses]
        tr.write_line(f"{m:<32}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    tr.section("用例统计（运行环境 × 结果）")
    by_env = Counter((e, s) for _, _, e, s in _RESULTS.values())
    for e in ("mock",) + ENV_MARKERS:
        row = [by_env[(e, s)] for s in statuses]
        if sum(row):
            tr.write_line(f"{e:<20}" + "".join(f"{n:>9}" for n in row) + f"{sum(row):>9}")
    out = os.environ.get("GUI_AGENT_RESULTS_CSV")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("nodeid,module,category,env,status\n")
            for nodeid, (m, c, e, s) in sorted(_RESULTS.items()):
                f.write(f"\"{nodeid}\",{m},{c},{e},{s}\n")


# --------------------------------------------------------------------------- #
# 通用 fixture
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_gui(monkeypatch):
    """把 control.mouse_keyboard 里的 pyautogui 替换为记录调用的假对象，保证 mock 用例绝不会真的动鼠标。"""
    from fakes import FakePyAutoGUI
    import control.mouse_keyboard as mk

    fake = FakePyAutoGUI()
    monkeypatch.setattr(mk, "pyautogui", fake)
    monkeypatch.setattr(mk, "CLIPBOARD_SETTLE_SEC", 0)
    return fake


@pytest.fixture
def fake_clipboard(monkeypatch):
    from fakes import FakeClipboard

    clip = FakeClipboard()
    monkeypatch.setitem(sys.modules, "pyperclip", clip)
    return clip
