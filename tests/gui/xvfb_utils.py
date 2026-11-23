"""Xvfb 真实 GUI 测试的辅助工具：独立虚拟屏幕、事件记录窗口。"""
import contextlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_counter = itertools.count()


def tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False


@contextlib.contextmanager
def virtual_display(resolution: str):
    """临时起一个指定分辨率的 Xvfb，并把 DISPLAY 指向它（仅影响本进程之后新建的连接）。"""
    if not shutil.which("Xvfb"):
        raise RuntimeError("Xvfb not installed")
    disp = ":%d" % (300 + (os.getpid() % 50) * 10 + next(_counter) % 10)
    proc = subprocess.Popen(["Xvfb", disp, "-screen", "0", f"{resolution}x24", "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sock = Path(f"/tmp/.X11-unix/X{disp[1:]}")
    for _ in range(100):
        if sock.exists():
            break
        time.sleep(0.05)
    old = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = disp
    try:
        yield disp
    finally:
        if old is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old
        proc.terminate()
        proc.wait(timeout=5)


def fill_screen(color: str, width: int, height: int, display: str):
    """在指定显示器上铺一个纯色窗口，返回进程（调用方负责 terminate）。"""
    code = (
        "import tkinter as tk,sys;r=tk.Tk();r.overrideredirect(True);"
        f"r.geometry('{width}x{height}+0+0');r.configure(bg='{color}');"
        "r.after(200,lambda:print('ready',flush=True));r.mainloop()"
    )
    env = dict(os.environ, DISPLAY=display)
    proc = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    time.sleep(0.3)
    return proc


class EventLogger:
    """启动 event_logger_app.py 全屏窗口，读取其记录的事件。"""

    def __init__(self, tmp_path, width, height):
        self.log = Path(tmp_path) / "events.jsonl"
        self.log.touch()
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "event_logger_app.py"), str(self.log), str(width), str(height)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.wait_for(lambda evs: any(e["ev"] == "ready" for e in evs), timeout=10)

    def events(self):
        lines = self.log.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            evs = self.events()
            if predicate(evs):
                return evs
            time.sleep(0.05)
        return self.events()

    def of(self, kind):
        return [e for e in self.events() if e["ev"] == kind]

    def last_text(self, timeout=3.0):
        evs = self.wait_for(lambda es: any(e["ev"] == "text" for e in es), timeout)
        texts = [e["value"] for e in evs if e["ev"] == "text"]
        return texts[-1] if texts else None

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)
