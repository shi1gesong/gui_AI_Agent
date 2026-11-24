"""
应用启动 / 关闭 / 窗口管理（第4周新增，对应动作 open_app / close_app）

为什么不只靠“Win 键 + 输入程序名 + 回车”：
  - 开始菜单搜索结果随系统语言、索引状态变化，第一项不一定是要打开的程序；
  - 打开文件、打开默认浏览器这种需求，用系统 API（os.startfile）一步就能做到，而且结果可校验。
所以 open_app 先查内置程序表（浏览器、记事本、计算器、文件资源管理器、模拟聊天……），
是存在的文件/文件夹路径时直接用系统默认程序打开，都不是时才退回 Win 搜索。

关闭只关“标题匹配”的窗口（发送 WM_CLOSE，等同于点右上角 ×，程序有未保存内容时会弹窗询问），
不会 taskkill 强杀进程，不会影响标题不匹配的窗口。

所有系统调用（启动进程、枚举窗口、按键）都可以注入替身，单元测试不会真的打开程序。
"""
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_CHAT_PATH = REPO_ROOT / "apps" / "mock_chat.py"
MOCK_CHAT_TITLE = "GUI Agent 模拟聊天"
DEFAULT_HOME_URL = "https://www.bing.com"

# name: (别名, 窗口标题关键字, 启动方式)
#   启动方式 ("url", x) 用默认浏览器打开 x；("exe", [...]) 启动进程；("python", script) 用当前解释器运行脚本
APP_REGISTRY: Dict[str, dict] = {
    "浏览器": {"aliases": ["浏览器", "默认浏览器", "browser", "edge", "chrome", "firefox", "microsoft edge"],
            "titles": ["Edge", "Chrome", "Firefox", "浏览器", "Bing", "必应"],
            "launch": ("url", DEFAULT_HOME_URL)},
    "记事本": {"aliases": ["记事本", "notepad", "notepad.exe"], "titles": ["记事本", "Notepad"],
            "launch": ("exe", ["notepad.exe"])},
    "计算器": {"aliases": ["计算器", "calculator", "calc", "calc.exe"], "titles": ["计算器", "Calculator"],
            "launch": ("exe", ["calc.exe"])},
    "画图": {"aliases": ["画图", "paint", "mspaint"], "titles": ["画图", "Paint"], "launch": ("exe", ["mspaint.exe"])},
    "文件资源管理器": {"aliases": ["文件资源管理器", "资源管理器", "文件管理器", "此电脑", "explorer"],
                "titles": ["文件资源管理器", "File Explorer", "此电脑", "This PC"], "launch": ("exe", ["explorer.exe"])},
    "命令提示符": {"aliases": ["命令提示符", "cmd"], "titles": ["命令提示符", "Command Prompt", "cmd.exe"],
              "launch": ("exe", ["cmd.exe", "/c", "start", "cmd.exe"])},
    "模拟聊天": {"aliases": ["模拟聊天", "聊天应用", "聊天程序", "聊天软件", "mock chat", "mock_chat"],
             "titles": [MOCK_CHAT_TITLE], "launch": ("python", str(MOCK_CHAT_PATH))},
    "微信": {"aliases": ["微信", "wechat"], "titles": ["微信", "WeChat"], "launch": ("search", "微信")},
}


# 命令行 / 终端窗口的标题关键字：Agent 自己在这些窗口里打印日志，感知时要排除
CONSOLE_TITLES = ["Anaconda Prompt", "Anaconda Powershell", "命令提示符", "Command Prompt", "cmd.exe", "PowerShell",
                  "Windows Terminal", "终端", "Terminal", "python.exe", "pytest", "Miniconda"]


def _norm(s) -> str:
    return "".join(str(s or "").split()).lower()


def _default_spawn(cmd, **kw):
    return subprocess.Popen(cmd, **kw)


def _default_startfile(target: str):
    if hasattr(os, "startfile"):
        os.startfile(target)  # noqa: S606  Windows：用默认程序打开文件/网址
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


class PyGetWindowProvider:
    """真实窗口枚举（Windows 上 pyautogui 自带 pygetwindow）"""

    def all_windows(self):
        import pygetwindow as gw
        return [w for w in gw.getAllWindows() if (w.title or "").strip()]

    def active_window(self):
        import pygetwindow as gw
        return gw.getActiveWindow()


class AppManager:
    def __init__(self, spawn: Callable = None, startfile: Callable = None, windows=None,
                 controller=None, sleep: Callable = time.sleep, registry: Dict[str, dict] = None,
                 search_paths: Optional[List[Path]] = None, mock_chat_log: Optional[str] = None,
                 open_timeout: float = 10.0, close_timeout: float = 5.0, poll: float = 0.25,
                 platform: str = sys.platform):
        """
        spawn(cmd: list) 启动进程；startfile(path_or_url) 用默认程序打开；windows 提供 all_windows()/active_window()；
        controller 用于 Win 搜索退路（press_key/type_text）。
        """
        self.spawn = spawn or _default_spawn
        self.startfile = startfile or _default_startfile
        self.windows = windows
        self.controller = controller
        self.sleep = sleep
        self.registry = registry or APP_REGISTRY
        home = Path.home()
        self.search_paths = search_paths if search_paths is not None else [
            Path.cwd(), home / "Desktop", home / "桌面", home / "Documents", home / "Downloads"]
        self.mock_chat_log = mock_chat_log
        self.open_timeout, self.close_timeout, self.poll = open_timeout, close_timeout, poll
        self.platform = platform

    # ---------- 名称解析 ----------
    def resolve_app(self, name: str) -> Optional[str]:
        n = _norm(name).replace("默认", "")
        if not n or _looks_like_path(str(name)):
            return None
        for key, spec in self.registry.items():
            if n == _norm(key) or any(n == _norm(a) for a in spec["aliases"]):
                return key
        for key, spec in self.registry.items():  # “打开edge浏览器”“谷歌浏览器”
            if any(_norm(a) in n for a in [key] + spec["aliases"] if len(_norm(a)) >= 2):
                return key
        return None

    def resolve_file(self, name: str) -> Optional[Path]:
        raw = str(name or "").strip().strip("“”\"'")
        if not raw:
            return None
        p = Path(os.path.expandvars(os.path.expanduser(raw)))
        if p.is_absolute():
            return p if p.exists() else None
        for base in self.search_paths:
            cand = Path(base) / raw
            if cand.exists():
                return cand
        return None

    def title_keywords(self, name: str) -> List[str]:
        key = self.resolve_app(name)
        if key:
            return list(self.registry[key]["titles"])
        f = self.resolve_file(name)
        if f is not None:
            return [f.name, f.stem]
        return [str(name).strip()]

    # ---------- 窗口查询 ----------
    def list_windows(self):
        if self.windows is None:
            return []
        try:
            return list(self.windows.all_windows())
        except Exception:
            return []

    def find_windows(self, keywords: List[str]):
        keys = [_norm(k) for k in keywords if _norm(k)]
        return [w for w in self.list_windows() if any(k in _norm(getattr(w, "title", "")) for k in keys)]

    def active_title(self) -> str:
        if self.windows is None:
            return ""
        try:
            w = self.windows.active_window()
            return getattr(w, "title", "") or ""
        except Exception:
            return ""

    @staticmethod
    def window_rect(win):
        try:
            l, t = int(win.left), int(win.top)
            r, b = l + int(win.width), t + int(win.height)
        except Exception:
            return None
        if r <= l or b <= t or (l <= -30000 and t <= -30000):  # 最小化窗口坐标为 -32000
            return None
        return [l, t, r, b]

    @staticmethod
    def is_console(title: str) -> bool:
        t = _norm(title)
        return any(_norm(k) in t for k in CONSOLE_TITLES)

    def screen_regions(self) -> dict:
        """给感知模块用：include = 前台窗口（是命令行时不限制），exclude = 所有命令行窗口"""
        if self.windows is None:
            return {}
        include = None
        try:
            act = self.windows.active_window()
        except Exception:
            act = None
        if act is not None and not self.is_console(getattr(act, "title", "")):
            include = self.window_rect(act)
        exclude = [r for r in (self.window_rect(w) for w in self.list_windows()
                               if self.is_console(getattr(w, "title", ""))) if r]
        return {"include": include, "exclude": exclude}

    def wait_for_window(self, keywords: List[str], timeout: float, exclude_handles=()):
        """等待出现标题匹配的窗口；exclude_handles 为启动前已存在的窗口（优先返回新窗口）"""
        deadline = timeout
        waited = 0.0
        while True:
            wins = self.find_windows(keywords)
            new = [w for w in wins if _handle(w) not in exclude_handles]
            if new or (wins and waited >= min(2.0, timeout)):
                return (new or wins)[0]
            if waited >= deadline:
                return wins[0] if wins else None
            self.sleep(self.poll)
            waited += self.poll

    @staticmethod
    def _activate(win):
        for attr in ("restore", "activate"):
            try:
                if attr == "restore" and not getattr(win, "isMinimized", False):
                    continue
                getattr(win, attr)()
            except Exception:
                pass

    # ---------- open / close ----------
    def open(self, name: str, wait: bool = True) -> dict:
        """打开程序或文件。返回 {"ok", "method", "target", "window"}，失败时 ok=False 并带 error"""
        name = str(name or "").strip().strip("“”\"'")
        if not name:
            return {"ok": False, "error": "要打开的程序名为空"}
        if is_url(name):  # 第6周：打开网址 = 默认浏览器打开
            return self._open_url(name, wait)
        f = self.resolve_file(name) if _looks_like_path(name) else None
        key = None if f is not None else self.resolve_app(name)
        before = {_handle(w) for w in self.list_windows()}
        try:
            if f is not None:
                self.startfile(str(f))
                method, keywords = "file", [f.name, f.stem]
            elif key:
                kind, arg = self.registry[key]["launch"]
                if kind == "url":
                    self.startfile(arg)
                elif kind == "exe":
                    self.spawn(list(arg))
                elif kind == "python":
                    cmd = [sys.executable, arg]
                    if self.mock_chat_log:
                        cmd += ["--log", self.mock_chat_log]
                    cmd += list(getattr(self, "mock_chat_args", []) or [])  # 第6周：--load-delay / --popup-on-select
                    self.spawn(cmd)
                elif kind == "search":
                    self._win_search(arg)
                method, keywords = f"{kind}:{key}", self.registry[key]["titles"]
            elif _looks_like_path(name):
                return {"ok": False, "error": f"文件不存在: {name}"}
            else:
                self._win_search(name)
                method, keywords = "search", [name]
        except Exception as e:
            return {"ok": False, "error": f"启动「{name}」失败: {type(e).__name__}: {e}"}

        result = {"ok": True, "method": method, "target": name}
        if wait and self.windows is not None:
            win = self.wait_for_window(keywords, self.open_timeout, exclude_handles=before)
            if win is None:
                return {**result, "ok": False, "error": f"{self.open_timeout:g} 秒内没有出现标题含 {keywords} 的窗口"}
            self._activate(win)
            result["window"] = getattr(win, "title", "")
        return result

    def _open_url(self, url: str, wait: bool = True) -> dict:
        url = url if url.lower().startswith(("http://", "https://")) else "https://" + url
        before = {_handle(w) for w in self.list_windows()}
        try:
            self.startfile(url)
        except Exception as e:
            return {"ok": False, "error": f"打开网址「{url}」失败: {type(e).__name__}: {e}"}
        result = {"ok": True, "method": "url", "target": url}
        if wait and self.windows is not None:
            win = self.wait_for_window(self.registry["浏览器"]["titles"], self.open_timeout, exclude_handles=before)
            if win is None:
                return {**result, "ok": False, "error": f"{self.open_timeout:g} 秒内没有出现浏览器窗口"}
            self._activate(win)
            result["window"] = getattr(win, "title", "")
        return result

    def activate(self, name: str) -> dict:
        """第6周 switch_app：把已打开的窗口切到前台（不重新启动程序）"""
        name = str(name or "").strip().strip("“”\"'")
        if self.windows is None:
            return {"ok": False, "error": "没有窗口管理器，无法切换窗口"}
        keywords = self.title_keywords(name)
        wins = self.find_windows(keywords)
        if not wins:
            return {"ok": False, "error": f"没有找到标题含 {keywords} 的窗口，无法切换"}
        w = wins[0]
        self._activate(w)
        try:
            w.activate()
        except Exception as e:
            return {"ok": False, "error": f"激活窗口「{getattr(w, 'title', '')}」失败: {e}"}
        return {"ok": True, "method": "activate", "window": getattr(w, "title", "")}

    def close(self, name: str) -> dict:
        """关闭标题匹配的所有窗口（WM_CLOSE）。name 为“当前窗口”时关闭前台窗口"""
        name = str(name or "").strip()
        if self.windows is None:
            if self.controller is None:
                return {"ok": False, "error": "没有窗口管理器，无法关闭窗口"}
            self.controller.hotkey("alt", "f4")
            return {"ok": True, "method": "alt+f4", "closed": ["(前台窗口)"]}
        if name in ("", "当前", "当前窗口", "窗口"):
            try:
                wins = [self.windows.active_window()]
            except Exception:
                wins = []
            keywords = []
        else:
            keywords = self.title_keywords(name)
            wins = self.find_windows(keywords)
        wins = [w for w in wins if w is not None]
        if not wins:
            return {"ok": False, "error": f"没有找到标题含 {keywords or [name]} 的窗口"}
        closed = []
        for w in wins:
            try:
                w.close()
                closed.append(getattr(w, "title", ""))
            except Exception as e:
                return {"ok": False, "error": f"关闭窗口「{getattr(w, 'title', '')}」失败: {e}", "closed": closed}
        # 等窗口真正消失（程序弹出“是否保存”对话框时不会消失）
        waited = 0.0
        while keywords and self.find_windows(keywords) and waited < self.close_timeout:
            self.sleep(self.poll)
            waited += self.poll
        remaining = [getattr(w, "title", "") for w in self.find_windows(keywords)] if keywords else []
        if remaining:
            return {"ok": False, "error": f"窗口仍未关闭（可能弹出了保存确认框）: {remaining}", "closed": closed}
        return {"ok": True, "method": "wm_close", "closed": closed}

    def _win_search(self, name: str):
        if self.controller is None:
            raise RuntimeError(f"不认识程序「{name}」，且没有 controller 可以用开始菜单搜索")
        self.controller.press_key("win")
        self.sleep(0.8)
        self.controller.type_text(name)
        self.sleep(0.8)
        self.controller.press_key("enter")


def _handle(w):
    return getattr(w, "_hWnd", None) or getattr(w, "handle", None) or id(w)


def is_url(name: str) -> bool:
    import re
    return bool(re.match(r"^(?:https?://|www\.)\S+$", str(name or "").strip(), re.IGNORECASE))


def _looks_like_path(name: str) -> bool:
    import re
    name = str(name or "").strip()
    return bool("\\" in name or "/" in name or re.search(r"\.[A-Za-z0-9]{1,5}$", name)) and not re.fullmatch(
        r"[A-Za-z]+\.exe", name, re.IGNORECASE)
