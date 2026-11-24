"""
模拟桌面（第4周测试辅助）：在内存里模拟浏览器、记事本、计算器、模拟聊天程序的窗口与界面元素，
同时充当 controller（接收点击/输入/按键）、窗口提供者、程序启动器、截图与 OCR。

用它可以在任何机器上把 5 个基础任务从“指令”一路跑到“检查器判定”，覆盖 规划 → 感知 → 决策 → grounding →
执行 → 反馈 的全部代码路径；dpi 参数模拟 Windows 显示缩放（OCR 坐标 = 逻辑坐标 × dpi）。
"""
import json
from pathlib import Path

from PIL import Image

from control.app_manager import MOCK_CHAT_TITLE, AppManager
from fakes import FakeWindowProvider

CHAT_CONTACTS = ["张三", "李四", "王五", "文件传输助手"]


class SimDesktop:
    def __init__(self, tmp_path, logical=(1920, 1080), dpi=1.0, chat_log=None, chat_origin=(100, 100)):
        self.tmp = Path(tmp_path)
        self.logical = logical
        self.dpi = dpi
        self.chat_log = Path(chat_log) if chat_log else self.tmp / "chat.jsonl"
        self.windows = FakeWindowProvider()
        self.calls = []
        self.ox, self.oy = chat_origin
        self.chat = None          # 模拟聊天的状态
        self.browser = None       # 浏览器状态
        self.typed_to_nowhere = []
        self.ocr_calls = 0
        self.hide_contacts = set()

    # ---------------- 启动器 ----------------
    def spawn(self, cmd, **kw):
        s = " ".join(map(str, cmd))
        self.calls.append(("spawn", s))
        if "mock_chat.py" in s:
            self.chat = {"selected": None, "focus": False, "text": "", "sent": []}
            self.windows.add(MOCK_CHAT_TITLE)
        elif "notepad" in s:
            self.windows.add("无标题 - 记事本")
        elif "calc" in s:
            self.windows.add("计算器")

    def startfile(self, target):
        self.calls.append(("startfile", str(target)))
        if str(target).startswith("http"):
            self.browser = {"focus": False, "query": None}
            self.windows.add("必应 - Microsoft Edge")
        else:
            self.windows.add(f"{Path(str(target)).name} - 记事本")

    def app_manager(self, **kw):
        return AppManager(spawn=self.spawn, startfile=self.startfile, windows=self.windows, controller=self,
                          sleep=lambda s: None, search_paths=[self.tmp], open_timeout=1.0, close_timeout=0.5,
                          mock_chat_log=str(self.chat_log), **kw)

    # ---------------- 界面元素（逻辑坐标） ----------------
    def _active(self):
        a = self.windows.active
        return a.title if a in self.windows.wins else ""

    def chat_widgets(self):
        ox, oy = self.ox, self.oy
        w = [(n, [ox + 10, oy + 60 + 50 * i, ox + 200, oy + 100 + 50 * i]) for i, n in enumerate(CHAT_CONTACTS)
             if n not in self.hide_contacts]
        w.append(("当前聊天：" + self.chat["selected"] if self.chat["selected"] else "请选择联系人",
                  [ox + 230, oy + 10, ox + 600, oy + 50]))
        if not self.chat["focus"] or self.chat["text"]:
            w.append((self.chat["text"] or "消息内容", [ox + 230, oy + 500, ox + 700, oy + 540]))
        w.append(("发送", [ox + 720, oy + 500, ox + 800, oy + 540]))
        return w

    def elements_logical(self):
        title = self._active()
        if title == MOCK_CHAT_TITLE and self.chat is not None:
            return self.chat_widgets()
        if "Edge" in title:
            return [("必应", [800, 300, 1100, 360]), ("搜索", [1120, 400, 1200, 440])]
        return [("回收站", [10, 10, 80, 40]), ("此电脑", [10, 80, 80, 110])]

    # ---------------- 截图 + OCR ----------------
    @property
    def save_dir(self):
        return self.tmp

    def capture_full_screen(self, monitor_index=1, save=True):
        w, h = self.logical
        return Image.new("RGB", (round(w * self.dpi), round(h * self.dpi)), "white")

    def detect(self, image_path):
        self.ocr_calls += 1
        d = self.dpi
        return [{"text": t, "bbox": [b[0] * d, b[1] * d, b[2] * d, b[3] * d], "confidence": 0.95}
                for t, b in self.elements_logical()]

    def logical_size(self):
        return self.logical

    # ---------------- controller 接口 ----------------
    def _hit(self, x, y):
        for t, (x1, y1, x2, y2) in self.elements_logical():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return t
        return None

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y))
        hit = self._hit(x, y)
        if self._active() == MOCK_CHAT_TITLE and self.chat is not None:
            if hit in CHAT_CONTACTS:
                self.chat["selected"] = hit
                self.chat["focus"] = False
            elif hit == "发送":
                self._chat_send()
            elif hit is not None and (hit == "消息内容" or hit == self.chat["text"]):
                self.chat["focus"] = True
            else:
                self.chat["focus"] = False

    def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    def type_text(self, text, interval=0.02):
        self.calls.append(("type_text", text))
        title = self._active()
        if title == MOCK_CHAT_TITLE and self.chat and self.chat["focus"]:
            self.chat["text"] += text
        elif "Edge" in title and self.browser and self.browser["focus"]:
            self.browser["query"] = (self.browser["query"] or "") + text
        else:
            self.typed_to_nowhere.append(text)

    def press_key(self, key):
        self.calls.append(("press_key", key))
        title = self._active()
        if key == "enter":
            if title == MOCK_CHAT_TITLE and self.chat and self.chat["focus"]:
                self._chat_send()
            elif "Edge" in title and self.browser and self.browser["focus"] and self.browser["query"]:
                self.windows.active.title = f"{self.browser['query']} - 搜索 - Microsoft Edge"
                self.browser["focus"] = False

    def hotkey(self, *keys):
        self.calls.append(("hotkey",) + keys)
        if keys == ("ctrl", "l") and "Edge" in self._active() and self.browser:
            self.browser["focus"] = True
            self.browser["query"] = ""
        if keys == ("alt", "f4") and self.windows.active in self.windows.wins:
            self.windows.active.close()

    def scroll(self, amount, x=None, y=None):
        self.calls.append(("scroll", amount))

    def _chat_send(self):
        c = self.chat
        if c["selected"] and c["text"].strip():
            c["sent"].append((c["selected"], c["text"]))
            with open(self.chat_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "send", "to": c["selected"], "text": c["text"]}, ensure_ascii=False) + "\n")
            c["text"] = ""

    def names(self):
        return [c[0] for c in self.calls]
