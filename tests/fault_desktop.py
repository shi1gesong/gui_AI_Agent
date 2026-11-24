"""
带故障注入的模拟桌面（第6周鲁棒性测试）

与第4周 tests/sim_desktop.py 的区别：
  1. 截图是真的“画”出来的：每个界面元素画成一个独特颜色的色块（同一文字永远同一颜色），窗口有底色，
     输入框画成白底黑边。这样截图前后对比（EffectVerifier）、OpenCV 控件检测、按前台窗口裁剪、
     OCR 缓存/增量识别都在真实像素上工作。
  2. “OCR” 从像素反查：在图里找每种颜色的外接矩形 → 还原出文字和坐标。所以裁剪、放大、DPI 缩放后
     识别结果会跟着像素变化，和真实 OCR 一样只认截图内容。
  3. 虚拟时钟：Agent / 执行器 / AppManager 的 sleep 都推进 self.t，界面加载慢、定时弹窗按虚拟时间发生，
     单元测试不需要真的等待。
  4. 故障（faults）：
       slow_load        聊天程序联系人 load_delay 秒后才出现（默认 4 秒）
       popup_on_select  第一次选中联系人后弹出模态“系统通知”，抢走焦点、挡住输入
       popup_timed      聊天程序启动 popup_at 秒后弹出同样的通知（默认 2 秒）
       lost_click       每个目标的第一次点击被吞掉（程序没响应）
       ocr_flaky        OCR 每次以概率 ocr_drop 漏掉元素（默认 0.35）
       ocr_typo         OCR 每次以概率 ocr_typo 把元素文字错一个字（默认 0.35）
       focus_steal      选中联系人后，用户原有的“文件资源管理器”窗口被激活一次（焦点被抢）
       screenshot_fail  第 2、3 次截图抛异常（mss 偶发错误）
       slow_window      聊天程序窗口启动 window_delay 秒后才出现（默认 4 秒，超过打开等待时间）
       dpi_change       聊天程序打开后显示缩放从 100% 变成 150%
       llm_down         （由测试构造的 LLM 替身实现：Ollama 连不上）
     超出 v2 能力范围的两种（用来如实标出边界）：
       slow_load_long   联系人 15 秒后才出现（超过重试退避总时长）
       ocr_blind        “发送”按钮只有图标没有文字，OCR 永远识别不到
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from control.app_manager import MOCK_CHAT_TITLE, AppManager
from fakes import FakeWindowProvider

CONTACTS = ["张三", "李四", "王五", "文件传输助手"]
POPUP_TITLE, POPUP_TEXT = "系统通知", "网络连接已恢复"
EXPLORER_TITLE = "文档 - 文件资源管理器"
FAULTS = ("none", "slow_load", "popup_on_select", "popup_timed", "lost_click", "ocr_flaky", "ocr_typo",
          "focus_steal", "screenshot_fail", "slow_window", "dpi_change", "llm_down", "slow_load_long", "ocr_blind")
TYPO_CHARS = "口日目王十土工干士大"


class FaultyDesktop:
    def __init__(self, tmp_path, logical=(1280, 720), dpi=1.0, faults=(), seed=0, **params):
        self.tmp = Path(tmp_path)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.logical = logical
        self.dpi = dpi
        self.faults = set(faults)
        unknown = self.faults - set(FAULTS)
        if unknown:
            raise ValueError(f"未知故障 {unknown}")
        self.p = {"load_delay": 4.0, "popup_at": 2.0, "ocr_drop": 0.35, "ocr_typo": 0.35, "window_delay": 4.0,
                  "fail_shots": (2, 3), "new_dpi": 1.5, **params}
        self.rng = random.Random(seed)
        self.t = 0.0
        self.chat_log = self.tmp / "chat.jsonl"
        self.windows = FakeWindowProvider()
        self.explorer = self._add_window(EXPLORER_TITLE, (40, 40, 700, 520))   # 用户原有窗口
        self.calls = []
        self.chat = None
        self.chat_win = None
        self.browser = None
        self.notepads = {}           # FakeWindow -> {"path", "text"}
        self.popup = None
        self.typed_to_nowhere = []
        self.pending = []            # [(time, callable)]
        self.clicked_once = set()
        self.shots = 0
        self.ocr_calls = 0
        self._colors = {}
        self._color_list = []

    # ---------------- 时间 ----------------
    def sleep(self, sec):
        self.t += max(0.0, float(sec or 0))
        self._tick()

    def _tick(self):
        due = [p for p in self.pending if p[0] <= self.t + 1e-9]
        self.pending = [p for p in self.pending if p[0] > self.t + 1e-9]
        for _, fn in sorted(due, key=lambda p: p[0]):
            fn()

    def _at(self, delay, fn):
        self.pending.append((self.t + delay, fn))

    # ---------------- 窗口 ----------------
    def _add_window(self, title, rect):
        w = self.windows.add(title)
        w.left, w.top, w.width, w.height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
        return w

    def active(self):
        a = self.windows.active
        if a not in self.windows.wins:       # 前台窗口被关掉：系统激活最上层的剩余窗口
            a = self.windows.wins[-1] if self.windows.wins else None
            self.windows.active = a
        return a

    def _title(self):
        a = self.active()
        return a.title if a is not None else ""

    def _popup_open(self):
        return self.popup is not None and self.popup in self.windows.wins

    def _open_chat(self):
        self.chat = {"selected": None, "focus": False, "text": "", "sent": [], "history": [], "opened": self.t,
                     "popped": False, "stolen": False}
        self.chat_win = self._add_window(MOCK_CHAT_TITLE, (100, 60, 920, 640))
        if "popup_timed" in self.faults:
            self._at(self.p["popup_at"], self._show_popup)
        if "dpi_change" in self.faults:
            self.dpi = self.p["new_dpi"]

    def _show_popup(self):
        if self._popup_open() or self.chat_win not in self.windows.wins:
            return
        self.popup = self._add_window(POPUP_TITLE, (380, 260, 740, 420))

    def spawn(self, cmd, **kw):
        s = " ".join(map(str, cmd))
        self.calls.append(("spawn", s))
        if "mock_chat.py" in s:
            if "slow_window" in self.faults:
                self._at(self.p["window_delay"], self._open_chat)
            else:
                self._open_chat()
        elif "calc" in s:
            self._add_window("计算器", (900, 80, 1220, 560))
        elif "notepad" in s:
            w = self._add_window("无标题 - 记事本", (200, 100, 1000, 600))
            self.notepads[w] = {"path": None, "text": ""}

    def startfile(self, target):
        self.calls.append(("startfile", str(target)))
        if str(target).startswith("http"):
            self.browser = {"focus": False, "query": None}
            self._add_window("必应 - Microsoft Edge", (0, 0, 1280, 700))
        else:
            p = Path(str(target))
            w = self._add_window(f"{p.name} - 记事本", (200, 100, 1000, 600))
            self.notepads[w] = {"path": p, "text": p.read_text(encoding="utf-8-sig") if p.exists() else ""}

    def app_manager(self, **kw):
        kw.setdefault("open_timeout", 3.0)
        return AppManager(spawn=self.spawn, startfile=self.startfile, windows=self.windows, controller=self,
                          sleep=self.sleep, search_paths=[self.tmp], close_timeout=1.0, poll=0.25,
                          mock_chat_log=str(self.chat_log), **kw)

    # ---------------- 界面元素（逻辑坐标） ----------------
    def _chat_elements(self):
        c, w = self.chat, self.chat_win
        ox, oy = w.left, w.top
        els = [(MOCK_CHAT_TITLE, [ox + 10, oy + 5, ox + 300, oy + 35])]
        delay = 15.0 if "slow_load_long" in self.faults else self.p["load_delay"]
        loaded = not ({"slow_load", "slow_load_long"} & self.faults) or self.t - c["opened"] >= delay
        if loaded:
            els += [(n, [ox + 10, oy + 60 + 50 * i, ox + 200, oy + 100 + 50 * i]) for i, n in enumerate(CONTACTS)]
        else:
            els.append(("加载中…", [ox + 10, oy + 60, ox + 200, oy + 100]))
        els.append(("当前聊天：" + c["selected"] if c["selected"] else "请选择联系人", [ox + 230, oy + 50, ox + 600, oy + 90]))
        for i, line in enumerate(c["history"][-8:]):
            els.append((line, [ox + 240, oy + 110 + 40 * i, ox + 700, oy + 140 + 40 * i]))
        els.append(("消息内容", [ox + 230, oy + 500, ox + 330, oy + 540]))
        if c["text"]:
            els.append((c["text"], [ox + 350, oy + 505, ox + 350 + 24 * len(c["text"]), oy + 535]))
        els.append(("发送", [ox + 720, oy + 500, ox + 800, oy + 540]))
        return els

    def input_rect(self):
        w = self.chat_win
        return [w.left + 340, w.top + 500, w.left + 700, w.top + 540]

    def _window_elements(self, w):
        if w is self.chat_win and self.chat is not None:
            return self._chat_elements()
        if w is self.popup:
            return [(POPUP_TEXT, [w.left + 30, w.top + 30, w.left + 330, w.top + 70]),
                    ("确定", [w.left + 140, w.top + 100, w.left + 220, w.top + 140])]
        if "Edge" in w.title:
            return [(w.title[:18], [10, 5, 400, 35]), ("必应", [500, 250, 780, 310]), ("搜索", [800, 260, 880, 300])]
        if w in self.notepads:
            n = self.notepads[w]
            els = [(w.title[:30], [w.left + 10, w.top + 5, w.left + 500, w.top + 35])]
            if n["text"]:
                els.append((n["text"][:40], [w.left + 20, w.top + 60, w.left + 20 + 22 * min(len(n["text"]), 30), w.top + 90]))
            return els
        if w.title == "计算器":
            return [("计算器", [w.left + 10, w.top + 5, w.left + 120, w.top + 35]), ("0", [w.left + 250, w.top + 60, w.left + 300, w.top + 120])]
        return [(w.title[:12], [w.left + 10, w.top + 5, w.left + 300, w.top + 35]), ("此电脑", [w.left + 20, w.top + 60, w.left + 120, w.top + 90])]

    def visible(self):
        """[(window, elements)]：前台窗口画在最上面；模态弹窗打开时它和它的主窗口都可见"""
        a = self.active()
        order = [w for w in self.windows.wins if w is not a] + ([a] if a is not None else [])
        if self._popup_open() and a is not self.popup:
            order = [w for w in order if w is not self.popup] + [self.popup]
        return [(w, self._window_elements(w)) for w in order[-2:]]

    def elements_logical(self):
        out = []
        for _, els in self.visible():
            out.extend(els)
        return out

    # ---------------- 截图（真的画出来）+ “OCR”（从像素反查） ----------------
    def _color(self, text):
        if text not in self._colors:
            i = len(self._color_list) + 1
            c = (40 + (i * 53) % 160, 40 + (i * 97) % 160, 40 + (i * 31) % 160)
            while c in self._color_list:
                i += 7
                c = (40 + (i * 53) % 160, 40 + (i * 97) % 160, 40 + (i * 31) % 160)
            self._colors[text] = c
            self._color_list.append(c)
        return self._colors[text]

    @property
    def save_dir(self):
        return self.tmp

    def capture_full_screen(self, monitor_index=1, save=True):
        self.shots += 1
        if "screenshot_fail" in self.faults and self.shots in self.p["fail_shots"]:
            raise OSError("截图失败: 模拟 mss 抓屏错误")
        d = self.dpi
        W, H = self.logical
        img = Image.new("RGB", (round(W * d), round(H * d)), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        s = lambda b: [round(v * d) for v in b]  # noqa: E731
        for w, els in self.visible():
            dr.rectangle(s([w.left, w.top, w.left + w.width, w.top + w.height]), fill=(236, 236, 236), outline=(90, 90, 90))
            if w is self.chat_win and self.chat is not None:
                dr.rectangle(s(self.input_rect()), fill=(255, 255, 255), outline=(0, 0, 0))
            for text, b in els:
                dr.rectangle(s(b), fill=self._color(text))
        return img

    def detect(self, image_path):
        self.ocr_calls += 1
        arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int32)
        code = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
        out = []
        for text, c in self._colors.items():
            mask = code == ((c[0] << 16) | (c[1] << 8) | c[2])
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            if "ocr_blind" in self.faults and text == "发送":
                continue
            if "ocr_flaky" in self.faults and self.rng.random() < self.p["ocr_drop"]:
                continue
            t = text
            if "ocr_typo" in self.faults and len(t) >= 2 and self.rng.random() < self.p["ocr_typo"]:
                i = self.rng.randrange(len(t))
                t = t[:i] + self.rng.choice(TYPO_CHARS) + t[i + 1:]
            out.append({"text": t, "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                        "confidence": 0.93})
        return out

    def logical_size(self):
        return self.logical

    # ---------------- controller ----------------
    def _hit(self, x, y):
        for w, els in reversed(self.visible()):
            for t, (x1, y1, x2, y2) in els:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return w, t
            if w.left <= x <= w.left + w.width and w.top <= y <= w.top + w.height:
                return w, None
        return None, None

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y))
        w, hit = self._hit(x, y)
        if "lost_click" in self.faults:
            key = hit or (round(x, -1), round(y, -1))
            if key not in self.clicked_once:
                self.clicked_once.add(key)
                return  # 程序没响应：第一次点击被吞掉
        if self._popup_open():
            if w is self.popup:
                self.windows.active = self.popup
                if hit == "确定":
                    self.popup.close()
                    self.windows.active = self.chat_win
            return  # 模态弹窗：主窗口不响应
        if w is not None and w is not self.active():
            self.windows.active = w      # 点到哪个窗口哪个窗口到前台
        if w is self.chat_win and self.chat is not None:
            c = self.chat
            ir = self.input_rect()
            if hit in CONTACTS:
                c["selected"] = hit
                c["focus"] = True
                if "popup_on_select" in self.faults and not c["popped"]:
                    c["popped"] = True
                    self._show_popup()
                if "focus_steal" in self.faults and not c["stolen"]:
                    c["stolen"] = True
                    self._at(0.01, lambda: setattr(self.windows, "active", self.explorer))
            elif hit == "发送":
                self._chat_send()
            elif hit == "消息内容" or hit == c["text"] or (ir[0] <= x <= ir[2] and ir[1] <= y <= ir[3]):
                c["focus"] = True
            else:
                c["focus"] = False

    def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    def move_to(self, x, y):
        self.calls.append(("move_to", x, y))

    def type_text(self, text, interval=0.02):
        self.calls.append(("type_text", text))
        a = self.active()
        if a is self.chat_win and self.chat and self.chat["focus"] and not self._popup_open():
            self.chat["text"] += text
        elif a is not None and "Edge" in a.title and self.browser and self.browser["focus"]:
            self.browser["query"] = (self.browser["query"] or "") + text
        elif a in self.notepads:
            self.notepads[a]["text"] += text
        else:
            self.typed_to_nowhere.append(text)

    def press_key(self, key):
        self.calls.append(("press_key", key))
        a = self.active()
        if key == "esc" and self._popup_open() and a is self.popup:
            self.popup.close()
            self.windows.active = self.chat_win
            return
        if key == "enter":
            if a is self.chat_win and self.chat and self.chat["focus"] and not self._popup_open():
                self._chat_send()
            elif a is not None and "Edge" in a.title and self.browser and self.browser["focus"] and self.browser["query"]:
                a.title = f"{self.browser['query']} - 搜索 - Microsoft Edge"
                self.browser["focus"] = False

    def hotkey(self, *keys):
        self.calls.append(("hotkey",) + keys)
        a = self.active()
        if keys == ("ctrl", "l") and a is not None and "Edge" in a.title and self.browser:
            self.browser["focus"] = True
            self.browser["query"] = ""
        elif keys == ("alt", "f4") and a is not None:
            a.close()
        elif keys == ("ctrl", "s") and a in self.notepads and self.notepads[a]["path"]:
            n = self.notepads[a]
            n["path"].write_text(n["text"], encoding="utf-8")
        elif keys == ("ctrl", "w") and a in self.notepads:
            a.close()

    def scroll(self, amount, x=None, y=None):
        self.calls.append(("scroll", amount))

    def _chat_send(self):
        c = self.chat
        if c["selected"] and c["text"].strip():
            c["sent"].append((c["selected"], c["text"]))
            c["history"].append(f"我 → {c['selected']}：{c['text']}")
            with open(self.chat_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "send", "to": c["selected"], "text": c["text"]}, ensure_ascii=False) + "\n")
            c["text"] = ""
