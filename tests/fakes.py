"""测试替身：模拟 pyautogui / mss / 剪贴板 / LLM，使大部分用例可在无屏幕、无模型的环境运行。"""
import types


class FakePyAutoGUI:
    """记录每一次调用的 pyautogui 替身。"""

    class FailSafeException(Exception):
        pass

    def __init__(self, width=1920, height=1080):
        self.calls = []
        self.width, self.height = width, height
        self.pos = (width // 2, height // 2)
        self.PAUSE, self.FAILSAFE = 0.3, True

    def _rec(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def names(self):
        return [c[0] for c in self.calls]

    def size(self):
        return (self.width, self.height)

    def position(self):
        return self.pos

    def click(self, x=None, y=None, button="left", clicks=1, **kw):
        self._rec("click", x=x, y=y, button=button, clicks=clicks, **kw)
        if x is not None and y is not None:
            self.pos = (x, y)

    def doubleClick(self, x=None, y=None, **kw):
        self._rec("doubleClick", x=x, y=y, **kw)
        if x is not None and y is not None:
            self.pos = (x, y)

    def moveTo(self, x, y, duration=0.0, **kw):
        self._rec("moveTo", x, y, duration=duration, **kw)
        self.pos = (x, y)

    def dragTo(self, x, y, duration=0.0, button="left", **kw):
        self._rec("dragTo", x, y, duration=duration, button=button, **kw)
        self.pos = (x, y)

    def scroll(self, clicks, **kw):
        self._rec("scroll", clicks, **kw)

    def typewrite(self, text, interval=0.0, **kw):
        self._rec("typewrite", text, interval=interval, **kw)

    def press(self, key, **kw):
        self._rec("press", key, **kw)

    def hotkey(self, *keys, **kw):
        self._rec("hotkey", *keys, **kw)


class FakeClipboard(types.ModuleType):
    """pyperclip 替身"""

    def __init__(self, initial="原剪贴板内容"):
        super().__init__("pyperclip")
        self.content = initial
        self.history = []

    def copy(self, text):
        self.history.append(text)
        self.content = text

    def paste(self):
        return self.content


class _Shot:
    def __init__(self, width, height, bgrx=(10, 20, 30, 255)):
        self.size = (width, height)
        self.width, self.height = width, height
        self.bgra = bytes(bgrx) * (width * height)


class FakeMSS:
    """mss.mss() 替身。monitors[0] 为所有屏幕拼接，monitors[1..] 为物理屏。"""

    def __init__(self, screens=((1920, 1080),), bgrx=(10, 20, 30, 255)):
        self.grabs = []
        self.bgrx = bgrx
        mons, left = [], 0
        for w, h in screens:
            mons.append({"left": left, "top": 0, "width": w, "height": h})
            left += w
        total_w = sum(w for w, _ in screens)
        total_h = max(h for _, h in screens)
        self.monitors = [{"left": 0, "top": 0, "width": total_w, "height": total_h}] + mons

    def grab(self, region):
        self.grabs.append(dict(region))
        w, h = region["width"], region["height"]
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid region size {w}x{h}")
        return _Shot(w, h, self.bgrx)


class ScriptedLLM:
    """按顺序返回预设文本的 BaseLLMInterface 替身，并记录收到的 prompt。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, image_path=None):
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("ScriptedLLM: 预设回复已用完")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
