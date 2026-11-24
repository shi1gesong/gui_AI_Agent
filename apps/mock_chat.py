"""
GUI Agent 模拟聊天程序（第4周基础任务“发送消息”的测试目标）

为什么不直接用微信：发送消息是对外的、不可撤回的操作，测试时给真实联系人发消息不合适；
真实聊天软件的界面也会随版本变化，不利于反复对比实验。这个模拟程序界面元素与常见聊天软件一致：
左侧联系人列表、顶部当前聊天对象、底部消息输入框（占位文字“消息内容”）和“发送”按钮，
所有元素都是可被 OCR 识别的中文文字。每发送一条消息就把 {"to","text","time"} 追加写入日志，
基础任务测试读取日志客观判定“是否把正确的消息发给了正确的人”。

用法:
    python apps/mock_chat.py [--log 日志路径] [--layout 布局文件] [--borderless] [--geometry 900x600+80+80]
  --layout     把每个控件的文字与屏幕坐标写成 JSON（Xvfb 测试中代替 OCR，验证点击坐标）
  --borderless 去掉标题栏（Xvfb 没有窗口管理器时，窗口坐标 = 屏幕坐标）
"""
import argparse
import json
import os
import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

TITLE = "GUI Agent 模拟聊天"
CONTACTS = ["张三", "李四", "王五", "文件传输助手"]
PLACEHOLDER = "消息内容"
DEFAULT_LOG = Path(tempfile.gettempdir()) / "gui_agent_mock_chat.jsonl"


def pick_font():
    if sys.platform.startswith("win"):
        return "Microsoft YaHei"
    try:
        import tkinter.font as tkfont
        fams = set(tkfont.families())
        for f in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "PingFang SC", "Microsoft YaHei"):
            if f in fams:
                return f
    except Exception:
        pass
    return "TkDefaultFont"


class MockChatApp:
    def __init__(self, root, log_path, layout_path=None):
        self.root, self.log_path, self.layout_path = root, Path(log_path), layout_path
        self.current = None
        font = pick_font()
        big, mid = (font, 18), (font, 16)
        root.title(TITLE)
        root.configure(bg="#f5f5f5")

        left = tk.Frame(root, bg="#2e2e2e", width=220)
        left.pack(side="left", fill="y")
        tk.Label(left, text="联系人", fg="#cccccc", bg="#2e2e2e", font=mid).pack(pady=(16, 8))
        self.contact_widgets = {}
        for name in CONTACTS:
            lbl = tk.Label(left, text=name, fg="white", bg="#2e2e2e", font=big, padx=20, pady=10, anchor="w",
                           cursor="hand2")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, n=name: self.select(n))
            self.contact_widgets[name] = lbl

        right = tk.Frame(root, bg="#f5f5f5")
        right.pack(side="left", fill="both", expand=True)
        self.header = tk.Label(right, text="请选择联系人", font=big, bg="#f5f5f5", anchor="w", padx=16, pady=12)
        self.header.pack(fill="x")
        self.history = tk.Text(right, font=mid, height=10, state="disabled", bg="white", relief="flat")
        self.history.pack(fill="both", expand=True, padx=16)

        bottom = tk.Frame(right, bg="#f5f5f5")
        bottom.pack(fill="x", padx=16, pady=16)
        self.entry = tk.Entry(bottom, font=big, fg="#666666", relief="solid", bd=1)
        self.entry.insert(0, PLACEHOLDER)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.entry.bind("<FocusIn>", self._clear_placeholder)
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = tk.Button(bottom, text="发送", font=big, command=self.send, padx=18, bg="#07c160", fg="white")
        self.send_btn.pack(side="left", padx=(12, 0))
        self.status = tk.Label(right, text="", font=mid, fg="#c0392b", bg="#f5f5f5")
        self.status.pack(fill="x")
        root.after(500, self.dump_layout)

    def _clear_placeholder(self, _e=None):
        if self.entry.get() == PLACEHOLDER and self.entry.cget("fg") == "#666666":
            self.entry.delete(0, "end")
            self.entry.configure(fg="black")

    def select(self, name):
        self.current = name
        for n, w in self.contact_widgets.items():
            w.configure(bg="#07c160" if n == name else "#2e2e2e")
        self.header.configure(text=f"当前聊天：{name}")
        self.status.configure(text="")
        self._write({"event": "select", "to": name})
        self.root.after(100, self.dump_layout)

    def send(self):
        text = self.entry.get()
        if self.entry.cget("fg") == "#666666":
            text = ""
        text = text.strip()
        if not self.current:
            self.status.configure(text="请先选择联系人")
            self._write({"event": "error", "reason": "no_contact", "text": text})
            return
        if not text:
            self.status.configure(text="消息不能为空")
            return
        self.history.configure(state="normal")
        self.history.insert("end", f"我 → {self.current}：{text}\n")
        self.history.configure(state="disabled")
        self.entry.delete(0, "end")
        self._write({"event": "send", "to": self.current, "text": text})
        self.root.after(100, self.dump_layout)

    def _write(self, rec):
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def dump_layout(self):
        """导出控件文字与屏幕坐标（只在 --layout 时）。坐标 = 控件在屏幕上的 [x1,y1,x2,y2]"""
        if not self.layout_path:
            return
        self.root.update_idletasks()
        items = []

        def add(widget, text):
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            items.append({"text": text, "bbox": [x, y, x + widget.winfo_width(), y + widget.winfo_height()],
                          "confidence": 0.99})

        for n, w in self.contact_widgets.items():
            add(w, n)
        add(self.header, self.header.cget("text"))
        entry_text = self.entry.get()
        if entry_text:
            add(self.entry, entry_text)
        add(self.send_btn, "发送")
        tmp = str(self.layout_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        os.replace(tmp, self.layout_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--log", default=os.environ.get("GUI_AGENT_CHAT_LOG", str(DEFAULT_LOG)))
    ap.add_argument("--layout", default=None)
    ap.add_argument("--borderless", action="store_true")
    ap.add_argument("--geometry", default="900x600+80+80")
    args = ap.parse_args(argv)
    root = tk.Tk()
    root.geometry(args.geometry)
    if args.borderless:
        root.overrideredirect(True)
    app = MockChatApp(root, args.log, args.layout)
    app._write({"event": "start"})
    root.after(200, lambda: (root.lift(), root.focus_force()))
    root.mainloop()


if __name__ == "__main__":
    main()
