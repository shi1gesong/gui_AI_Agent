"""
GUI Agent 模拟聊天程序（第4周基础任务“发送消息”的测试目标）

为什么不直接用微信：发送消息是对外的、不可撤回的操作，测试时给真实联系人发消息不合适；
真实聊天软件的界面也会随版本变化，不利于反复对比实验。这个模拟程序界面元素与常见聊天软件一致：
左侧联系人列表、顶部当前聊天对象、底部“消息内容”标签 + 输入框和“发送”按钮，
所有元素都是可被 OCR 识别的中文文字。每发送一条消息就把 {"to","text","time"} 追加写入日志，
基础任务测试读取日志客观判定“是否把正确的消息发给了正确的人”。

用法:
    python apps/mock_chat.py [--log 日志路径] [--layout 布局文件] [--borderless] [--geometry 900x600+80+80]
  --layout     把每个控件的文字与屏幕坐标写成 JSON（Xvfb 测试中代替 OCR，验证点击坐标）
  --borderless 去掉标题栏（Xvfb 没有窗口管理器时，窗口坐标 = 屏幕坐标）

第6周鲁棒性测试用的干扰选项（真实界面上复现常见故障，v1 / v2 对比）：
  --load-delay 秒      联系人列表先显示“加载中…”，过几秒才出现（界面加载慢）
  --popup-on-select    第一次选中联系人后弹出模态“系统通知”窗口抢走焦点（弹窗打断）
  --popup-after 秒     启动后若干秒弹出同样的通知窗口
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


POPUP_TITLE = "系统通知"
POPUP_TEXT = "网络连接已恢复"


class MockChatApp:
    def __init__(self, root, log_path, layout_path=None, load_delay: float = 0.0, popup_on_select: bool = False):
        self.root, self.log_path, self.layout_path = root, Path(log_path), layout_path
        self.current = None
        self.popup = None
        self.popup_on_select = popup_on_select
        self._popup_shown = False
        font = pick_font()
        big, mid = (font, 18), (font, 16)
        root.title(TITLE)
        root.configure(bg="#f5f5f5")

        left = tk.Frame(root, bg="#2e2e2e", width=220)
        left.pack(side="left", fill="y")
        tk.Label(left, text="联系人", fg="#cccccc", bg="#2e2e2e", font=mid).pack(pady=(16, 8))
        self.contact_widgets = {}
        self._left, self._big = left, big
        self.loading = None
        if load_delay and load_delay > 0:
            self.loading = tk.Label(left, text="加载中…", fg="#cccccc", bg="#2e2e2e", font=big, padx=20, pady=10)
            self.loading.pack(fill="x")
            root.after(int(load_delay * 1000), self.load_contacts)
        else:
            self.load_contacts()

        right = tk.Frame(root, bg="#f5f5f5")
        right.pack(side="left", fill="both", expand=True)
        self.header = tk.Label(right, text="请选择联系人", font=big, bg="#f5f5f5", anchor="w", padx=16, pady=12)
        self.header.pack(fill="x")
        self.history = tk.Text(right, font=mid, height=10, state="disabled", bg="white", relief="flat")
        self.history.pack(fill="both", expand=True, padx=16)

        bottom = tk.Frame(right, bg="#f5f5f5")
        bottom.pack(fill="x", padx=16, pady=16)
        # 输入框左侧的标签“消息内容”：黑色文字，OCR 能稳定识别；点击标签等同于点击输入框。
        # （2026-11-24 真机：EasyOCR 识别不出输入框里灰色的占位文字，Agent 找不到输入框）
        self.field_label = tk.Label(bottom, text=PLACEHOLDER, font=big, bg="#f5f5f5", fg="black", padx=6,
                                    cursor="xterm")
        self.field_label.pack(side="left", padx=(0, 8))
        self.field_label.bind("<Button-1>", lambda e: self.entry.focus_set())
        self.entry = tk.Entry(bottom, font=big, fg="black", relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True, ipady=8)
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = tk.Button(bottom, text="发送", font=big, command=self.send, padx=18, bg="#07c160", fg="white")
        self.send_btn.pack(side="left", padx=(12, 0))
        self.status = tk.Label(right, text="", font=mid, fg="#c0392b", bg="#f5f5f5")
        self.status.pack(fill="x")
        root.after(500, self.dump_layout)

    def load_contacts(self):
        if self.loading is not None:
            self.loading.destroy()
            self.loading = None
        for name in CONTACTS:
            lbl = tk.Label(self._left, text=name, fg="white", bg="#2e2e2e", font=self._big, padx=20, pady=10,
                           anchor="w", cursor="hand2")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda e, n=name: self.select(n))
            self.contact_widgets[name] = lbl
        self._write({"event": "loaded"})
        self.root.after(100, self.dump_layout)

    def show_popup(self):
        """模态通知窗口：抢焦点、置顶、挡住主窗口的输入（grab_set）"""
        if self.popup is not None:
            return
        self._popup_shown = True
        top = tk.Toplevel(self.root)
        top.title(POPUP_TITLE)
        x, y = self.root.winfo_rootx() + 260, self.root.winfo_rooty() + 180
        top.geometry(f"360x160+{x}+{y}")
        font = (pick_font(), 16)
        tk.Label(top, text=POPUP_TEXT, font=font).pack(pady=(28, 12))
        tk.Button(top, text="确定", font=font, command=self.close_popup, padx=16).pack()
        top.protocol("WM_DELETE_WINDOW", self.close_popup)
        top.bind("<Escape>", lambda e: self.close_popup())
        top.transient(self.root)
        top.lift()
        top.attributes("-topmost", True)
        top.focus_force()
        try:
            top.grab_set()
        except tk.TclError:
            pass
        self.popup = top
        self._write({"event": "popup"})
        self.root.after(100, self.dump_layout)

    def close_popup(self):
        if self.popup is None:
            return
        try:
            self.popup.grab_release()
        except tk.TclError:
            pass
        self.popup.destroy()
        self.popup = None
        self._write({"event": "popup_closed"})
        self.root.after(100, self.dump_layout)

    def select(self, name):
        self.current = name
        for n, w in self.contact_widgets.items():
            w.configure(bg="#07c160" if n == name else "#2e2e2e")
        self.header.configure(text=f"当前聊天：{name}")
        self.entry.focus_set()  # 与常见聊天软件一致：选中联系人后光标自动进入输入框
        self.status.configure(text="")
        self._write({"event": "select", "to": name})
        if self.popup_on_select and not self._popup_shown:
            self.root.after(300, self.show_popup)
        self.root.after(100, self.dump_layout)

    def send(self):
        text = self.entry.get().strip()
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
        add(self.field_label, PLACEHOLDER)
        entry_text = self.entry.get()
        if entry_text:
            add(self.entry, entry_text)
        add(self.send_btn, "发送")
        if self.loading is not None:
            add(self.loading, "加载中…")
        # 第6周：聊天记录逐行导出（后置条件“屏幕上出现 张三：消息”在 Xvfb 用例中用它代替 OCR）
        hx, hy = self.history.winfo_rootx(), self.history.winfo_rooty()
        for i, line in enumerate(self.history.get("1.0", "end").splitlines(), 1):
            info = self.history.dlineinfo(f"{i}.0") if line.strip() else None
            if info:
                x, y, w, h, _ = info
                items.append({"text": line, "bbox": [hx + x, hy + y, hx + x + max(w, 10), hy + y + h], "confidence": 0.99})
        if self.popup is not None:
            try:
                px, py = self.popup.winfo_rootx(), self.popup.winfo_rooty()
                items.append({"text": POPUP_TEXT, "bbox": [px + 20, py + 20, px + 340, py + 60], "confidence": 0.99,
                              "window": POPUP_TITLE})
            except tk.TclError:
                pass
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
    ap.add_argument("--load-delay", type=float, default=0.0, help="联系人延迟出现的秒数")
    ap.add_argument("--popup-on-select", action="store_true", help="第一次选中联系人后弹出模态通知")
    ap.add_argument("--popup-after", type=float, default=0.0, help="启动后多少秒弹出通知（0 = 不弹）")
    args = ap.parse_args(argv)
    root = tk.Tk()
    root.geometry(args.geometry)
    if args.borderless:
        root.overrideredirect(True)
    app = MockChatApp(root, args.log, args.layout, load_delay=args.load_delay, popup_on_select=args.popup_on_select)
    app._write({"event": "start"})
    if args.popup_after and args.popup_after > 0:
        root.after(int(args.popup_after * 1000), app.show_popup)
    root.after(200, lambda: (root.lift(), root.focus_force()))
    root.mainloop()


if __name__ == "__main__":
    main()
