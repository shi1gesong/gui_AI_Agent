"""
Xvfb 真实 GUI 测试用的“事件记录器”窗口（被测对象之外的测试夹具）。

在虚拟屏幕上铺满一个 Tk 窗口：上半部分是 Text 输入框，下半部分是 Canvas。
所有鼠标/键盘/滚轮事件按 JSON Lines 写入日志文件，测试用例读取日志来断言
pyautogui 产生的真实 X11 事件（点击位置、拖拽起止、滚轮方向、输入的文字等）。

用法: python event_logger_app.py <log_path> <width> <height>
"""
import json
import sys
import tkinter as tk


def main():
    log_path, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(**kw):
        log.write(json.dumps(kw, ensure_ascii=False) + "\n")

    root = tk.Tk()
    root.geometry(f"{width}x{height}+0+0")
    root.overrideredirect(True)  # 无窗口管理器时去掉标题栏，保证窗口坐标 == 屏幕坐标

    text = tk.Text(root, font=("Noto Sans CJK SC", 14))
    text.place(x=0, y=0, width=width, height=height // 2)
    canvas = tk.Canvas(root, bg="white")
    canvas.place(x=0, y=height // 2, width=width, height=height - height // 2)

    def dump_text():
        write(ev="text", value=text.get("1.0", "end-1c"))

    for btn in (1, 2, 3):
        root.bind_all(f"<ButtonPress-{btn}>",
                      lambda e, b=btn: write(ev="press", b=b, x=e.x_root, y=e.y_root), add="+")
        root.bind_all(f"<ButtonRelease-{btn}>",
                      lambda e, b=btn: write(ev="release", b=b, x=e.x_root, y=e.y_root), add="+")
    root.bind_all("<Double-Button-1>", lambda e: write(ev="double", x=e.x_root, y=e.y_root), add="+")
    # X11 滚轮：Button-4 = 向上，Button-5 = 向下
    for btn, direction in ((4, "up"), (5, "down")):
        root.bind_all(f"<Button-{btn}>",
                      lambda e, d=direction: write(ev="wheel", dir=d, x=e.x_root, y=e.y_root), add="+")
    root.bind_all("<B1-Motion>", lambda e: write(ev="drag_motion", x=e.x_root, y=e.y_root), add="+")
    root.bind_all("<KeyPress>", lambda e: write(ev="key", keysym=e.keysym, state=e.state), add="+")
    root.bind_all("<KeyRelease>", lambda e: root.after(20, dump_text), add="+")
    text.bind("<<Paste>>", lambda e: root.after(100, dump_text), add="+")

    def on_ready():
        text.focus_force()
        write(ev="ready", w=root.winfo_screenwidth(), h=root.winfo_screenheight())

    root.after(300, on_ready)
    root.mainloop()


if __name__ == "__main__":
    main()
