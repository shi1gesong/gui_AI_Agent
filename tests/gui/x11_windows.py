"""
Xvfb 下的窗口提供者（第6周）：用 python-xlib 实现与 pygetwindow 相同的最小接口
（all_windows / active_window；窗口有 title/left/top/width/height/close()/activate()），
让 v2 的弹窗检测、前台窗口检查、重新激活窗口在 Linux 虚拟屏幕的真实 Tk 窗口上也能验证。
Xvfb 没有窗口管理器：“前台窗口” = 当前拥有键盘焦点的顶层窗口；关闭 = 发送 WM_DELETE_WINDOW。
"""
from Xlib import X, display, protocol


class XWindow:
    def __init__(self, prov, win, title):
        self.prov, self.win, self.title = prov, win, title
        self._hWnd = win.id
        g = win.get_geometry()
        c = win.translate_coords(prov.root, 0, 0)
        self.left, self.top = -c.x, -c.y
        self.width, self.height = g.width, g.height
        self.isMinimized = False

    def close(self):
        d = self.prov.d
        ev = protocol.event.ClientMessage(window=self.win, client_type=d.intern_atom("WM_PROTOCOLS"),
                                          data=(32, [d.intern_atom("WM_DELETE_WINDOW"), X.CurrentTime, 0, 0, 0]))
        self.win.send_event(ev, event_mask=X.NoEventMask)
        d.sync()

    def activate(self):
        self.win.raise_window()
        self.win.set_input_focus(X.RevertToParent, X.CurrentTime)
        self.prov.d.sync()

    def restore(self):
        pass

    def __repr__(self):
        return f"XWindow({self.title!r})"


class XlibWindowProvider:
    def __init__(self):
        self.d = display.Display()
        self.root = self.d.screen().root
        self._net_name = self.d.intern_atom("_NET_WM_NAME")
        self._utf8 = self.d.intern_atom("UTF8_STRING")

    def _title(self, w):
        try:
            p = w.get_full_property(self._net_name, self._utf8)
            if p is not None and p.value:
                v = p.value
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
            n = w.get_wm_name()
            return n.decode("utf-8", "replace") if isinstance(n, bytes) else (n or "")
        except Exception:
            return ""

    def all_windows(self):
        out = []
        for w in self.root.query_tree().children:
            try:
                if w.get_attributes().map_state != X.IsViewable:
                    continue
            except Exception:
                continue
            t = self._title(w)
            if t:
                out.append(XWindow(self, w, t))
        return out

    def active_window(self):
        f = self.d.get_input_focus().focus
        if not hasattr(f, "query_tree"):   # PointerRoot / None：没有窗口拥有焦点
            return None
        w = f
        while True:
            try:
                parent = w.query_tree().parent
            except Exception:
                return None
            if not hasattr(parent, "id") or parent.id == self.root.id:
                break
            w = parent
        for x in self.all_windows():
            if x.win.id == w.id:
                return x
        return None
