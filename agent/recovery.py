"""
错误检测与自动重试（第6周新增）

第4周 v1.0 的容错只有一条规则：子任务失败 → 固定等 1 秒 → 再试 1 次。真机上遇到的失败原因各不相同，
处理方式也应该不同：

  错误类别              典型现象                                  v2.0 处理
  ─────────────────────────────────────────────────────────────────────────────────────────────
  target_not_found      OCR 没找到“张三”（界面还在加载 / 被遮挡）     指数退避后重新感知，最多 4 次；先检查弹窗、重新激活目标窗口
  popup                 前台突然出现“系统通知/更新提示”等新窗口          关闭本次运行期间新出现的弹窗（只关新窗口，不碰用户原有窗口）→ 激活目标窗口
  window_lost           焦点跑到别的窗口（命令行、其它程序）           重新激活目标程序窗口
  no_effect             动作执行了，但截图前后界面没有任何变化          重新激活窗口后重试（点击被吞、窗口没响应）
  invalid_action        规则/LLM 推断不出动作、动作未通过校验           LLM 重新决策；纯规则推断不出时不重试（重试也没用）
  execution             执行器异常（坐标格式错、组合键无法识别…）         重试 1 次
  app_launch            程序没有在超时时间内出现窗口                   再等一次（延长等待）
  perception            截图 / OCR 抛异常                          退避后重试
  service               Ollama / VLM 服务连不上、超时               熔断器：连续失败 N 次后本次运行降级为规则决策
  abort                 用户把鼠标甩到屏幕角落（FailSafe）/ Ctrl+C     立即停止，绝不重试
  postcondition         子任务都成功了，但目标的结果检查不通过          可重放的目标整体重做 1 次；“发送消息”这类不可重复的目标不重放

所有等待都用注入的 sleep，单元测试不会真的等。
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

ERROR_CLASSES = ("target_not_found", "popup", "window_lost", "no_effect", "invalid_action", "execution",
                 "app_launch", "perception", "service", "abort", "postcondition", "unknown")

# 弹窗标题关键字（只用于“本次运行期间新出现的窗口”，不会去关用户原来就开着的窗口）
POPUP_HINTS = ["通知", "提示", "警告", "错误", "更新", "确认", "广告", "notification", "alert", "warning", "error",
               "update", "confirm", "dialog", "消息框"]

_PATTERNS = [
    ("abort", r"FailSafe|KeyboardInterrupt|用户中止|紧急停止"),
    ("service", r"ConnectionError|ConnectionRefused|Connection refused|Max retries exceeded|ReadTimeout|"
                r"ConnectTimeout|Timeout|timed out|HTTP 5\d\d|VLM 服务|Ollama|连接失败|服务不可用"),
    ("popup", r"弹窗|popup"),
    ("window_lost", r"前台窗口不是|焦点丢失|window_lost|目标窗口不在前台"),
    ("no_effect", r"界面没有变化|no_effect|动作没有生效"),
    ("app_launch", r"秒内没有出现标题含|启动「.*」失败|没有找到标题含"),
    ("target_not_found", r"找不到|没有找到|LookupError|不在当前屏幕|屏幕上没有"),
    ("perception", r"截图失败|OCR|感知失败|perceive|ScreenPerceiver|mss"),
    ("invalid_action", r"未通过校验|规则无法推断|无法推断|没有给出可执行动作|_invalid"),
    ("postcondition", r"结果检查不通过|postcondition"),
    ("execution", r"ValueError|RuntimeError|TypeError|无法执行|无法识别|坐标格式错误|执行失败"),
]
_COMPILED = [(c, re.compile(p, re.IGNORECASE)) for c, p in _PATTERNS]


def classify_error(error, action: Optional[dict] = None) -> str:
    """把一次失败归类。error 可以是异常对象或错误字符串；action 是本次尝试的动作（可选）"""
    if isinstance(error, BaseException):
        name = type(error).__name__
        if name in ("FailSafeException", "KeyboardInterrupt"):
            return "abort"
        text = f"{name}: {error}"
    else:
        text = str(error or "")
    if isinstance(action, dict) and action.get("_invalid"):
        if action.get("_retryable"):
            return "target_not_found"
        reasons = " ".join(map(str, action.get("_reasons") or []))
        for c, rx in _COMPILED:
            if c in ("target_not_found", "service") and rx.search(reasons):
                return c
        return "invalid_action"
    for c, rx in _COMPILED:
        if rx.search(text):
            return c
    return "unknown"


@dataclass
class RetryPolicy:
    """每类错误最多尝试几次（含第一次）+ 指数退避"""
    max_attempts: Dict[str, int] = field(default_factory=lambda: {
        "target_not_found": 4, "popup": 3, "window_lost": 3, "no_effect": 2, "invalid_action": 2, "execution": 2,
        "app_launch": 2, "perception": 3, "service": 2, "abort": 1, "postcondition": 2, "unknown": 2})
    base_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 4.0
    max_total_attempts: int = 6       # 一个子任务所有错误类别加起来最多尝试几次

    def allowed(self, error_class: str, attempt: int, total_attempts: Optional[int] = None) -> bool:
        """attempt: 刚刚失败的是这一类错误的第几次（从 1 开始）"""
        if error_class == "abort":
            return False
        if total_attempts is not None and total_attempts >= self.max_total_attempts:
            return False
        return attempt < self.max_attempts.get(error_class, self.max_attempts.get("unknown", 2))

    def delay(self, attempt: int, error_class: str = "unknown") -> float:
        """第 attempt 次失败后等待多久再试：0.5, 1, 2, 4, 4 ...；popup/window_lost 处理完马上重试"""
        if error_class in ("popup", "window_lost", "abort"):
            return 0.0
        return round(min(self.max_delay, self.base_delay * (self.factor ** max(0, attempt - 1))), 3)


class CircuitBreaker:
    """服务熔断：连续失败 threshold 次后打开（allow() = False），cooldown 秒后半开放行一次试探"""

    def __init__(self, threshold: int = 2, cooldown: float = 60.0, clock: Callable[[], float] = None):
        import time
        self.threshold = max(1, int(threshold))
        self.cooldown = cooldown
        self.clock = clock or time.time
        self.failures = 0
        self.opened_at = None
        self.trips = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self):
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> bool:
        """返回 True 表示这次失败让熔断器打开"""
        self.failures += 1
        if self.state == "half_open" or (self.opened_at is None and self.failures >= self.threshold):
            self.opened_at = self.clock()
            self.trips += 1
            return True
        return False


def _norm(s) -> str:
    return "".join(str(s or "").split()).lower()


def is_popup_title(title: str) -> bool:
    t = _norm(title)
    return bool(t) and any(_norm(h) in t for h in POPUP_HINTS)


class WindowGuard:
    """
    窗口状态守卫：记住运行开始时已有的窗口；执行每个子任务前检查
      1) 前台是不是本次运行新弹出的“通知/提示”类窗口 → 关掉
      2) 前台是不是目标程序 → 不是就重新激活目标程序窗口
    只操作本次运行期间新出现的弹窗；用户原有窗口一律不关。
    """

    def __init__(self, app_manager, log: Callable = None):
        self.apps = app_manager
        self.log = log or (lambda *a, **k: None)
        self.baseline = set()

    @staticmethod
    def _handle(w):
        return getattr(w, "_hWnd", None) or getattr(w, "handle", None) or id(w)

    def snapshot(self):
        self.baseline = {self._handle(w) for w in self.apps.list_windows()} if self.apps is not None else set()

    def active(self):
        if self.apps is None or self.apps.windows is None:
            return None
        try:
            return self.apps.windows.active_window()
        except Exception:
            return None

    def new_popups(self) -> list:
        if self.apps is None:
            return []
        return [w for w in self.apps.list_windows()
                if self._handle(w) not in self.baseline and is_popup_title(getattr(w, "title", ""))]

    def dismiss_popups(self) -> List[str]:
        """关闭本次运行期间新出现的弹窗，返回关掉的标题"""
        closed = []
        for w in self.new_popups():
            title = getattr(w, "title", "")
            try:
                w.close()
                closed.append(title)
            except Exception:
                try:  # 关不掉就按 Esc
                    if self.apps.controller is not None:
                        self.apps.controller.press_key("esc")
                        closed.append(title + "（Esc）")
                except Exception:
                    pass
        return closed

    def is_foreground(self, keywords: List[str]) -> bool:
        a = self.active()
        if a is None or not keywords:
            return True  # 拿不到窗口信息时不做判断
        t = _norm(getattr(a, "title", ""))
        return any(_norm(k) in t for k in keywords if _norm(k))

    def refocus(self, keywords: List[str]) -> Optional[str]:
        """把标题含 keywords 的窗口切到前台，返回窗口标题；找不到返回 None"""
        if self.apps is None or not keywords:
            return None
        wins = self.apps.find_windows(keywords)
        if not wins:
            return None
        # 优先本次运行新开的窗口
        wins.sort(key=lambda w: self._handle(w) in self.baseline)
        w = wins[0]
        self.apps._activate(w)
        try:
            w.activate()
        except Exception:
            pass
        return getattr(w, "title", "")

    def ensure(self, keywords: Optional[List[str]]) -> List[dict]:
        """执行前的环境检查，返回做过的恢复动作列表"""
        done = []
        if self.apps is None:
            return done
        closed = self.dismiss_popups()
        if closed:
            done.append({"action": "dismiss_popup", "result": closed})
        if keywords and not self.is_foreground(keywords):
            a = self.active()
            title = self.refocus(keywords)
            done.append({"action": "refocus", "from": getattr(a, "title", None), "result": title or "未找到目标窗口"})
        return done


def recovery_actions(error_class: str) -> List[str]:
    """每类错误在下一次尝试前要做的恢复动作（顺序执行）"""
    return {
        "target_not_found": ["dismiss_popup", "refocus", "backoff"],
        "popup": ["dismiss_popup", "refocus"],
        "window_lost": ["dismiss_popup", "refocus"],
        "no_effect": ["dismiss_popup", "refocus", "backoff"],
        "perception": ["backoff"],
        "app_launch": ["backoff"],
        "service": ["degrade"],
        "invalid_action": [],
        "execution": ["backoff"],
        "postcondition": ["dismiss_popup", "refocus", "backoff"],
        "unknown": ["backoff"],
        "abort": [],
    }.get(error_class, ["backoff"])
