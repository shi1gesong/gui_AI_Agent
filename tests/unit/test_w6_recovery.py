"""第6周：agent/recovery.py 错误分类 / 重试策略 / 熔断器 / 窗口守卫"""
import pytest

from agent.recovery import (ERROR_CLASSES, CircuitBreaker, RetryPolicy, WindowGuard, classify_error, is_popup_title,
                            recovery_actions)
from control.app_manager import AppManager
from fakes import FakeWindowProvider, RecordingController


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


CLASSIFY_CASES = [
    ("LookupError: 屏幕上找不到点击目标「发送」的坐标", "target_not_found"),
    ("LookupError: 屏幕上找不到输入框「消息内容」，未输入", "target_not_found"),
    ("RuntimeError: 3 秒内没有出现标题含 ['GUI Agent 模拟聊天'] 的窗口", "app_launch"),
    ("界面没有变化（变化像素 0.000% < 0.200%），动作没有生效", "no_effect"),
    ("决策失败: ConnectionError: Ollama 连接失败: Connection refused", "service"),
    ("决策失败: ReadTimeout: HTTPConnectionPool read timed out", "service"),
    ("VLM 服务报错（HTTP 500）: CUDA OOM", "service"),
    ("决策失败: OSError: 截图失败: 模拟 mss 抓屏错误", "perception"),
    ("FailSafeException: PyAutoGUI fail-safe triggered", "abort"),
    ("动作未通过校验，未执行", "invalid_action"),
    ("ValueError: 无法识别的组合键「ctrl+」", "execution"),
    ("结果检查不通过：屏幕上出现「张三：你好」", "postcondition"),
    ("前台窗口不是目标程序", "window_lost"),
    ("检测到弹窗 popup", "popup"),
    ("something completely different", "unknown"),
]


@pytest.mark.functional
class TestClassify:
    @pytest.mark.parametrize("text,cls", CLASSIFY_CASES)
    def test_classify_text(self, text, cls):
        assert classify_error(text) == cls

    def test_classify_exception_objects(self):
        assert classify_error(ConnectionError("refused")) == "service"
        assert classify_error(LookupError("屏幕上找不到「张三」")) == "target_not_found"
        assert classify_error(KeyboardInterrupt()) == "abort"

    def test_failsafe_exception_by_name(self):
        class FailSafeException(Exception):
            pass
        assert classify_error(FailSafeException("corner")) == "abort"

    def test_invalid_retryable_action_is_target_not_found(self):
        a = {"_invalid": True, "_retryable": True, "_reasons": ["屏幕上没有找到 ['张三']"]}
        assert classify_error("动作未通过校验，未执行", a) == "target_not_found"

    def test_invalid_rule_action(self):
        a = {"_invalid": True, "_reasons": ["规则无法推断动作，且未启用 LLM"]}
        assert classify_error("动作未通过校验，未执行", a) == "invalid_action"

    def test_all_classes_have_recovery_entry(self):
        for c in ERROR_CLASSES:
            assert isinstance(recovery_actions(c), list)

    def test_recovery_actions_content(self):
        assert recovery_actions("popup") == ["dismiss_popup", "refocus"]
        assert "backoff" in recovery_actions("target_not_found")
        assert recovery_actions("abort") == []
        assert recovery_actions("service") == ["degrade"]


@pytest.mark.functional
class TestPolicyAndBreaker:
    @pytest.mark.parametrize("attempt,delay", [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 4.0)])
    def test_exponential_backoff_capped(self, attempt, delay):
        assert RetryPolicy().delay(attempt, "target_not_found") == delay

    @pytest.mark.parametrize("cls", ["popup", "window_lost", "abort"])
    def test_no_delay_for_immediate_recoveries(self, cls):
        assert RetryPolicy().delay(1, cls) == 0.0

    def test_allowed_per_class(self):
        p = RetryPolicy()
        assert p.allowed("target_not_found", 3) and not p.allowed("target_not_found", 4)
        assert p.allowed("no_effect", 1) and not p.allowed("no_effect", 2)

    def test_abort_never_retried(self):
        assert not RetryPolicy().allowed("abort", 0)

    def test_total_attempt_cap(self):
        p = RetryPolicy(max_total_attempts=3)
        assert not p.allowed("target_not_found", 1, total_attempts=3)

    def test_breaker_opens_after_threshold(self):
        b = CircuitBreaker(threshold=2, cooldown=10, clock=Clock())
        assert b.allow() and not b.record_failure()
        assert b.record_failure() and b.state == "open" and not b.allow()

    def test_breaker_half_open_after_cooldown(self):
        clk = Clock()
        b = CircuitBreaker(threshold=1, cooldown=10, clock=clk)
        b.record_failure()
        clk.t = 11
        assert b.state == "half_open" and b.allow()
        assert b.record_failure() and b.state == "open" and b.trips == 2

    def test_breaker_success_resets(self):
        b = CircuitBreaker(threshold=2, clock=Clock())
        b.record_failure()
        b.record_success()
        assert b.failures == 0 and b.state == "closed"

    @pytest.mark.parametrize("title,expected", [
        ("系统通知", True), ("更新提示", True), ("Windows 安全警告", True), ("Error", True),
        ("GUI Agent 模拟聊天", False), ("必应 - Microsoft Edge", False), ("", False)])
    def test_popup_title(self, title, expected):
        assert is_popup_title(title) is expected


def guard_env(*titles):
    wp = FakeWindowProvider(titles)
    ctl = RecordingController()
    apps = AppManager(windows=wp, controller=ctl, sleep=lambda s: None)
    return WindowGuard(apps), wp, ctl


@pytest.mark.functional
class TestWindowGuard:
    def test_dismiss_only_new_popups(self):
        g, wp, _ = guard_env("系统通知", "GUI Agent 模拟聊天")   # 运行前就有的“系统通知”不碰
        g.snapshot()
        new = wp.add("系统通知")
        closed = g.dismiss_popups()
        assert closed == ["系统通知"] and new not in wp.wins and len(wp.wins) == 2

    def test_non_popup_new_window_not_closed(self):
        g, wp, _ = guard_env("GUI Agent 模拟聊天")
        g.snapshot()
        wp.add("必应 - Microsoft Edge")
        assert g.dismiss_popups() == [] and len(wp.wins) == 2

    def test_refocus_target(self):
        g, wp, _ = guard_env("GUI Agent 模拟聊天", "文档 - 文件资源管理器")
        g.snapshot()
        assert not g.is_foreground(["模拟聊天"])
        assert g.refocus(["模拟聊天"]) == "GUI Agent 模拟聊天"
        assert g.is_foreground(["模拟聊天"])

    def test_ensure_dismiss_then_refocus(self):
        g, wp, _ = guard_env("GUI Agent 模拟聊天")
        g.snapshot()
        wp.add("系统通知")
        done = g.ensure(["模拟聊天"])
        assert [d["action"] for d in done] == ["dismiss_popup", "refocus"]
        assert wp.active.title == "GUI Agent 模拟聊天"

    def test_ensure_noop_when_ok(self):
        g, wp, _ = guard_env("GUI Agent 模拟聊天")
        g.snapshot()
        assert g.ensure(["模拟聊天"]) == []


@pytest.mark.boundary
class TestRecoveryBoundary:
    def test_classify_none_and_empty(self):
        assert classify_error(None) == "unknown" and classify_error("") == "unknown"

    def test_policy_unknown_class_uses_default(self):
        assert RetryPolicy().allowed("brand_new_class", 1) and not RetryPolicy().allowed("brand_new_class", 2)

    def test_breaker_threshold_min_one(self):
        b = CircuitBreaker(threshold=0, clock=Clock())
        assert b.threshold == 1 and b.record_failure()

    def test_guard_without_app_manager(self):
        g = WindowGuard(None)
        g.snapshot()
        assert g.ensure(["x"]) == [] and g.new_popups() == [] and g.refocus(["x"]) is None

    def test_is_foreground_without_keywords(self):
        g, _, _ = guard_env("a")
        assert g.is_foreground([])

    def test_refocus_missing_window(self):
        g, _, _ = guard_env("a")
        assert g.refocus(["不存在"]) is None


@pytest.mark.exception
class TestRecoveryException:
    def test_popup_close_failure_falls_back_to_esc(self):
        g, wp, ctl = guard_env()
        g.snapshot()
        w = wp.add("系统通知")

        def boom():
            raise OSError("拒绝访问")
        w.close = boom
        closed = g.dismiss_popups()
        assert closed == ["系统通知（Esc）"] and ("press_key", "esc") in ctl.calls

    def test_active_window_exception(self):
        g, wp, _ = guard_env("x")

        def boom():
            raise RuntimeError("pygetwindow 错误")
        wp.active_window = boom
        assert g.active() is None and g.is_foreground(["x"])

    def test_classify_non_string_error(self):
        assert classify_error(12345) == "unknown"
