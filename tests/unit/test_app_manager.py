"""第4周：control/app_manager.py（open_app / close_app 的程序启动与窗口管理）—— mock 用例，不会真的打开程序"""
import sys

import pytest

from control.app_manager import MOCK_CHAT_TITLE, AppManager
from fakes import FakeLauncher, FakeWindowProvider, RecordingController

LAUNCH = {"bing.com": "必应 - Microsoft Edge", "notepad.exe": "无标题 - 记事本", "calc.exe": "计算器",
          "mock_chat.py": MOCK_CHAT_TITLE, "demo.txt": "demo.txt - 记事本"}


@pytest.fixture
def env(tmp_path):
    wins = FakeWindowProvider(launch_map=LAUNCH)
    launcher = FakeLauncher(wins)
    ctl = RecordingController()
    mgr = AppManager(spawn=launcher.spawn, startfile=launcher.startfile, windows=wins, controller=ctl,
                     sleep=lambda s: None, search_paths=[tmp_path], open_timeout=1.0, close_timeout=0.5)
    return mgr, wins, launcher, ctl


@pytest.mark.functional
class TestAppManagerFunctional:
    @pytest.mark.parametrize("name,key", [("浏览器", "浏览器"), ("默认浏览器", "浏览器"), ("Edge", "浏览器"),
                                          ("chrome", "浏览器"), ("谷歌浏览器", "浏览器"), ("记事本", "记事本"),
                                          ("Notepad", "记事本"), ("计算器", "计算器"), ("calc", "计算器"),
                                          ("模拟聊天", "模拟聊天"), ("聊天应用", "模拟聊天"), ("微信", "微信"),
                                          ("文件资源管理器", "文件资源管理器"), ("此电脑", "文件资源管理器")])
    def test_resolve_app(self, env, name, key):
        assert env[0].resolve_app(name) == key

    def test_open_browser_uses_default_browser(self, env):
        mgr, wins, launcher, _ = env
        r = mgr.open("浏览器")
        assert r["ok"] and launcher.started == ["https://www.bing.com"] and "Edge" in r["window"]

    def test_open_notepad_spawns_exe(self, env):
        mgr, _, launcher, _ = env
        assert mgr.open("记事本")["ok"] and launcher.spawned == [["notepad.exe"]]

    def test_open_mock_chat_uses_python_with_log(self, env):
        mgr, _, launcher, _ = env
        mgr.mock_chat_log = "chat.jsonl"
        assert mgr.open("模拟聊天")["ok"]
        cmd = launcher.spawned[0]
        assert cmd[0] == sys.executable and cmd[1].endswith("mock_chat.py") and cmd[-2:] == ["--log", "chat.jsonl"]

    def test_open_file_with_default_program(self, env, tmp_path):
        mgr, _, launcher, _ = env
        f = tmp_path / "demo.txt"
        f.write_text("x")
        r = mgr.open(str(f))
        assert r["ok"] and r["method"] == "file" and launcher.started == [str(f)]

    def test_open_relative_file_from_search_path(self, env, tmp_path):
        (tmp_path / "demo.txt").write_text("x")
        assert env[0].open("demo.txt")["ok"]

    def test_open_activates_window(self, env):
        mgr, wins, *_ = env
        mgr.open("计算器")
        assert wins.wins[-1].activated == 1

    def test_unknown_app_uses_win_search(self, env):
        mgr, wins, _, ctl = env
        wins.launch_map = {}
        mgr.windows = None
        r = mgr.open("QQ音乐")
        assert r["ok"] and r["method"] == "search"
        assert ctl.calls == [("press_key", "win"), ("type_text", "QQ音乐"), ("press_key", "enter")]

    def test_close_by_name(self, env):
        mgr, wins, *_ = env
        wins.add("计算器")
        wins.add("无标题 - 记事本")
        r = mgr.close("计算器")
        assert r["ok"] and wins.titles() == ["无标题 - 记事本"]

    def test_close_only_matching_windows(self, env):
        mgr, wins, *_ = env
        for t in ("计算器", "我的文档.txt - 记事本", "必应 - Microsoft Edge"):
            wins.add(t)
        mgr.close("记事本")
        assert wins.titles() == ["计算器", "必应 - Microsoft Edge"]

    def test_close_file_window(self, env, tmp_path):
        mgr, wins, *_ = env
        (tmp_path / "demo.txt").write_text("x")
        wins.add("demo.txt - 记事本")
        assert mgr.close("demo.txt")["ok"] and wins.titles() == []

    def test_close_current_window(self, env):
        mgr, wins, *_ = env
        wins.add("其他")
        wins.add("前台窗口")
        assert mgr.close("当前窗口")["ok"] and wins.titles() == ["其他"]

    def test_close_english_title(self, env):
        mgr, wins, *_ = env
        wins.add("Calculator")
        assert mgr.close("计算器")["ok"]

    def test_active_title(self, env):
        mgr, wins, *_ = env
        wins.add("必应 - Microsoft Edge")
        assert mgr.active_title() == "必应 - Microsoft Edge"

    def test_title_keywords_for_file(self, env, tmp_path):
        (tmp_path / "周报.docx").write_text("x")
        assert env[0].title_keywords("周报.docx") == ["周报.docx", "周报"]

    def test_open_existing_window_reused(self, env):
        """浏览器已打开时（新标签页不产生新窗口），等待一会后返回已有窗口"""
        mgr, wins, launcher, _ = env
        wins.add("新标签页 - Microsoft Edge")
        wins.launch_map = {}
        r = mgr.open("浏览器")
        assert r["ok"] and r["window"] == "新标签页 - Microsoft Edge"


@pytest.mark.boundary
class TestAppManagerBoundary:
    def test_resolve_empty(self, env):
        assert env[0].resolve_app("") is None and env[0].resolve_file("") is None

    def test_path_never_resolved_as_app(self, env):
        """路径里含 edge / calc 等字样时不能被当成程序名（C:\\Users\\knowledge\\a.txt）"""
        assert env[0].resolve_app("C:\\Users\\knowledge\\calc.txt") is None

    def test_notepad_exe_is_app_not_file(self, env):
        assert env[0].resolve_app("notepad.exe") == "记事本"

    def test_open_without_wait(self, env):
        mgr, wins, *_ = env
        wins.launch_map = {}
        assert mgr.open("记事本", wait=False)["ok"]

    def test_quoted_file_name(self, env, tmp_path):
        (tmp_path / "a b.txt").write_text("x")
        assert env[0].resolve_file("“a b.txt”") == tmp_path / "a b.txt"

    def test_close_waits_until_gone(self, tmp_path):
        """窗口延迟关闭：轮询直到消失"""
        wins = FakeWindowProvider(["计算器"])
        w = wins.wins[0]
        w.closable = False
        ticks = []

        def sleep(s):
            ticks.append(s)
            if len(ticks) == 2:
                wins.wins.clear()

        mgr = AppManager(windows=wins, sleep=sleep, search_paths=[], close_timeout=5, poll=0.25)
        assert mgr.close("计算器")["ok"] and len(ticks) == 2

    def test_list_windows_without_provider(self):
        assert AppManager(windows=None).list_windows() == [] and AppManager(windows=None).active_title() == ""


@pytest.mark.exception
class TestAppManagerException:
    def test_open_empty_name(self, env):
        assert env[0].open("")["ok"] is False

    def test_open_missing_file(self, env):
        r = env[0].open("C:\\不存在\\x.txt")
        assert r["ok"] is False and "文件不存在" in r["error"]

    def test_window_never_appears(self, env):
        mgr, wins, *_ = env
        wins.launch_map = {}
        r = mgr.open("计算器")
        assert r["ok"] is False and "没有出现" in r["error"]

    def test_spawn_failure_reported(self, tmp_path):
        launcher = FakeLauncher(fail=FileNotFoundError("notepad.exe"))
        mgr = AppManager(spawn=launcher.spawn, startfile=launcher.startfile, windows=FakeWindowProvider(),
                         sleep=lambda s: None, search_paths=[])
        r = mgr.open("记事本")
        assert r["ok"] is False and "FileNotFoundError" in r["error"]

    def test_close_no_matching_window(self, env):
        r = env[0].close("计算器")
        assert r["ok"] is False and "没有找到" in r["error"]

    def test_close_window_stays_open(self, env):
        """程序弹出“是否保存”对话框，窗口没有关掉"""
        mgr, wins, *_ = env
        wins.add("未保存 - 记事本", closable=False)
        r = mgr.close("记事本")
        assert r["ok"] is False and "仍未关闭" in r["error"]

    def test_close_exception(self, env):
        mgr, wins, *_ = env
        w = wins.add("计算器")
        w.close = lambda: (_ for _ in ()).throw(OSError("拒绝访问"))
        r = mgr.close("计算器")
        assert r["ok"] is False and "拒绝访问" in r["error"]

    def test_unknown_app_without_controller(self, tmp_path):
        r = AppManager(windows=None, controller=None, search_paths=[]).open("QQ音乐")
        assert r["ok"] is False and "开始菜单" in r["error"]

    def test_close_without_provider_uses_alt_f4(self):
        ctl = RecordingController()
        assert AppManager(windows=None, controller=ctl).close("计算器")["ok"] and ctl.calls == [("hotkey", "alt", "f4")]

    def test_close_without_provider_or_controller(self):
        assert AppManager(windows=None).close("计算器")["ok"] is False

    def test_provider_exception_treated_as_no_windows(self):
        class Broken:
            def all_windows(self):
                raise OSError("枚举失败")

            def active_window(self):
                raise OSError("x")
        mgr = AppManager(windows=Broken())
        assert mgr.list_windows() == [] and mgr.active_title() == ""


@pytest.mark.multi_resolution
class TestAppManagerMultiResolution:
    def test_window_lookup_independent_of_geometry(self, env):
        """窗口查找只依赖标题，与分辨率/DPI 无关"""
        mgr, wins, *_ = env
        w = wins.add("计算器")
        w.left, w.top, w.width, w.height = 3000, 100, 400, 600  # 第二块屏幕上
        assert mgr.find_windows(["计算器"]) == [w]
