"""第4周：agent/planner.py（基础任务规则模板 + LLM 混合规划）—— mock 用例"""
import pytest

from action_validator import fallback_action
from planner import HybridPlanner, RulePlanner

R = RulePlanner()


@pytest.mark.functional
class TestPlannerFunctional:
    @pytest.mark.parametrize("text", ["打开浏览器", "打开默认浏览器", "启动浏览器", "请打开浏览器", "帮我打开浏览器。"])
    def test_t1_open_browser(self, text):
        p = R.match(text)
        assert p.source == "template:open_browser" and p.subtasks[0] == "打开浏览器"

    @pytest.mark.parametrize("text,q", [("在浏览器中搜索天气预报", "天气预报"), ("搜索天气预报", "天气预报"),
                                        ("搜索“GUI Agent”", "GUI Agent"), ("打开浏览器搜索北京天气", "北京天气"),
                                        ("在浏览器里搜一下 Python 教程", "Python 教程"), ("用必应搜索：大模型", "大模型"),
                                        ("查一下明天的天气", "明天的天气")])
    def test_t2_search(self, text, q):
        p = R.match(text)
        assert p.source == "template:search" and p.params["query"] == q
        assert f"直接输入“{q}”并按回车" in p.subtasks and "按 ctrl+l 聚焦地址栏" in p.subtasks

    @pytest.mark.parametrize("text,path", [("打开文件 C:\\Users\\a\\demo.txt", "C:\\Users\\a\\demo.txt"),
                                           ("打开 report.docx", "report.docx"), ("打开文件“周报.xlsx”", "周报.xlsx"),
                                           ("打开 D:\\资料\\第4周 报告.pdf", "D:\\资料\\第4周 报告.pdf"),
                                           ("查看文件 notes.md", "notes.md")])
    def test_t3_open_file(self, text, path):
        p = R.match(text)
        assert p.source == "template:open_file" and p.params["path"] == path
        assert p.subtasks[0] == f"打开文件“{path}”"

    @pytest.mark.parametrize("text,to,msg", [("给张三发送消息：今晚7点开会", "张三", "今晚7点开会"),
                                             ("给李四发消息“明天见”", "李四", "明天见"),
                                             ("向王五发送：周报已提交", "王五", "周报已提交"),
                                             ("发消息给张三：你好", "张三", "你好"),
                                             ("发送消息“收到”给李四", "李四", "收到"),
                                             ("给文件传输助手发一条消息：测试", "文件传输助手", "测试")])
    def test_t4_send_message(self, text, to, msg):
        p = R.match(text)
        assert p.source == "template:send_message" and (p.params["to"], p.params["message"]) == (to, msg)
        assert f"点击联系人“{to}”" in p.subtasks and f"在“消息内容”框中输入“{msg}”" in p.subtasks

    @pytest.mark.parametrize("text,app", [("关闭计算器", "计算器"), ("关掉记事本", "记事本"), ("退出模拟聊天", "模拟聊天"),
                                          ("关闭“画图”", "画图")])
    def test_t5_close_app(self, text, app):
        p = R.match(text)
        assert p.source == "template:close_app" and p.subtasks[0] == f"关闭{app}"

    def test_open_known_app(self):
        assert RulePlanner(app_resolver=lambda n: "计算器" if n == "计算器" else None).match("打开计算器").subtasks[0] \
            == "打开计算器"

    def test_compound_instruction(self):
        p = R.match("打开记事本，然后关闭记事本")
        assert p.task_ids == ["open_app", "close_app"] and p.subtasks[0] == "打开记事本" and "关闭记事本" in p.subtasks

    def test_wechat_template(self):
        p = R.match("用微信给张三发消息：hi")
        assert p.params["app"] == "微信" and p.subtasks[0] == "打开微信" and "按 ctrl+f 搜索联系人" in p.subtasks

    def test_custom_chat_app(self):
        assert RulePlanner(chat_app="模拟聊天").match("给张三发消息：hi").subtasks[0] == "打开模拟聊天"

    @pytest.mark.parametrize("text", ["打开浏览器", "在浏览器中搜索天气预报", "打开文件 C:\\a\\demo.txt",
                                      "给张三发送消息：今晚7点开会", "关闭计算器"])
    def test_every_template_subtask_is_rule_executable(self, text):
        """模板产出的每条子任务都必须能被规则解析成动作（否则 hybrid 模式会退化为调用 LLM）"""
        screen = [{"text": "张三", "bbox": [0, 0, 10, 10]}, {"text": "消息内容", "bbox": [0, 20, 10, 30]},
                  {"text": "发送", "bbox": [0, 40, 10, 50]}]
        for st in R.match(text).subtasks:
            assert fallback_action(st, screen) is not None, st

    def test_hybrid_uses_template_first(self):
        called = []
        hp = HybridPlanner("auto", llm_plan=lambda i: called.append(i) or ["x"])
        assert hp.plan("打开浏览器").source == "template:open_browser" and called == []

    def test_hybrid_falls_back_to_llm(self):
        hp = HybridPlanner("auto", llm_plan=lambda i: ["打开设置", "点击“蓝牙”"])
        p = hp.plan("把蓝牙关掉吧")
        assert p.source == "llm" and p.subtasks == ["打开设置", "点击“蓝牙”"]

    def test_llm_mode_skips_template(self):
        hp = HybridPlanner("llm", llm_plan=lambda i: ["a"])
        assert hp.plan("打开浏览器").source == "llm"

    def test_plan_to_dict(self):
        d = R.match("搜索天气").to_dict()
        assert set(d) == {"subtasks", "source", "task_ids", "params"}


@pytest.mark.boundary
class TestPlannerBoundary:
    def test_empty_instruction(self):
        assert R.match("") is None and HybridPlanner("rule").plan("  ").source == "empty"

    def test_unknown_instruction_no_template(self):
        assert R.match("今天天气怎么样") is None

    def test_open_unknown_app_rejected_with_resolver(self):
        assert RulePlanner(app_resolver=lambda n: None).match("打开一个不存在的程序") is None

    def test_trailing_punctuation(self):
        assert R.match("关闭计算器。").params["app"] == "计算器"

    def test_search_with_quotes_stripped(self):
        assert R.match("搜索「人工智能」").params["query"] == "人工智能"

    def test_open_wait_configurable(self):
        assert RulePlanner(open_wait=5).match("打开浏览器").subtasks[1] == "等待5秒"

    def test_compound_partially_unknown_falls_back(self):
        assert R.match("打开浏览器然后帮我写一首诗") is None or R.match("打开浏览器然后帮我写一首诗").source != \
            "template:open_browser+unknown"

    def test_message_with_comma_in_content(self):
        p = R.match("给张三发送消息：好的，明天见")
        assert p.params["message"] == "好的，明天见"


@pytest.mark.exception
class TestPlannerException:
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            HybridPlanner("magic")

    def test_llm_mode_without_llm(self):
        with pytest.raises(ValueError):
            HybridPlanner("llm")

    def test_rule_mode_unknown_instruction_raises(self):
        with pytest.raises(ValueError, match="规则模板无法识别"):
            HybridPlanner("rule").plan("今天天气怎么样")

    def test_auto_without_llm_unknown_raises(self):
        with pytest.raises(ValueError):
            HybridPlanner("auto", llm_plan=None).plan("今天天气怎么样")

    def test_llm_returns_none(self):
        assert HybridPlanner("llm", llm_plan=lambda i: None).plan("x").subtasks == []

    def test_llm_exception_propagates(self):
        def boom(i):
            raise ConnectionError("Ollama 未启动")
        with pytest.raises(ConnectionError):
            HybridPlanner("auto", llm_plan=boom).plan("今天天气怎么样")

    def test_message_without_content_not_matched(self):
        p = R.match("给张三发送消息")
        assert p is None or p.source != "template:send_message"


@pytest.mark.multi_resolution
class TestPlannerMultiResolution:
    def test_plan_independent_of_screen(self):
        """规划只依赖指令文本，不依赖屏幕分辨率：同一指令多次规划结果一致"""
        assert R.match("在浏览器中搜索天气预报").subtasks == R.match("在浏览器中搜索天气预报").subtasks
