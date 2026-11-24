"""
任务规划（第4周新增）：规则模板 + LLM 规划的混合规划器

第3周只用 LLM 规划，qwen3:8b 对“打开浏览器搜索X”会拆出“打开默认浏览器 / 在地址栏输入…”这类
不同写法的子任务，每次都不一样，后面的 Acting 也跟着不稳定。第4周的 5 个基础任务有固定套路，
先用规则模板匹配，匹配成功时直接给出“可被规则 100% 解析”的标准子任务；匹配不上的指令再交给 LLM。

  模式 auto：先模板，匹配失败再 LLM（默认）
  模式 rule：只用模板，匹配失败直接报错（不需要 Ollama）
  模式 llm ：只用 LLM（与第3周一致，用于对比实验）

标准子任务写法（每一条都能被 agent/action_validator.py 的 SubtaskIntent 唯一解析成一个动作）：
  打开X / 打开文件“路径” -> open_app      关闭X -> close_app        等待N秒 -> wait
  按 ctrl+l …           -> hotkey        按回车 -> key             点击“X” -> click
  在“X”框中输入“Y”       -> 点击 X 后 type   直接输入“Y”并按回车 -> 向当前焦点 type + enter
"""
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

_Q = "[“”\"'‘’「」『』]"
_SPLIT_RE = re.compile(r"\s*(?:，\s*然后|,\s*然后|然后|，\s*再|；|;|，\s*接着|接着)\s*")

DEFAULT_CHAT_APP = "模拟聊天"


@dataclass
class Plan:
    subtasks: List[str]
    source: str                    # template:<task_id> / llm / empty
    task_ids: List[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return {"subtasks": self.subtasks, "source": self.source, "task_ids": self.task_ids, "params": self.params}


def _strip_quotes(s: str) -> str:
    return re.sub(rf"^{_Q}+|{_Q}+$", "", (s or "").strip()).strip()


class RulePlanner:
    """5 个基础任务（+ 通用打开/关闭程序）的规则模板"""

    SEARCH_RE = re.compile(
        rf"^(?:请)?(?:打开浏览器(?:并|然后|，|,)?\s*)?(?:在|用)?(?:默认)?(?:浏览器|必应|bing|网上)?(?:中|里|上)?\s*"
        rf"(?:搜索|搜一下|搜|查找|查询|查一下)(?:一下)?\s*[:：]?\s*(?P<q>.+?)\s*$", re.IGNORECASE)
    OPEN_BROWSER_RE = re.compile(r"^(?:请)?(?:帮我)?(?:打开|启动|运行)(?:一下)?(?:默认的?)?(?:浏览器|网页浏览器)$")
    MESSAGE_RES = [
        re.compile(rf"^(?:请)?(?:(?:用|在)(?P<app>.+?)(?:里|中|上)?)?\s*(?:给|向)\s*(?P<to>[^\s，,：:{_Q[1:-1]}]+?)\s*"
                   rf"发(?:送)?(?:一条|个)?(?:消息|信息|微信)?\s*(?:说)?\s*[:：，,]?\s*(?P<msg>.+?)\s*$"),
        re.compile(rf"^(?:请)?(?:(?:用|在)(?P<app>.+?)(?:里|中|上)?)?\s*发(?:送)?(?:一条)?(?:消息|信息)\s*"
                   rf"[“\"']?(?P<msg>[^”\"']+?)[”\"']?\s*给\s*(?P<to>[^\s，,：:]+?)\s*$"),
        re.compile(rf"^(?:请)?(?:(?:用|在)(?P<app>.+?)(?:里|中|上)?)?\s*发(?:送)?(?:消息|信息)给\s*(?P<to>[^\s，,：:]+?)\s*"
                   rf"[:：，,]\s*(?P<msg>.+?)\s*$"),
    ]
    OPEN_FILE_RE = re.compile(
        rf"^(?:请)?(?:帮我)?(?:打开|双击打开|查看)(?:一下)?(?:文件|文档)?\s*{_Q}?(?P<path>(?:[A-Za-z]:[\\/]|~[\\/]|%\w+%)?"
        rf"[^“”\"'‘’「」]*?\.[A-Za-z0-9]{{1,5}}){_Q}?\s*(?:文件)?$")
    CLOSE_RE = re.compile(rf"^(?:请)?(?:帮我)?(?:关闭|关掉|退出)(?:一下)?\s*{_Q}?(?P<app>[^“”\"'‘’「」]+?){_Q}?\s*$")
    OPEN_APP_RE = re.compile(rf"^(?:请)?(?:帮我)?(?:打开|启动|运行)(?:一下)?\s*{_Q}?(?P<app>[^“”\"'‘’「」]+?){_Q}?\s*$")

    def __init__(self, chat_app: str = DEFAULT_CHAT_APP, app_resolver: Optional[Callable[[str], Optional[str]]] = None,
                 open_wait: float = 2.0):
        """app_resolver: 判断“打开X”里的 X 是否是已知程序（通常传 AppManager.resolve_app）"""
        self.chat_app = chat_app
        self.app_resolver = app_resolver
        self.open_wait = open_wait

    # ---- 单个任务模板 ----
    def _w(self, sec=None):
        return f"等待{(self.open_wait if sec is None else sec):g}秒"

    def browser(self):
        return ["打开浏览器", self._w()]

    def search(self, query: str):
        return self.browser() + ["按 ctrl+l 聚焦地址栏", f"直接输入“{query}”并按回车", self._w()]

    def open_file(self, path: str):
        return [f"打开文件“{path}”", self._w()]

    def send_message(self, to: str, msg: str, app: Optional[str] = None):
        app = app or self.chat_app
        if "微信" in app or "wechat" in app.lower():
            return ["打开微信", self._w(3), "按 ctrl+f 搜索联系人", f"直接输入“{to}”", self._w(2), "按回车",
                    f"直接输入“{msg}”并按回车"]
        return [f"打开{app}", self._w(), f"点击联系人“{to}”", f"在“消息内容”框中输入“{msg}”", "点击“发送”", self._w(1)]

    def close_app(self, app: str):
        return [f"关闭{app}", self._w(1)]

    def open_app(self, app: str):
        return [f"打开{app}", self._w()]

    # ---- 匹配 ----
    def match_one(self, text: str) -> Optional[Plan]:
        t = (text or "").strip().rstrip("。.!！")
        if not t:
            return None
        if self.OPEN_BROWSER_RE.match(t):
            return Plan(self.browser(), "template:open_browser", ["open_browser"])
        for rx in self.MESSAGE_RES:
            m = rx.match(t)
            if m:
                to, msg = _strip_quotes(m.group("to")), _strip_quotes(m.group("msg"))
                if to and msg and msg not in ("消息", "信息", "微信", "一条消息", "条消息", "息"):
                    app = _strip_quotes(m.group("app") or "") or None
                    return Plan(self.send_message(to, msg, app), "template:send_message", ["send_message"],
                                {"to": to, "message": msg, "app": app or self.chat_app})
        m = self.SEARCH_RE.match(t)
        if m and _strip_quotes(m.group("q")):
            q = _strip_quotes(m.group("q"))
            return Plan(self.search(q), "template:search", ["search"], {"query": q})
        m = self.OPEN_FILE_RE.match(t)
        if m:
            path = _strip_quotes(m.group("path"))
            return Plan(self.open_file(path), "template:open_file", ["open_file"], {"path": path})
        m = self.CLOSE_RE.match(t)
        if m:
            app = _strip_quotes(m.group("app"))
            return Plan(self.close_app(app), "template:close_app", ["close_app"], {"app": app})
        m = self.OPEN_APP_RE.match(t)
        if m:
            app = _strip_quotes(m.group("app"))
            if self.app_resolver is None or self.app_resolver(app):
                return Plan(self.open_app(app), "template:open_app", ["open_app"], {"app": app})
        return None

    def match(self, instruction: str) -> Optional[Plan]:
        """含“然后/；/接着”时先拆成几段，每段都匹配上才算成功；否则整句匹配"""
        parts = [p for p in _SPLIT_RE.split((instruction or "").strip()) if p.strip()]
        if len(parts) >= 2:
            plans = [self.match_one(p) for p in parts]
            if all(plans):
                subtasks, ids, params = [], [], {}
                for i, p in enumerate(plans):
                    subtasks += p.subtasks
                    ids += p.task_ids
                    params.update({f"{k}_{i + 1}": v for k, v in p.params.items()})
                return Plan(subtasks, "template:" + "+".join(ids), ids, params)
        return self.match_one(instruction)


class HybridPlanner:
    MODES = ("auto", "rule", "llm")

    def __init__(self, mode: str = "auto", rule: Optional[RulePlanner] = None, llm_plan: Optional[Callable] = None):
        """llm_plan: callable(instruction) -> List[str]，通常是 BaseGUIAgent.plan"""
        if mode not in self.MODES:
            raise ValueError(f"不支持的规划模式 {mode!r}，可选 {self.MODES}")
        if mode == "llm" and llm_plan is None:
            raise ValueError("llm 规划模式需要提供 llm_plan（启用 LLM）")
        self.mode = mode
        self.rule = rule or RulePlanner()
        self.llm_plan = llm_plan

    def plan(self, instruction: str) -> Plan:
        if not (instruction or "").strip():
            return Plan([], "empty")
        if self.mode in ("auto", "rule"):
            p = self.rule.match(instruction)
            if p:
                return p
            if self.mode == "rule" or self.llm_plan is None:
                raise ValueError(f"规则模板无法识别指令「{instruction}」，请启用 LLM 规划（--planner auto/llm）")
        subtasks = self.llm_plan(instruction)
        return Plan(list(subtasks or []), "llm")
