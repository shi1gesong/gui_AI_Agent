"""
复杂任务的自动拆解（第6周新增）

第4周的 RulePlanner 只认 5 个基础任务的整句模板，外加“A然后B”这种两段都能整句匹配的组合。
真实指令往往是多个目标串起来、后面的目标还依赖前面的上下文，例如：

    “打开文件 D:\\a.txt，输入‘第6周周报’，然后保存，最后关闭 a.txt”
    “先在浏览器里搜索天气预报，再给张三发送消息：天气已查好”
    “1. 给张三发消息：明天10点开会 2. 给李四发消息：收到请回复”

第6周把规划升级为两层：  指令 ──► 目标（Goal）列表 ──► 每个目标的子任务

  1) 切分：保护引号内文字 → 按编号（1. / 第一步 / ①）与连接词（然后/再/接着/之后/最后/并且/；）切分 →
     逗号分隔且每段都以动词开头时继续切分（“给张三发消息：你好，明天见”这种消息里的逗号不会被切开）
  2) 逐段匹配：先用第4周 5 个基础模板，再用第6周新增的上下文相关模板：
        输入“X” / 在记事本中输入X     → 输入到“当前目标程序”（上一个目标打开的程序）
        保存                        → Ctrl+S；上下文里有打开的文件时，自动加“文件内容包含刚输入的文字”检查
        切换到X / 回到X               → switch_app（激活已打开的窗口，不重新启动）
        全选 / 复制 / 粘贴 / 按 ctrl+x  → 组合键
        访问 www.xxx.com / 打开网址     → 默认浏览器打开网址
        等待N秒 / 点击“X”             → 原样
     都匹配不上的段落才交给 LLM 拆解（只拆这一段，不是整句重来）；没有 LLM 时报出是哪一段不认识
  3) 每个目标附带：目标程序（用于执行前确认前台窗口、被弹窗打断后重新激活）、后置条件（目标完成后检查）、
     是否可重放（发送消息、输入文字不可重放，失败时不整体重做，避免重复发送）
"""
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    from .planner import Plan, RulePlanner, _strip_quotes
    from .verifier import Postcondition
except ImportError:
    from planner import Plan, RulePlanner, _strip_quotes
    from verifier import Postcondition

try:
    from control.app_manager import APP_REGISTRY
except ImportError:  # pragma: no cover
    APP_REGISTRY = {}

MAX_GOALS = 12

_OPEN_Q, _CLOSE_Q = "“\"'‘「『", "”\"'’」』"
_QUOTED_RE = re.compile(r"[“\"'‘「『][^”\"'’」』]*[”\"'’」』]")
_NUM_RE = re.compile(r"(?:^|(?<=[\s，,；;。]))(?:\d{1,2}\s*[.、)）](?![A-Za-z0-9])|[（(]\d{1,2}[)）]|第[一二三四五六七八九十\d]+步\s*[:：，,]?|[①②③④⑤⑥⑦⑧⑨⑩])\s*")
_LEAD_RE = re.compile(r"^(?:请|帮我|麻烦|先|首先|再|然后|接着|最后|并且|并)\s*")
VERBS = ("打开", "启动", "运行", "关闭", "关掉", "退出", "搜索", "搜", "查找", "查询", "给", "向", "发送", "发", "输入",
         "键入", "写入", "填写", "保存", "等待", "等", "切换", "回到", "切回", "按", "点击", "单击", "双击", "在", "用",
         "访问", "复制", "粘贴", "全选", "剪切", "撤销", "刷新", "新建", "浏览", "把", "将", "让", "使用", "选择", "选中",
         "拖动", "拖拽", "滚动", "下载", "上传", "登录", "查看", "阅读", "回复", "转发", "删除", "创建", "设置", "最小化",
         "最大化", "截图", "填", "写")
_VERB_ALT = "|".join(sorted(VERBS, key=len, reverse=True))
# 连接词后面必须紧跟动词才切分：“给张三发消息：好的，再见”“…：最后一次提醒”里的“再/最后”不会被当成连接词
_CONNECT_RE = re.compile(r"\s*(?:(?:[，,。]?\s*(?:然后|接着|之后|随后|最后|完成后|并且|再然后)|[，,]\s*再)\s*[，,]?\s*"
                         rf"(?=(?:请|先|再)?(?:{_VERB_ALT}))|\s*[；;。]\s*)")

BROWSER_KEYS = ["Edge", "Chrome", "Firefox", "浏览器", "Bing", "必应"]


@dataclass
class Goal:
    id: str
    description: str
    subtasks: List[str]
    source: str = "template"
    app: Optional[str] = None                 # 目标程序名（AppManager 能识别的名字或文件名）
    app_keywords: List[str] = field(default_factory=list)   # 窗口标题关键字，执行前确认前台窗口用
    postconditions: List[Postcondition] = field(default_factory=list)
    idempotent: bool = True                   # False：失败时不整体重做（发送消息 / 输入文字）
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return {"id": self.id, "description": self.description, "subtasks": list(self.subtasks), "source": self.source,
                "app": self.app, "app_keywords": list(self.app_keywords), "idempotent": self.idempotent,
                "postconditions": [p.to_dict() for p in self.postconditions], "params": dict(self.params)}


@dataclass
class TaskGraph:
    instruction: str
    goals: List[Goal]
    source: str

    @property
    def subtasks(self) -> List[str]:
        return [s for g in self.goals for s in g.subtasks]

    def to_plan(self) -> Plan:
        """转成第4周的 Plan（--plan-only、v1 兼容）"""
        return Plan(self.subtasks, self.source, [g.params.get("task_id", g.source) for g in self.goals],
                    {g.id: g.params for g in self.goals})

    def to_dict(self):
        return {"instruction": self.instruction, "source": self.source, "goals": [g.to_dict() for g in self.goals],
                "subtasks": self.subtasks}


# ---------------------------------------------------------------------- #
# 切分
# ---------------------------------------------------------------------- #
def mask_quotes(text: str):
    """把引号内的内容换成占位符，切分时不会被引号里的“然后/，”切开；返回 (masked, restore 函数)"""
    saved = []

    def sub(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    masked = _QUOTED_RE.sub(sub, text or "")

    def restore(s: str) -> str:
        return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], s)

    return masked, restore


def _starts_with_verb(s: str) -> bool:
    s = _LEAD_RE.sub("", s.strip())
    return s.startswith(VERBS)


def split_clauses(instruction: str) -> List[str]:
    masked, restore = mask_quotes((instruction or "").strip())
    masked = masked.strip().rstrip("。.!！")
    masked = re.sub(r"(?<=\S)\s*([①②③④⑤⑥⑦⑧⑨⑩])", r"；\1", masked)   # “①打开A②关闭A”：编号前补分隔
    parts = [p for p in _NUM_RE.split(masked) if p and p.strip()]
    out = []
    for p in parts:
        for q in _CONNECT_RE.split(p):
            q = q.strip(" ，,；;。")
            if not q:
                continue
            pieces = [x.strip() for x in re.split(r"[，,]", q) if x.strip()]
            if len(pieces) > 1 and all(_starts_with_verb(x) for x in pieces):
                out.extend(pieces)
            else:
                out.append(q)
    clauses = []
    for c in out:
        c = _LEAD_RE.sub("", restore(c).strip()).strip(" ，,；;。")
        c = _LEAD_RE.sub("", c).strip()
        if c:
            clauses.append(c)
    return clauses


# ---------------------------------------------------------------------- #
# 第6周新增的上下文相关模板
# ---------------------------------------------------------------------- #
_Q = "[“”\"'‘’「」『』]"
TYPE_RES = [
    re.compile(rf"^(?:在|往|向)(?P<app>.+?)(?:中|里|内|上)?\s*(?:输入|键入|写入|写上|填写|填入)(?:文字|内容)?\s*[:：]?\s*(?P<text>.+)$"),
    re.compile(rf"^(?:输入|键入|写入|写上|填写|填入)(?:文字|内容)?\s*[:：]?\s*(?P<text>.+)$"),
]
SAVE_RE = re.compile(r"^(?:按\s*ctrl\s*\+\s*s\s*)?(?:保存|存盘|保存文件|保存一下|保存修改)(?:文件|一下)?$", re.IGNORECASE)
SWITCH_RE = re.compile(rf"^(?:切换到|切换至|切回|回到|返回到|返回)\s*{_Q}?(?P<app>[^“”\"'‘’「」]+?){_Q}?(?:窗口)?$")
HOTKEY_WORDS = {"全选": "ctrl+a", "复制": "ctrl+c", "粘贴": "ctrl+v", "剪切": "ctrl+x", "撤销": "ctrl+z", "刷新": "f5"}
HOTKEY_RE = re.compile(r"^(?:按\s*)?(?:下\s*)?((?:ctrl|alt|shift|win)(?:\s*\+\s*[a-z0-9]+)+)", re.IGNORECASE)
KEY_RE = re.compile(r"^按(?:下)?\s*(回车|enter|tab|esc|空格|退格)(?:键)?$", re.IGNORECASE)
URL_RE = re.compile(rf"^(?:用浏览器)?(?:访问|打开网址|打开网页|浏览|打开)\s*{_Q}?(?P<url>(?:https?://|www\.)[^\s“”\"'‘’「」]+){_Q}?$",
                    re.IGNORECASE)
WAIT_RE = re.compile(r"^等(?:待)?\s*(\d+(?:\.\d+)?)\s*(秒|s)?(?:钟)?$")
CLICK_RE = re.compile(rf"^(?:点击|单击|双击)\s*{_Q}(?P<t>[^“”\"'‘’「」]+){_Q}")


def _app_keywords(name: Optional[str], app_resolver=None) -> List[str]:
    if not name:
        return []
    for key, spec in APP_REGISTRY.items():
        n = name.strip().lower()
        if n == key or n in [a.lower() for a in spec.get("aliases", [])]:
            return list(spec["titles"])
    key = app_resolver(name) if app_resolver else None
    if key and key in APP_REGISTRY:
        return list(APP_REGISTRY[key]["titles"])
    base = re.split(r"[\\/]", name.strip())[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    # 文件名主干太短（a.txt -> "a"）会误匹配无关窗口，只在 4 个字符以上时作为关键字
    return [base] + ([stem] if stem != base and len(stem) >= 4 else [])


class Decomposer:
    MODES = ("auto", "rule", "llm")

    def __init__(self, rule: Optional[RulePlanner] = None, llm_plan: Optional[Callable[[str], List[str]]] = None,
                 mode: str = "auto", max_goals: int = MAX_GOALS):
        if mode not in self.MODES:
            raise ValueError(f"不支持的拆解模式 {mode!r}，可选 {self.MODES}")
        if mode == "llm" and llm_plan is None:
            raise ValueError("llm 拆解模式需要 llm_plan")
        self.rule = rule or RulePlanner()
        self.llm_plan = llm_plan
        self.mode = mode
        self.max_goals = max_goals

    @property
    def app_resolver(self):
        return self.rule.app_resolver

    # ---- 单段匹配（带上下文） ----
    def match_clause(self, clause: str, ctx: dict) -> Optional[Goal]:
        c = clause.strip().rstrip("。.!！")
        w = self.rule._w
        cur_app, cur_kw = ctx.get("app"), ctx.get("app_keywords") or []

        m = WAIT_RE.match(c)
        if m:
            return Goal("", c, [f"等待{float(m.group(1)):g}秒"], "template:wait", cur_app, cur_kw)
        m = URL_RE.match(c)
        if m:
            url = m.group("url")
            url = url if url.lower().startswith("http") else "https://" + url
            host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
            return Goal("", c, [f"打开网址“{url}”", w()], "template:open_url", "浏览器", BROWSER_KEYS,
                        [Postcondition("window_exists", {"keywords": BROWSER_KEYS})], True,
                        {"task_id": "open_url", "url": url, "host": host})
        if SAVE_RE.match(c):
            conds = []
            if ctx.get("file") and ctx.get("typed"):
                conds.append(Postcondition("file_contains", {"path": ctx["file"], "text": ctx["typed"][-1]}))
            return Goal("", c, ["按 ctrl+s 保存", w(1)], "template:save", cur_app, cur_kw, conds, True,
                        {"task_id": "save", "file": ctx.get("file")})
        m = SWITCH_RE.match(c)
        if m:
            app = _strip_quotes(m.group("app"))
            kws = _app_keywords(app, self.app_resolver)
            return Goal("", c, [f"切换到“{app}”"], "template:switch_app", app, kws,
                        [Postcondition("active_window", {"keywords": kws})], True, {"task_id": "switch_app"})
        if c in HOTKEY_WORDS:
            return Goal("", c, [f"按 {HOTKEY_WORDS[c]} {c}"], "template:hotkey", cur_app, cur_kw, [], True,
                        {"task_id": "hotkey"})
        m = HOTKEY_RE.match(c)
        if m:
            return Goal("", c, [f"按 {m.group(1).replace(' ', '').lower()}"], "template:hotkey", cur_app, cur_kw,
                        [], True, {"task_id": "hotkey"})
        m = KEY_RE.match(c)
        if m:
            return Goal("", c, [f"按{m.group(1)}"], "template:key", cur_app, cur_kw, [], False, {"task_id": "key"})
        m = CLICK_RE.match(c)
        if m:
            return Goal("", c, [c], "template:click", cur_app, cur_kw, [], False, {"task_id": "click"})
        for rx in TYPE_RES:
            m = rx.match(c)
            if not m:
                continue
            text = _strip_quotes(m.group("text"))
            app = re.sub(_Q, "", m.groupdict().get("app") or "").strip() or None
            if not text:
                break
            if app and app.endswith(("框", "栏")):   # “在“消息内容”框中输入X”：界面上的输入框，交给第4周的子任务写法
                return Goal("", c, [f"在“{app.rstrip('框栏')}”框中输入“{text}”"], "template:type_field", cur_app,
                            cur_kw, [], False, {"task_id": "type", "text": text})
            if app and app not in ("当前窗口", "当前程序"):
                kws = _app_keywords(app, self.app_resolver)
            else:
                app, kws = cur_app, cur_kw
            return Goal("", c, [f"直接输入“{text}”"], "template:type", app, kws, [], False,
                        {"task_id": "type", "text": text})

        p = self.rule.match_one(c)
        if p is None:
            return None
        tid = p.task_ids[0] if p.task_ids else "template"
        g = Goal("", c, list(p.subtasks), p.source, params={"task_id": tid, **p.params})
        if tid in ("open_browser",):
            g.app, g.app_keywords = "浏览器", BROWSER_KEYS
            g.postconditions = [Postcondition("window_exists", {"keywords": BROWSER_KEYS})]
        elif tid == "search":
            g.app, g.app_keywords = "浏览器", BROWSER_KEYS
            g.postconditions = [Postcondition("window_exists", {"keywords": [p.params["query"]]})]
        elif tid == "open_file":
            g.app = p.params["path"]
            g.app_keywords = _app_keywords(p.params["path"])
            g.postconditions = [Postcondition("window_exists", {"keywords": g.app_keywords})]
        elif tid == "open_app":
            g.app = p.params["app"]
            g.app_keywords = _app_keywords(p.params["app"], self.app_resolver)
            g.postconditions = [Postcondition("window_exists", {"keywords": g.app_keywords})]
        elif tid == "close_app":
            kws = _app_keywords(p.params["app"], self.app_resolver)
            g.app, g.app_keywords = None, []
            if p.params["app"] not in ("当前窗口", "窗口", "当前"):
                g.postconditions = [Postcondition("window_absent", {"keywords": kws})]
        elif tid == "send_message":
            app = p.params.get("app") or self.rule.chat_app
            g.app, g.app_keywords = app, _app_keywords(app, self.app_resolver)
            g.idempotent = False
            if app in ctx.get("opened", ()) and g.subtasks and g.subtasks[0] == f"打开{app}":
                # 前面的目标已经打开过聊天程序：切换过去，不再启动第二个实例
                g.subtasks = [f"切换到“{app}”"] + g.subtasks[2:]
            if "微信" not in app:
                g.postconditions = [Postcondition("text_present", {"text": f"{p.params['to']}：{p.params['message']}"})]
        return g

    # ---- 整体拆解 ----
    def decompose(self, instruction: str) -> TaskGraph:
        text = (instruction or "").strip()
        if not text:
            return TaskGraph(text, [], "empty")
        if self.mode == "llm":
            return self._llm_graph(text, [text])

        clauses = split_clauses(text)
        if not clauses:
            return TaskGraph(text, [], "empty")
        if len(clauses) > self.max_goals:
            raise ValueError(f"指令拆出 {len(clauses)} 个目标，超过上限 {self.max_goals}，请分几次执行")

        ctx = {"app": None, "app_keywords": [], "file": None, "typed": [], "opened": set()}
        goals, unmatched = [], []
        for c in clauses:
            g = self.match_clause(c, ctx)
            if g is None:
                unmatched.append(c)
                if self.mode == "rule" or self.llm_plan is None:
                    continue
                g = Goal("", c, list(self.llm_plan(c) or []), "llm", ctx.get("app"), ctx.get("app_keywords") or [],
                         [], False, {"task_id": "llm"})
            goals.append(g)
            self._advance(ctx, g)
        if unmatched and (self.mode == "rule" or self.llm_plan is None):
            raise ValueError("规则模板无法识别：" + "；".join(f"「{u}」" for u in unmatched) +
                             "（请启用 LLM 规划，或改写成“打开X / 给X发送消息：Y / 输入“Y” / 保存 / 关闭X”这类写法）")
        for i, g in enumerate(goals, 1):
            g.id = f"G{i}"
        srcs = {g.source.split(":")[0] for g in goals}
        source = "mixed" if len(srcs) > 1 else ("llm" if srcs == {"llm"} else
                                                 "template:" + "+".join(g.params.get("task_id", "?") for g in goals))
        return TaskGraph(text, goals, source)

    @staticmethod
    def _advance(ctx: dict, g: Goal):
        tid = g.params.get("task_id")
        if tid == "close_app":
            closed = g.params.get("app")
            ctx["opened"] = {a for a in ctx.get("opened", set()) if a != closed}
            ctx.update(app=None, app_keywords=[], file=None, typed=[])
            return
        if tid in ("open_app", "send_message", "open_file", "open_browser", "search") and g.app:
            ctx.setdefault("opened", set()).add(g.app)
        if tid == "open_file":
            ctx.update(file=g.params.get("path"), typed=[])
        elif tid in ("open_app", "open_browser", "search", "send_message", "open_url", "switch_app"):
            if g.app != ctx.get("app"):
                ctx.update(file=None, typed=[])
        if tid == "type":
            ctx.setdefault("typed", []).append(g.params.get("text"))
        if g.app:
            ctx.update(app=g.app, app_keywords=g.app_keywords)

    def _llm_graph(self, text: str, clauses: List[str]) -> TaskGraph:
        goals = [Goal(f"G{i}", c, list(self.llm_plan(c) or []), "llm", params={"task_id": "llm"}, idempotent=False)
                 for i, c in enumerate(clauses, 1)]
        return TaskGraph(text, goals, "llm")
