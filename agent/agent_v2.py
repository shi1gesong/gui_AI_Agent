"""
GUI 智能体 v2.0（第6周）：复杂任务分步执行 + 错误检测与自动重试 + 实时监控

    指令 ──► Decomposer：切分成目标（Goal），每个目标 = 子任务列表 + 目标程序 + 后置条件 + 是否可重放
              │
              ▼
    目标 i ──► 子任务 j ──► 执行前检查：关闭本次运行新冒出的弹窗、目标程序不在前台就重新激活（WindowGuard）
                  │
                  ▼
              决策（规则 / LLM / VLM；服务连续失败 → 熔断降级为规则）
                  ▼
              截图 ─► 执行 ─► 截图：前后对比，应当变化却没变化 = 动作没生效（EffectVerifier）
                  ▼
              失败？──► 错误分类（recovery.classify_error）──► 该类错误还能重试吗？
                  │            是：恢复动作（关弹窗 / 重新激活窗口 / 指数退避 / 换预处理重新识别）→ 再试
                  │            否：子任务失败
                  ▼
    目标的子任务全部成功 ──► 检查后置条件（窗口出现/消失、屏幕出现某文字、文件内容）
                  │            不通过且目标可重放：恢复后整个目标重做 1 次；不可重放（发消息）：如实判失败
                  ▼
    全部目标结束 ──► Reflect（LLM 不可用时自动改用规则）──► 任务检查器（可选）
    全程每一步写事件到 TaskMonitor：events.jsonl / agent.log / status.json（python -m agent.monitor 实时查看）

与 v1.0（agent/gui_agent.py）的关系：继承 GUIAgent，复用其决策（decide）、反思（reflect）与动作描述；
v1.0 保持不变，CLI 用 --engine v1 仍可运行，便于做对比实验。
"""
import json
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

try:
    from .action_validator import fallback_action
    from .decomposer import Decomposer, Goal, TaskGraph
    from .gui_agent import GUIAgent, _describe_action
    from .monitor import TaskMonitor
    from .planner import HybridPlanner, RulePlanner
    from .recovery import CircuitBreaker, RetryPolicy, WindowGuard, classify_error, recovery_actions
    from .verifier import EFFECT_REQUIRED, EffectVerifier, check_postconditions
except ImportError:  # agent/ 在 sys.path 上时
    from action_validator import fallback_action
    from decomposer import Decomposer, Goal, TaskGraph
    from gui_agent import GUIAgent, _describe_action
    from monitor import TaskMonitor
    from planner import HybridPlanner, RulePlanner
    from recovery import CircuitBreaker, RetryPolicy, WindowGuard, classify_error, recovery_actions
    from verifier import EFFECT_REQUIRED, EffectVerifier, check_postconditions

# 这些动作本身会改变窗口（打开/关闭/切换），执行前不要求目标程序在前台
_WINDOW_CHANGING = ("open_app", "close_app", "switch_app", "wait", "done")


class GUIAgentV2(GUIAgent):
    ENGINE = "v2"

    def __init__(self, perceiver, executor, decomposer: Optional[Decomposer] = None, brain=None,
                 act_mode: str = "hybrid", reflect_mode: str = "auto", app_manager=None,
                 policy: Optional[RetryPolicy] = None, verify_effect: bool = True,
                 verifier: Optional[EffectVerifier] = None, goal_retries: int = 1,
                 breaker: Optional[CircuitBreaker] = None, max_steps: int = 40, stop_on_failure: bool = True,
                 trace_dir: Optional[str] = None, sleep: Callable = time.sleep, log: Callable = print, vlm=None,
                 frame_fn: Optional[Callable] = None, settle_sec: float = 0.3, post_wait: float = 0.5,
                 post_checks: int = 4, budget_sec: Optional[float] = None, monitor: Optional[TaskMonitor] = None,
                 watchdog_sec: Optional[float] = None, stuck_after: float = 60.0, retry_preprocess: str = "upscale",
                 clock: Callable[[], float] = time.time, planner: Optional[HybridPlanner] = None,
                 late_window_sec: float = 8.0):
        rule = decomposer.rule if decomposer is not None else (planner.rule if planner is not None else RulePlanner())
        planner = planner or HybridPlanner(mode="auto" if brain else "rule", rule=rule,
                                           llm_plan=brain.plan if brain else None)
        super().__init__(perceiver, executor, planner=planner, brain=brain, act_mode=act_mode,
                         reflect_mode=reflect_mode, step_retries=0, max_steps=max_steps,
                         stop_on_failure=stop_on_failure, trace_dir=trace_dir, sleep=sleep, log=log, vlm=vlm)
        self.decomposer = decomposer or Decomposer(rule, llm_plan=brain.plan if brain else None,
                                                   mode="auto" if brain else "rule")
        self.apps = app_manager
        self.guard = WindowGuard(app_manager)
        self.policy = policy or RetryPolicy()
        self.verifier = (verifier or EffectVerifier()) if verify_effect else None
        self.breaker = breaker or CircuitBreaker(threshold=2, cooldown=120.0, clock=clock)
        self.goal_retries = max(0, int(goal_retries))
        self.frame_fn = frame_fn or getattr(perceiver, "grab", None)
        self.settle_sec, self.post_wait, self.post_checks = settle_sec, post_wait, max(1, post_checks)
        self.budget_sec = budget_sec
        self._monitor_given = monitor
        self.monitor: Optional[TaskMonitor] = monitor
        self.watchdog_sec, self.stuck_after = watchdog_sec, stuck_after
        self.retry_preprocess = retry_preprocess
        self.late_window_sec = late_window_sec
        self.clock = clock
        self.degraded: Optional[str] = None
        self.aborted = False

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _ev(self, kind: str, **data):
        if self.monitor is not None:
            return self.monitor.event(kind, **data)
        return None

    def _frame(self):
        if self.frame_fn is None:
            return None
        try:
            return self.frame_fn()
        except Exception:
            return None

    def _texts_fn(self):
        """后置条件里的 text_* 需要 OCR；vlm 模式不跑 OCR 时返回 None（跳过文字类检查）"""
        ocr = getattr(self.perceiver, "ocr", None)
        if ocr is None or type(ocr).__name__ == "NullOCR":
            return None
        return lambda: self.perceiver.perceive().texts()

    def _degrade(self, component: str, reason: str):
        if self.degraded:
            return
        self.degraded = "rule"
        self._ev("degrade", component=component, to="rule", reason=reason)

    def _over_budget(self, t0: float) -> bool:
        return self.budget_sec is not None and self.clock() - t0 > self.budget_sec

    # ------------------------------------------------------------------ #
    # 决策（熔断后降级为规则）
    # ------------------------------------------------------------------ #
    def decide(self, subtask: str, step: dict):
        if self.degraded and self.act_mode != "rule":
            saved = self.act_mode
            self.act_mode = "rule"
            try:
                return super().decide(subtask, step)
            finally:
                self.act_mode = saved
        return super().decide(subtask, step)

    def reflect(self, instruction: str, steps: List[dict]) -> dict:
        if self.degraded or (self.brain is not None and not self.breaker.allow()):
            saved = self.reflect_mode
            self.reflect_mode = "rule"
            try:
                r = super().reflect(instruction, steps)
            finally:
                self.reflect_mode = saved
            r["degraded"] = True
            return r
        r = super().reflect(instruction, steps)
        if "LLM 反思调用失败" in str(r.get("reason", "")):
            self.breaker.record_failure()
            self._degrade("reflect", "LLM 反思调用失败，改用规则反思")
            return self.reflect(instruction, steps)
        return r

    # ------------------------------------------------------------------ #
    # 子任务：带错误分类的重试
    # ------------------------------------------------------------------ #
    def _recover(self, cls: str, goal: Optional[Goal], delay: float, rec: dict):
        for act in recovery_actions(cls):
            result = None
            if act == "dismiss_popup":
                closed = self.guard.dismiss_popups()
                if closed:
                    result = closed
            elif act == "refocus":
                kws = goal.app_keywords if goal else []
                if kws and self.apps is not None and self.apps.find_windows(kws) and not self.guard.is_foreground(kws):
                    result = self.guard.refocus(kws)
            elif act == "backoff":
                if delay > 0:
                    self.sleep(delay)
                    result = f"等待 {delay:g} 秒"
            elif act == "degrade":
                if not self.breaker.allow():
                    self._degrade("act", "决策服务连续失败，熔断")
                    result = "已降级为规则决策"
            if result:
                rec.setdefault("recover", []).append({"action": act, "result": result})
                self._ev("recover", action=act, result=result, error_class=cls)
        if cls == "target_not_found" and hasattr(self.perceiver, "reset_cache"):
            self.perceiver.reset_cache()

    def _wait_late_window(self, goal: Optional[Goal], rec: dict) -> bool:
        """open_app 超时后再等 late_window_sec 秒，看目标窗口是不是只是来得晚"""
        if goal is None or not goal.app_keywords or self.apps is None or self.apps.windows is None:
            return False
        waited, step = 0.0, 0.5
        while waited < self.late_window_sec:
            self.sleep(step)
            waited += step
            wins = self.apps.find_windows(goal.app_keywords)
            if wins:
                self.apps._activate(wins[0])
                try:
                    wins[0].activate()
                except Exception:
                    pass
                title = getattr(wins[0], "title", "")
                rec.setdefault("recover", []).append({"action": "wait_window", "result": f"{waited:g} 秒后出现「{title}」"})
                self._ev("recover", action="wait_window", result=f"{waited:g} 秒后出现「{title}」", error_class="app_launch")
                return True
        self._ev("recover", action="wait_window", result=f"再等 {waited:g} 秒仍未出现", error_class="app_launch")
        return False

    def run_subtask_v2(self, index: int, subtask: str, goal: Optional[Goal] = None) -> dict:
        step = {"index": index, "subtask": subtask, "goal": goal.id if goal else None, "attempts": []}
        t0 = self.clock()
        self._ev("step_start", index=index, subtask=subtask, goal=goal.id if goal else None)
        counts, total = Counter(), 0
        planned = fallback_action(subtask, [])
        planned_type = (planned or {}).get("action_type")
        saved_pre = getattr(self.perceiver, "preprocess", None)
        last_cls = None
        while True:
            total += 1
            rec = {"attempt": total}
            if total > 1:
                self._ev("attempt", attempt=total)
            # 第3次及以后仍找不到目标：换一种预处理再识别（小字放大 / 提高对比度）
            if (last_cls == "target_not_found" and counts["target_not_found"] >= 2 and self.retry_preprocess
                    and hasattr(self.perceiver, "preprocess")):
                self.perceiver.preprocess = self.retry_preprocess
                rec["preprocess"] = self.retry_preprocess
            # 执行前检查：弹窗 / 前台窗口
            kws = goal.app_keywords if (goal and planned_type not in _WINDOW_CHANGING) else None
            if kws and self.apps is not None and not self.apps.find_windows(kws):
                kws = None
            for fix in self.guard.ensure(kws):
                rec.setdefault("recover", []).append(fix)
                self._ev("recover", action=fix["action"], result=fix.get("result"), error_class="precheck")
            # 决策
            td = self.clock()
            try:
                action, elements, source = self.decide(subtask, rec)
                err = None
            except Exception as e:
                action, elements, source, err = None, [], None, e
            self._ev("decide", sec=round(self.clock() - td, 4), source=source, ok=err is None,
                     action=_describe_action(action) if isinstance(action, dict) else None)
            obs = rec.get("observations") or []
            if obs and isinstance(obs[-1], dict) and isinstance(obs[-1].get("sec"), (int, float)):
                self._ev("perceive", sec=obs[-1]["sec"], elements=obs[-1].get("elements"), cache=obs[-1].get("cache"))
            if err is not None:
                rec.update(ok=False, error=f"决策失败: {type(err).__name__}: {err}")
                cls = classify_error(err)
                if cls == "service":
                    if self.breaker.record_failure() or not self.breaker.allow():
                        self._degrade("act", f"决策服务连续失败 {self.breaker.failures} 次，熔断")
            else:
                clean = {k: v for k, v in action.items() if k not in ("_rejected", "vlm_raw")}
                rec.update(action=clean, source=source)
                if action.get("_rejected"):
                    rec["llm_rejected"] = len(action["_rejected"])
                verify = self.verifier is not None and action.get("action_type") in EFFECT_REQUIRED \
                    and not action.get("_invalid")
                before = self._frame() if verify else None
                te = self.clock()
                result = self.executor(action, elements) or {}
                ok = bool(result.get("ok"))
                err_msg = result.get("error")
                if action.get("_invalid") and action.get("_reasons"):
                    err_msg = "动作未通过校验：" + "；".join(map(str, action["_reasons"]))[:200]
                rec.update(ok=ok, error=err_msg, detail=result.get("detail"))
                self._ev("execute", sec=round(self.clock() - te, 4), ok=ok, action=_describe_action(clean),
                         error=result.get("error"))
                if ok and source in ("llm", "vlm"):
                    self.breaker.record_success()
                if ok and before is not None:
                    if self.settle_sec:
                        self.sleep(self.settle_sec)
                    tv = self.clock()
                    v = self.verifier.verify(action, before, self._frame())
                    rec["verify"] = v
                    self._ev("verify", sec=round(self.clock() - tv, 4), ok=v["ok"], diff=v.get("diff"))
                    if not v["ok"]:
                        if counts["no_effect"] + 1 >= self.policy.max_attempts.get("no_effect", 2):
                            # 重试后界面仍无变化：认为该动作本身没有可见效果，记警告继续（结果由后置条件把关）
                            rec["verify_warning"] = v["reason"]
                        else:
                            rec.update(ok=False, error=v["reason"])
                cls = None if rec["ok"] else classify_error(rec.get("error"), action)
                if not rec["ok"] and action.get("_invalid") and not action.get("_retryable") and source == "rule":
                    cls = "invalid_action_final"
            step["attempts"].append(rec)
            desc = _describe_action(rec["action"]) if rec.get("action") else "（无动作）"
            if rec["ok"]:
                self.log(f"    ✓ [{source}] {desc}")
                break
            self.log(f"    ✗ [{source}] {desc} —— {rec.get('error')}")
            if cls == "abort":
                self.aborted = True
                step["error_class"] = cls
                break
            if cls == "invalid_action_final":  # 规则推断不出动作，重试也没用
                step["error_class"] = "invalid_action"
                break
            counts[cls] += 1
            last_cls = cls
            step["error_class"] = cls
            if cls == "app_launch" and self._wait_late_window(goal, rec):
                # 程序启动慢：窗口在超时之后才出现。不重新启动（否则会开出第二个实例），直接算成功
                rec.update(ok=True, error=None, late_window=True)
                break
            if not self.policy.allowed(cls, counts[cls], total) or self.aborted:
                break
            delay = self.policy.delay(counts[cls], cls)
            self._ev("retry", next_attempt=total + 1, error_class=cls, error=rec.get("error"), delay=delay)
            self._recover(cls, goal, delay, rec)
        if hasattr(self.perceiver, "preprocess"):
            self.perceiver.preprocess = saved_pre
        last = step["attempts"][-1] if step["attempts"] else {}
        step.update(ok=bool(last.get("ok")), action=last.get("action"), source=last.get("source"),
                    error=None if last.get("ok") else last.get("error"), sec=round(self.clock() - t0, 2),
                    attempts_n=total, retries=total - 1)
        if step["ok"]:
            step.pop("error_class", None)
        self._ev("step_end", index=index, ok=step["ok"], sec=step["sec"], error=step.get("error"),
                 error_class=step.get("error_class"), attempts=total)
        return step

    # ------------------------------------------------------------------ #
    # 目标：子任务 + 后置条件 + 可重放目标的整体重做
    # ------------------------------------------------------------------ #
    def _check_post(self, goal: Goal) -> dict:
        if not goal.postconditions:
            return {"ok": True, "results": []}
        texts_fn = self._texts_fn()
        conds = [c for c in goal.postconditions if not (c.needs_screen and texts_fn is None)]
        n = self.post_checks if not any(c.needs_screen for c in conds) else min(2, self.post_checks)
        res = {"ok": True, "results": []}
        for i in range(n):
            if i and hasattr(self.perceiver, "reset_cache"):
                self.perceiver.reset_cache()   # 重新检查时重新识别，不用缓存
            res = check_postconditions(conds, self.apps, texts_fn)
            if res["ok"]:
                break
            if i < n - 1:
                self.sleep(self.post_wait)
        res["checks"] = i + 1
        return res

    def run_goal(self, goal: Goal, start_index: int, t0: float) -> dict:
        g = {"id": goal.id, "description": goal.description, "source": goal.source, "replays": 0, "steps": []}
        self._ev("goal_start", goal=goal.id, description=goal.description, subtasks=len(goal.subtasks))
        for replay in range(self.goal_retries + 1):
            steps, ok = [], True
            for j, st in enumerate(goal.subtasks):
                if self._over_budget(t0):
                    steps.append({"index": start_index + j, "subtask": st, "ok": False, "attempts": [],
                                  "error": f"超出时间预算 {self.budget_sec:g} 秒", "error_class": "budget"})
                    ok = False
                    break
                s = self.run_subtask_v2(start_index + j, st, goal)
                steps.append(s)
                if not s["ok"]:
                    ok = False
                    break
            g["steps"].extend(steps)
            post = self._check_post(goal) if ok else None
            if post is not None:
                g["post"] = post
            reason = None
            if ok and post["ok"]:
                g.update(ok=True)
                break
            if ok:  # 子任务都成功，但结果不对
                failed = [r for r in post["results"] if not r["ok"] and r["required"]]
                reason = "结果检查不通过：" + "；".join(f"{r['desc']}（{r['message']}）" for r in failed)
                cls = "postcondition"
            else:
                reason = steps[-1].get("error") if steps else "无子任务"
                cls = steps[-1].get("error_class") or classify_error(reason) if steps else "unknown"
            g.update(ok=False, reason=reason, error_class=cls)
            can_replay = (goal.idempotent and replay < self.goal_retries and not self.aborted
                          and cls not in ("abort", "invalid_action", "budget", "service") and not self._over_budget(t0))
            if not can_replay:
                break
            g["replays"] += 1
            delay = self.policy.delay(replay + 1, cls)
            self._ev("retry", next_attempt=replay + 2, error_class=cls, error=reason, delay=delay, scope="goal",
                     goal=goal.id)
            self._recover(cls, goal, delay, g)
        self._ev("goal_end", goal=goal.id, ok=g.get("ok", False), reason=g.get("reason"), replays=g["replays"])
        return g

    # ------------------------------------------------------------------ #
    def run(self, instruction: str, checker: Optional[Callable[[], tuple]] = None,
            plan=None, graph: Optional[TaskGraph] = None) -> dict:
        t0 = self.clock()
        self._instruction = instruction
        self.aborted = False
        self.degraded = None
        if self._monitor_given is None:
            self.monitor = TaskMonitor(str(self.trace_dir) if self.trace_dir else None, stuck_after=self.stuck_after,
                                       listeners=[self._log_event])
        stop_watch = self._start_watchdog()
        self.guard.snapshot()
        if hasattr(self.perceiver, "reset_cache"):
            self.perceiver.reset_cache()
        result = {"instruction": instruction, "engine": self.ENGINE, "act_mode": self.act_mode, "steps": [],
                  "goals": []}
        self._ev("run_start", instruction=instruction, engine=self.ENGINE, act_mode=self.act_mode)
        try:
            try:
                tp = self.clock()
                if graph is None:
                    if plan is not None:  # 兼容 v1：直接给子任务列表
                        graph = TaskGraph(instruction, [Goal("G1", instruction, list(plan.subtasks), plan.source)],
                                          plan.source)
                    else:
                        graph = self.decomposer.decompose(instruction)
                self._ev("plan", sec=round(self.clock() - tp, 4), stage="plan", source=graph.source,
                         goals=[{"id": g.id, "description": g.description, "subtasks": g.subtasks} for g in graph.goals])
            except Exception as e:
                cls = classify_error(e)
                result.update(plan=None, success=False, error=f"规划失败: {type(e).__name__}: {e}",
                              reflection={"is_complete": False, "reason": "规划失败"}, error_class=cls)
                self._ev("error", error=result["error"], error_class=cls)
                return self._finish(result, t0)
            result["plan"] = graph.to_plan().to_dict()
            result["graph"] = graph.to_dict()
            if hasattr(self.perceiver, "add_words"):  # 任务里的目标词加入 OCR 纠错词表
                words = [w for g in graph.goals for p in g.params.values() if isinstance(p, str) for w in [p]
                         if 2 <= len(w) <= 12]
                self.perceiver.add_words(words)
            for i, g in enumerate(graph.goals, 1):
                self.log(f"   {g.id}. {g.description}：" + " → ".join(g.subtasks))
            n_sub = len(graph.subtasks)
            if n_sub > self.max_steps:
                result["truncated"] = True
            idx = 1
            for goal in graph.goals:
                if idx > self.max_steps:
                    break
                gr = self.run_goal(goal, idx, t0)
                result["goals"].append({k: v for k, v in gr.items() if k != "steps"})
                result["steps"].extend(gr["steps"])
                idx += len(goal.subtasks)
                if self.aborted or (not gr.get("ok") and self.stop_on_failure):
                    break
            tr = self.clock()
            result["reflection"] = self.reflect(instruction, result["steps"])
            self._ev("reflect", sec=round(self.clock() - tr, 4), ok=bool(result["reflection"].get("is_complete")))
            executed_all = (bool(graph.goals) and len(result["goals"]) == len(graph.goals)
                            and all(gr.get("ok") for gr in result["goals"]) and not result.get("truncated"))
            result["executed_all"] = executed_all
            success = executed_all and bool(result["reflection"].get("is_complete")) and not self.aborted
            if checker is not None:
                try:
                    ok, msg = checker()
                except Exception as e:
                    ok, msg = False, f"检查器异常: {type(e).__name__}: {e}"
                result["check"] = {"ok": bool(ok), "message": msg}
                success = success and bool(ok)
                self.log(f"[检查] {'通过' if ok else '不通过'}：{msg}")
            result["success"] = success
        except KeyboardInterrupt:
            self.aborted = True
            result.update(success=False, error="用户中止（Ctrl+C）")
        finally:
            stop_watch()
        return self._finish(result, t0)

    def _finish(self, result: dict, t0: float) -> dict:
        result.setdefault("success", False)
        result["aborted"] = self.aborted
        result["degraded"] = self.degraded
        result["sec"] = round(self.clock() - t0, 2)
        steps = result.get("steps", [])
        result["robustness"] = {
            "retries": sum(s.get("retries", 0) for s in steps),
            "recoveries": sum(len(a.get("recover", [])) for s in steps for a in s.get("attempts", [])),
            "goal_replays": sum(g.get("replays", 0) for g in result.get("goals", [])),
            "error_classes": dict(Counter(a_cls for s in steps for a_cls in
                                          [classify_error(a.get("error"), a.get("action")) for a in s.get("attempts", [])
                                           if not a.get("ok")])),
        }
        self._ev("run_end", success=result["success"], sec=result["sec"], aborted=self.aborted,
                 degraded=self.degraded)
        if self.monitor is not None:
            result["metrics"] = self.monitor.metrics()
            if self._monitor_given is None:
                self.monitor.close()
        if hasattr(self.perceiver, "stats"):
            result["perception_stats"] = dict(self.perceiver.stats)
        self._save(result)
        return result

    def _log_event(self, ev: dict):
        msg = TaskMonitor.format_event(ev)
        if msg and ev.get("event") not in ("step_end",):
            self.log(msg)

    def _start_watchdog(self):
        if not self.watchdog_sec or self.monitor is None:
            return lambda: None
        stop = threading.Event()
        mon = self.monitor

        def loop():
            while not stop.wait(self.watchdog_sec):
                try:
                    if mon.check_stuck():
                        self.log(f"[监控] 已超过 {mon.stuck_after:g} 秒没有进展，可能卡住")
                except Exception:
                    pass

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return stop.set
