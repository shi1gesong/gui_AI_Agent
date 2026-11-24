"""
端到端 GUI 智能体 v1.0（第4周）

闭环：用户指令 → 任务规划 → [屏幕感知 → 动作决策 → 动作执行 → 执行结果反馈] × 每个子任务 → 结果反思

    指令 ──► HybridPlanner（规则模板 / LLM）──► 子任务列表
                                              │
          ┌───────────────────────────────────┘
          ▼
    子任务 i ──► 需要看屏幕吗？──否──► 规则动作（open_app/hotkey/wait…）
                    │是
                    ▼
            ScreenPerceiver：截图 → OCR → DPI 换算 → 元素列表
                    ▼
            动作决策：hybrid = 规则能推断就用规则，否则 LLM（BaseGUIAgent.act，带校验/重试/兜底）
                    ▼
            ControllerExecutor：grounding（文本→坐标）→ pyautogui / AppManager 执行
                    ▼
            执行失败？──是──► 等待后重新感知、重新决策（step_retries 次）
                    ▼否
    全部子任务结束 ──► Reflect（LLM + 规则把关，或纯规则）──► 任务检查器（可选）──► 轨迹 trace.json

决策模式 act_mode：
  hybrid（默认）：规则优先，规则推断不出的子任务才调用 LLM —— 速度快、5 个基础任务稳定
  llm           ：每个子任务都调用 LLM（第3周方式 + 第4周扩展动作空间），用于对比实验
  rule          ：完全不用 LLM（Ollama 未启动时也能跑模板任务）
  vlm（第5周）  ：不需要看屏幕的子任务仍用规则；需要看屏幕的交给多模态模型（Qwen2-VL + LoRA），
                  模型看整张截图直接输出坐标，不依赖 OCR 文字匹配
"""
import json
import re
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

try:
    from .action_validator import SubtaskIntent, fallback_action, needs_screen
    from .planner import HybridPlanner, Plan
except ImportError:
    from action_validator import SubtaskIntent, fallback_action, needs_screen
    from planner import HybridPlanner, Plan

ACT_MODES = ("hybrid", "llm", "rule", "vlm")


_PASSIVE_RE = re.compile(r"(查看|阅读|浏览|看到|看一下|获取)[^，。,;；]{0,12}(结果|内容|信息|页面)")


class GUIAgent:
    def __init__(self, perceiver, executor, planner: Optional[HybridPlanner] = None, brain=None,
                 act_mode: str = "hybrid", reflect_mode: str = "auto", step_retries: int = 1,
                 retry_wait: float = 1.0, max_steps: int = 20, stop_on_failure: bool = True,
                 trace_dir: Optional[str] = None, sleep: Callable = time.sleep, log: Callable = print,
                 vlm=None):
        """
        perceiver: perception.screen_parser.ScreenPerceiver / StaticPerceiver（有 perceive() -> Observation）
        executor : agent.executor.ControllerExecutor / DryRunExecutor（callable(action, elements) -> {"ok",...}）
        brain    : agent.base_agent.BaseGUIAgent（LLM 的 plan/act/reflect）；None 表示不用 LLM
        reflect_mode: auto（有 brain 用 LLM+规则把关，否则规则）/ llm / rule
        """
        if act_mode not in ACT_MODES:
            raise ValueError(f"不支持的决策模式 {act_mode!r}，可选 {ACT_MODES}")
        if act_mode == "llm" and brain is None:
            raise ValueError("llm 决策模式需要 brain（BaseGUIAgent）")
        if act_mode == "vlm" and vlm is None:
            raise ValueError("vlm 决策模式需要 vlm（agent.vlm_client.VLMClient）")
        if reflect_mode not in ("auto", "llm", "rule"):
            raise ValueError(f"不支持的反思模式 {reflect_mode!r}")
        self.perceiver = perceiver
        self.executor = executor
        self.brain = brain
        self.planner = planner or HybridPlanner(mode="auto" if brain else "rule",
                                                llm_plan=brain.plan if brain else None)
        self.act_mode = act_mode
        self.vlm = vlm
        self._instruction = ""
        self.reflect_mode = reflect_mode
        self.step_retries = max(0, int(step_retries))
        self.retry_wait = retry_wait
        self.max_steps = max_steps
        self.stop_on_failure = stop_on_failure
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.sleep = sleep
        self.log = log or (lambda *a, **k: None)

    # ------------------------------------------------------------------ #
    def _perceive(self, step: dict):
        obs = self.perceiver.perceive()
        step.setdefault("observations", []).append(obs.summary())
        return obs

    def decide(self, subtask: str, step: dict):
        """返回 (action, elements, source)"""
        if self.act_mode in ("hybrid", "rule", "vlm"):
            a = fallback_action(subtask, [])
            if a is not None and not needs_screen(a):
                return a, [], "rule"
        obs = self._perceive(step)
        if self.act_mode == "vlm":
            a = self.vlm.act(subtask, self._instruction, obs)
            if getattr(self.vlm, "last", None):
                step["vlm"] = {k: v for k, v in self.vlm.last.items() if k != "actions"}
            return a, obs.elements, "vlm"
        if self.act_mode == "rule" or self.brain is None:
            a = fallback_action(subtask, obs.elements)
            if a is None:
                targets = SubtaskIntent(subtask).click_targets
                if targets:  # 目标暂时不在屏幕上（窗口还没加载完等），可以等一会再试
                    a = {"action_type": None, "_invalid": True, "_retryable": True,
                         "_reasons": [f"屏幕上没有找到 {targets}"]}
                else:
                    a = {"action_type": None, "_invalid": True, "_reasons": ["规则无法推断动作，且未启用 LLM"]}
            return a, obs.elements, "rule"
        if self.act_mode == "hybrid":
            a = fallback_action(subtask, obs.elements)
            if a is not None:
                return a, obs.elements, "rule"
        a = self.brain.act(subtask, obs.to_context())
        return a, obs.elements, "llm"

    def run_subtask(self, index: int, subtask: str) -> dict:
        step = {"index": index, "subtask": subtask, "attempts": []}
        t0 = time.time()
        for attempt in range(self.step_retries + 1):
            rec = {"attempt": attempt + 1}
            try:
                action, elements, source = self.decide(subtask, rec)
            except Exception as e:  # 感知 / LLM 调用失败
                rec.update(ok=False, error=f"决策失败: {type(e).__name__}: {e}")
                step["attempts"].append(rec)
                self.log(f"  ✗ 第{attempt + 1}次: {rec['error']}")
                if attempt < self.step_retries:
                    self.sleep(self.retry_wait)
                continue
            clean = {k: v for k, v in action.items() if k not in ("_rejected", "vlm_raw")}
            rec.update(action=clean, source=source)
            if action.get("_rejected"):
                rec["llm_rejected"] = len(action["_rejected"])
            result = self.executor(action, elements) or {}
            rec.update(ok=bool(result.get("ok")), error=result.get("error"), detail=result.get("detail"))
            step["attempts"].append(rec)
            desc = _describe_action(clean)
            if rec["ok"]:
                self.log(f"  ✓ [{source}] {desc}")
                break
            self.log(f"  ✗ [{source}] {desc} —— {rec['error']}")
            if action.get("_invalid") and not action.get("_retryable") and source == "rule":
                break  # 规则推断不出动作，重试也没有用
            if attempt < self.step_retries:
                self.sleep(self.retry_wait)
        last = step["attempts"][-1] if step["attempts"] else {}
        step.update(ok=bool(last.get("ok")), action=last.get("action"), source=last.get("source"),
                    error=None if last.get("ok") else last.get("error"), sec=round(time.time() - t0, 2))
        return step

    # ------------------------------------------------------------------ #
    def reflect(self, instruction: str, steps: List[dict]) -> dict:
        problems = [f"子任务「{s['subtask']}」失败: {s.get('error')}" for s in steps if not s["ok"]]
        if not steps:
            problems.append("没有执行任何子任务")
        use_llm = self.reflect_mode == "llm" or (self.reflect_mode == "auto" and self.brain is not None
                                                  and self.act_mode != "rule")
        if use_llm and self.brain is not None:
            self.brain.action_history = [
                {"subtask": s["subtask"], "action": s.get("action") or {"_invalid": True},
                 "executed": s["ok"], **({"error": s["error"]} if s.get("error") else {})} for s in steps]
            try:
                r = self.brain.reflect(instruction)
                r["mode"] = "llm+guard"
                return self._guard_over_strict(instruction, steps, r)
            except Exception as e:
                problems.append(f"LLM 反思调用失败: {type(e).__name__}: {e}")
        return {"is_complete": not problems, "mode": "rule",
                "reason": "所有子任务均已成功执行" if not problems else "；".join(problems)}

    @staticmethod
    def _guard_over_strict(instruction: str, steps: List[dict], r: dict) -> dict:
        """反思过严的规则把关（2026-11-24 真机 --act llm：T2 搜索 0/3，检查器都通过，
        qwen3:8b 却以“用户尚未查看天气预报结果”判未完成）。
        只在全部子任务真实执行成功、且 LLM 的理由只是要求“查看/阅读/浏览结果”这类指令没要求的被动操作时改判完成，
        原判断保留在 llm_is_complete / override_reason 里。"""
        if r.get("is_complete") or r.get("guard_reasons") or not steps or not all(s["ok"] for s in steps):
            return r
        text = f"{r.get('reason', '')} {r.get('next_action_suggestion', '')}"
        if not _PASSIVE_RE.search(text) or _PASSIVE_RE.search(instruction):
            return r
        return {**r, "llm_is_complete": False, "is_complete": True,
                "override_reason": "所有子任务均已成功执行；LLM 要求的“查看结果”超出指令范围"}

    def run(self, instruction: str, checker: Optional[Callable[[], tuple]] = None,
            plan: Optional[Plan] = None) -> dict:
        """
        checker: 可选，无参函数，返回 (是否成功, 说明)；用于基础任务测试时客观判定结果（看窗口标题、聊天记录等）
        plan: 可选，跳过规划直接使用给定子任务
        """
        t0 = time.time()
        self._instruction = instruction
        self.log(f"[指令] {instruction}")
        result = {"instruction": instruction, "act_mode": self.act_mode, "steps": []}
        try:
            plan = plan or self.planner.plan(instruction)
        except Exception as e:
            result.update(plan=None, success=False, error=f"规划失败: {type(e).__name__}: {e}",
                          reflection={"is_complete": False, "reason": "规划失败"}, sec=round(time.time() - t0, 2))
            self.log(f"[规划] 失败: {result['error']}")
            self._save(result)
            return result
        result["plan"] = plan.to_dict()
        self.log(f"[规划] 来源={plan.source}，{len(plan.subtasks)} 个子任务")
        for i, st in enumerate(plan.subtasks, 1):
            self.log(f"   {i}. {st}")
        if len(plan.subtasks) > self.max_steps:
            self.log(f"[规划] 子任务超过 max_steps={self.max_steps}，只执行前 {self.max_steps} 个")

        for i, st in enumerate(plan.subtasks[: self.max_steps], 1):
            self.log(f"[执行 {i}/{min(len(plan.subtasks), self.max_steps)}] {st}")
            step = self.run_subtask(i, st)
            result["steps"].append(step)
            if not step["ok"] and self.stop_on_failure:
                self.log("[执行] 子任务失败，停止后续子任务")
                break

        result["reflection"] = self.reflect(instruction, result["steps"])
        executed_all = (len(result["steps"]) == len(plan.subtasks) and bool(plan.subtasks)
                        and all(s["ok"] for s in result["steps"]))
        result["executed_all"] = executed_all
        success = executed_all and bool(result["reflection"].get("is_complete"))
        if checker is not None:
            try:
                ok, msg = checker()
            except Exception as e:
                ok, msg = False, f"检查器异常: {type(e).__name__}: {e}"
            result["check"] = {"ok": bool(ok), "message": msg}
            success = success and bool(ok)
        result["success"] = success
        result["sec"] = round(time.time() - t0, 2)
        self.log(f"[反思] {'完成' if result['reflection'].get('is_complete') else '未完成'}："
                 f"{result['reflection'].get('reason', '')}")
        if result["reflection"].get("override_reason"):
            self.log(f"[反思] 规则改判完成：{result['reflection']['override_reason']}")
        if "check" in result:
            self.log(f"[检查] {'通过' if result['check']['ok'] else '不通过'}：{result['check']['message']}")
        self.log(f"[结果] {'成功' if success else '失败'}，用时 {result['sec']} 秒")
        self._save(result)
        return result

    def _save(self, result: dict):
        if self.trace_dir is None:
            return
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / "trace.json"
            result["trace_path"] = str(path)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            result["trace_error"] = traceback.format_exc(limit=1)


def _describe_action(a: dict) -> str:
    t = a.get("action_type")
    tgt, txt = a.get("target_text") or "", a.get("input_text") or ""
    pt = a.get("point")
    if pt is not None and not tgt:
        tgt = f"({pt[0]}, {pt[1]})"
    s = {"click": f"{'右键' if a.get('button') == 'right' else ''}点击「{tgt}」", "double_click": f"双击「{tgt}」",
         "move": f"移动到「{tgt}」", "drag": f"拖拽「{tgt}」→ {a.get('end_point')}", "type": f"输入「{txt}」" + (f"（到「{tgt}」）" if tgt else ""),
         "key": f"按键 {txt or tgt}", "hotkey": f"组合键 {txt or tgt}", "scroll": f"滚动 {a.get('amount', '')}",
         "open_app": f"打开「{tgt}」", "close_app": f"关闭「{tgt}」", "wait": f"等待 {txt} 秒", "done": "完成"}.get(t, str(t))
    if a.get("follow_up"):
        s += " → " + " → ".join(_describe_action(f) for f in a["follow_up"])
    return s
