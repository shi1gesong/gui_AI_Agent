"""
任务执行状态的实时监控与日志记录（第6周新增）

第4~5周只在任务结束后写一份 trace.json：执行过程中卡在哪一步、已经重试了几次、每个阶段花了多少时间，
只能盯着命令行滚动的输出猜。第6周把执行过程变成“事件流”，同时落三份文件（都在本次运行目录 runs/<时间>/ 下）：

  events.jsonl   每个事件一行 JSON（run_start / plan / goal_start / step_start / perceive / decide / execute /
                 verify / retry / recover / degrade / goal_end / step_end / run_end ...），带时间戳和耗时，便于事后分析
  agent.log      Python logging 标准日志（INFO 及以上；滚动文件，单个 2MB、保留 3 份），人可直接阅读
  status.json    “当前状态快照”，每个事件后原子替换：状态机、当前目标/子任务/第几次尝试、进度、已用时间、
                 预计剩余时间、重试/恢复/错误计数、各阶段耗时统计、最后心跳时间

另开一个窗口即可实时查看（不影响正在执行的 Agent）：
    python -m agent.monitor                 # 自动找 runs/ 下最新的一次运行
    python -m agent.monitor runs\\20261124_101500 --interval 0.5

状态机：pending → planning → running ⇄ recovering → succeeded / failed / aborted
卡住检测：超过 stuck_after 秒没有任何事件，状态快照里 stuck=true，watch 界面标红提示。
"""
import json
import logging
import logging.handlers
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

STATES = ("pending", "planning", "running", "recovering", "succeeded", "failed", "aborted")
TERMINAL_STATES = ("succeeded", "failed", "aborted")
STAGES = ("plan", "perceive", "decide", "execute", "verify", "recover", "reflect")

# 事件 -> 日志级别
_LEVELS = {"retry": logging.WARNING, "recover": logging.WARNING, "degrade": logging.WARNING,
           "error": logging.ERROR, "stuck": logging.WARNING}

LOGGER_NAME = "gui_agent"


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"


def atomic_write_json(path: Path, data: dict):
    """先写临时文件再 os.replace：watch 端永远读不到写了一半的 JSON（Windows 上 replace 也是原子的）"""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for _ in range(5):  # Windows 上若 watch 端恰好在读，replace 可能短暂 PermissionError
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    try:
        tmp.unlink()
    except OSError:
        pass


def summarize_durations(values: Iterable[float]) -> dict:
    v = sorted(float(x) for x in values)
    if not v:
        return {"count": 0, "total": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    return {"count": len(v), "total": round(sum(v), 3), "mean": round(statistics.fmean(v), 3),
            "p50": round(statistics.median(v), 3), "p95": round(p95, 3), "max": round(v[-1], 3)}


def build_logger(run_dir: Optional[Path], name: str = LOGGER_NAME, console: bool = False,
                 level: int = logging.INFO) -> logging.Logger:
    """每次运行一个独立 logger（名字带运行目录），避免多次运行的 handler 叠加、重复打印"""
    logger = logging.getLogger(f"{name}.{abs(hash(str(run_dir))) % 10 ** 8}" if run_dir else name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    if run_dir is not None:
        fh = logging.handlers.RotatingFileHandler(Path(run_dir) / "agent.log", maxBytes=2 * 1024 * 1024,
                                                  backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TaskMonitor:
    """事件记录 + 状态快照 + 阶段耗时统计。所有方法都不抛异常（监控出错不能影响任务执行）"""

    def __init__(self, run_dir: Optional[str] = None, clock: Callable[[], float] = _now,
                 stuck_after: float = 60.0, listeners: Iterable[Callable[[dict], None]] = (),
                 console: bool = False, write_files: bool = True):
        self.run_dir = Path(run_dir) if run_dir else None
        self.clock = clock
        self.stuck_after = stuck_after
        self.listeners: List[Callable[[dict], None]] = list(listeners)
        self.write_files = write_files and self.run_dir is not None
        self.events: List[dict] = []
        self.timings: Dict[str, List[float]] = {s: [] for s in STAGES}
        self.counters = {"retries": 0, "recoveries": 0, "errors": 0, "degrades": 0, "steps_ok": 0,
                         "steps_failed": 0, "goals_ok": 0, "goals_failed": 0}
        self.listener_errors = 0
        self._lock = threading.RLock()
        self._t0 = None
        self._seq = 0
        self.status = {"state": "pending", "instruction": None, "goals": [], "current": {}, "progress":
                       {"done": 0, "total": 0, "pct": 0.0}, "elapsed": 0.0, "eta": None, "counters": self.counters,
                       "stuck": False, "last_event": None, "heartbeat": None}
        self.logger = None
        if self.write_files:
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                self.logger = build_logger(self.run_dir, console=console)
                self._events_fh = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
            except Exception:
                self.write_files = False
                self.logger = None
        if self.logger is None:
            self.logger = build_logger(None, console=console)

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self.status["state"]

    def set_state(self, state: str):
        if state not in STATES:
            raise ValueError(f"未知状态 {state!r}，可选 {STATES}")
        self.status["state"] = state

    def event(self, kind: str, **data) -> dict:
        """记录一个事件并更新状态快照；返回事件字典"""
        with self._lock:
            ts = self.clock()
            if self._t0 is None:
                self._t0 = ts
            self._seq += 1
            ev = {"seq": self._seq, "ts": round(ts, 3), "t": round(ts - self._t0, 3), "event": kind, **data}
            self.events.append(ev)
            self.status["elapsed"] = round(ts - self._t0, 3)
            try:
                self._apply(ev)
            except Exception as e:  # 快照更新失败只记一笔
                ev["monitor_error"] = f"{type(e).__name__}: {e}"
            self.status["last_event"] = {k: ev[k] for k in ("seq", "event", "t")}
            self.status["heartbeat"] = round(ts, 3)
            self.status["elapsed"] = round(ts - self._t0, 3)
            self.status["stuck"] = False
            self._persist(ev)
            for fn in list(self.listeners):
                try:
                    fn(ev)
                except Exception:
                    self.listener_errors += 1
            return ev

    def _apply(self, ev: dict):
        k, st, cur = ev["event"], self.status, self.status["current"]
        if k == "run_start":
            st["state"] = "planning"
            st["instruction"] = ev.get("instruction")
            st["engine"] = ev.get("engine")
        elif k == "plan":
            goals = ev.get("goals") or []
            st["goals"] = [{"id": g.get("id"), "description": g.get("description"), "steps": len(g.get("subtasks", [])),
                            "state": "pending"} for g in goals]
            total = sum(len(g.get("subtasks", [])) for g in goals) or len(ev.get("subtasks", []) or [])
            st["progress"] = {"done": 0, "total": total, "pct": 0.0}
            st["state"] = "running"
        elif k == "goal_start":
            cur.update(goal=ev.get("goal"), goal_desc=ev.get("description"))
            self._goal_state(ev.get("goal"), "running")
        elif k == "step_start":
            cur.update(step=ev.get("index"), subtask=ev.get("subtask"), attempt=1, step_t0=ev["t"])
            if st["state"] == "recovering":
                st["state"] = "running"
        elif k == "attempt":
            cur["attempt"] = ev.get("attempt")
        elif k in ("retry", "recover"):
            self.counters["retries" if k == "retry" else "recoveries"] += 1
            st["state"] = "recovering"
            if k == "retry":
                cur["attempt"] = ev.get("next_attempt", (cur.get("attempt") or 1) + 1)
                cur["last_error"] = ev.get("error")
        elif k == "degrade":
            self.counters["degrades"] += 1
            st["degraded"] = ev.get("to")
        elif k == "error":
            self.counters["errors"] += 1
            cur["last_error"] = ev.get("error")
        elif k == "step_end":
            self.counters["steps_ok" if ev.get("ok") else "steps_failed"] += 1
            p = st["progress"]
            p["done"] = min(p["done"] + 1, max(p["total"], p["done"] + 1))
            p["total"] = max(p["total"], p["done"])
            p["pct"] = round(100.0 * p["done"] / p["total"], 1) if p["total"] else 0.0
            if st["state"] == "recovering" and ev.get("ok"):
                st["state"] = "running"
            st["eta"] = self._eta()
        elif k == "goal_end":
            self.counters["goals_ok" if ev.get("ok") else "goals_failed"] += 1
            self._goal_state(ev.get("goal"), "succeeded" if ev.get("ok") else "failed")
        elif k == "run_end":
            st["state"] = "aborted" if ev.get("aborted") else ("succeeded" if ev.get("success") else "failed")
            st["eta"] = 0.0
            st["current"] = {}
            st["metrics"] = self.metrics()
        # 阶段耗时：事件里带 sec 且是已知阶段
        stage = ev.get("stage") or (k if k in STAGES else None)
        if stage in self.timings and isinstance(ev.get("sec"), (int, float)):
            self.timings[stage].append(float(ev["sec"]))

    def _goal_state(self, gid, state):
        for g in self.status["goals"]:
            if g["id"] == gid:
                g["state"] = state

    def _eta(self) -> Optional[float]:
        p = self.status["progress"]
        if not p["done"] or not p["total"] or self._t0 is None:
            return None
        per = self.status["elapsed"] / p["done"]
        return round(per * (p["total"] - p["done"]), 1)

    def _persist(self, ev: dict):
        try:
            msg = self.format_event(ev)
        except Exception:
            msg = f"[{ev.get('event')}]"
        if msg:
            try:
                self.logger.log(_LEVELS.get(ev["event"], logging.INFO), msg)
            except Exception:
                pass
        if not self.write_files:
            return
        try:
            self._events_fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            self._events_fh.flush()
            atomic_write_json(self.run_dir / "status.json", self.snapshot())
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    @contextmanager
    def timer(self, stage: str, **data):
        """with monitor.timer("perceive"): ...  —— 结束时发一个带 sec 的事件（异常时 ok=False 并继续抛出）"""
        t0 = self.clock()
        try:
            yield
        except BaseException as e:
            self.event(stage, sec=round(self.clock() - t0, 4), ok=False, error=f"{type(e).__name__}: {e}", **data)
            raise
        self.event(stage, sec=round(self.clock() - t0, 4), ok=True, **data)

    def record(self, stage: str, sec: float):
        """只记录耗时，不产生事件（高频调用时用）"""
        if stage in self.timings:
            self.timings[stage].append(float(sec))

    def metrics(self) -> dict:
        return {"stages": {s: summarize_durations(v) for s, v in self.timings.items() if v},
                "counters": dict(self.counters), "events": len(self.events)}

    def snapshot(self) -> dict:
        st = dict(self.status)
        st["counters"] = dict(self.counters)
        st["stage_mean_sec"] = {s: summarize_durations(v)["mean"] for s, v in self.timings.items() if v}
        st["stuck_after"] = self.stuck_after
        return st

    def check_stuck(self, now: Optional[float] = None) -> bool:
        """超过 stuck_after 秒没有新事件且未结束 → 卡住（由 watchdog 或 watch 端调用）"""
        if self.status["state"] in TERMINAL_STATES or self.status["heartbeat"] is None:
            return False
        now = self.clock() if now is None else now
        stuck = now - self.status["heartbeat"] > self.stuck_after
        if stuck and not self.status["stuck"]:
            self.status["stuck"] = True
            try:
                self.logger.warning(f"已 {now - self.status['heartbeat']:.0f} 秒没有进展，可能卡住"
                                    f"（当前: {self.status['current'].get('subtask')}）")
                if self.write_files:
                    atomic_write_json(self.run_dir / "status.json", self.snapshot())
            except Exception:
                pass
        return stuck

    def close(self):
        if self.write_files:
            try:
                self._events_fh.close()
                atomic_write_json(self.run_dir / "metrics.json", self.metrics())
            except Exception:
                pass
        for h in list(self.logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            self.logger.removeHandler(h)

    # ------------------------------------------------------------------ #
    @staticmethod
    def format_event(ev: dict) -> str:
        k = ev.get("event")
        if k == "run_start":
            return f"[开始] {ev.get('instruction')}（引擎 {ev.get('engine', '')}）"
        if k == "plan":
            gs = ev.get("goals") or []
            return f"[规划] {len(gs)} 个目标，{sum(len(g.get('subtasks', [])) for g in gs)} 个子任务（来源 {ev.get('source')}）"
        if k == "goal_start":
            return f"[目标 {ev.get('goal')}] {ev.get('description')}"
        if k == "step_start":
            return f"  [子任务 {ev.get('index')}] {ev.get('subtask')}"
        if k == "retry":
            return (f"    ↻ 第{ev.get('next_attempt')}次尝试（{ev.get('error_class')}: {ev.get('error')}），"
                    f"等待 {ev.get('delay', 0):g} 秒")
        if k == "recover":
            return f"    ⚕ 恢复动作 {ev.get('action')}：{ev.get('result')}"
        if k == "degrade":
            return f"    ⚠ 降级 {ev.get('component')} → {ev.get('to')}：{ev.get('reason')}"
        if k == "step_end":
            return f"  {'✓' if ev.get('ok') else '✗'} 子任务 {ev.get('index')} 用时 {ev.get('sec')} 秒" + (
                "" if ev.get("ok") else f"：{ev.get('error')}")
        if k == "goal_end":
            return f"[目标 {ev.get('goal')}] {'完成' if ev.get('ok') else '失败'}" + (
                f"：{ev.get('reason')}" if ev.get("reason") else "")
        if k == "error":
            return f"[错误] {ev.get('error')}"
        if k == "run_end":
            return f"[结束] {'成功' if ev.get('success') else '失败'}，用时 {ev.get('sec')} 秒"
        return ""


# ---------------------------------------------------------------------- #
# 查看端：python -m agent.monitor
# ---------------------------------------------------------------------- #
def read_status(run_dir) -> Optional[dict]:
    p = Path(run_dir) / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_latest_run(base) -> Optional[Path]:
    base = Path(base)
    if not base.exists():
        return None
    cands = [p.parent for p in base.rglob("status.json")]
    return max(cands, key=lambda p: (p / "status.json").stat().st_mtime) if cands else None


def _bar(pct: float, width: int = 30) -> str:
    n = int(round(width * max(0.0, min(100.0, pct)) / 100))
    return "█" * n + "░" * (width - n)


def render_status(st: Optional[dict], now: Optional[float] = None) -> str:
    if not st:
        return "（还没有状态文件，Agent 可能尚未开始）"
    now = time.time() if now is None else now
    p = st.get("progress") or {}
    cur = st.get("current") or {}
    c = st.get("counters") or {}
    lines = [f"指令：{st.get('instruction')}",
             f"状态：{st.get('state')}" + ("  ⚠ 已降级: " + str(st.get("degraded")) if st.get("degraded") else ""),
             f"进度：{_bar(p.get('pct', 0))} {p.get('done', 0)}/{p.get('total', 0)}（{p.get('pct', 0)}%）",
             f"已用：{st.get('elapsed', 0):.1f} 秒" + (f"   预计剩余：{st['eta']:.1f} 秒" if st.get("eta") else "")]
    hb = st.get("heartbeat")
    if hb is not None and st.get("state") not in TERMINAL_STATES:
        idle = now - hb
        if st.get("stuck") or idle > float(st.get("stuck_after") or 60):
            lines.append(f"!! 已 {idle:.0f} 秒没有新事件，可能卡住")
    if cur.get("subtask"):
        lines.append(f"当前：目标 {cur.get('goal')} / 子任务 {cur.get('step')}「{cur.get('subtask')}」"
                     f" 第 {cur.get('attempt', 1)} 次尝试")
    if cur.get("last_error"):
        lines.append(f"最近错误：{cur['last_error']}")
    lines.append(f"计数：重试 {c.get('retries', 0)}  恢复 {c.get('recoveries', 0)}  错误 {c.get('errors', 0)}  "
                 f"降级 {c.get('degrades', 0)}  子任务成功 {c.get('steps_ok', 0)} / 失败 {c.get('steps_failed', 0)}")
    for g in st.get("goals") or []:
        mark = {"succeeded": "✓", "failed": "✗", "running": "▶"}.get(g.get("state"), "·")
        lines.append(f"  {mark} {g.get('id')} {g.get('description')}（{g.get('steps')} 步）")
    sm = st.get("stage_mean_sec") or {}
    if sm:
        lines.append("阶段平均耗时：" + "  ".join(f"{k} {v:.2f}s" for k, v in sm.items()))
    return "\n".join(lines)


def watch(run_dir=None, interval: float = 1.0, once: bool = False, out: Callable = print,
          base: str = "runs", max_loops: Optional[int] = None, sleep: Callable = time.sleep) -> int:
    loops = 0
    while True:
        rd = Path(run_dir) if run_dir else find_latest_run(base)
        st = read_status(rd) if rd else None
        text = render_status(st)
        if not once:
            out("\033[2J\033[H" + f"运行目录：{rd}\n" + text)
        else:
            out(f"运行目录：{rd}\n" + text)
        loops += 1
        if once or (st and st.get("state") in TERMINAL_STATES) or (max_loops and loops >= max_loops):
            return 0 if st else 1
        sleep(interval)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m agent.monitor", description="实时查看 GUI Agent 执行状态")
    ap.add_argument("run_dir", nargs="?", default=None, help="运行目录（默认 runs/ 下最新的一次）")
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent / "runs"))
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="只打印一次")
    args = ap.parse_args(argv)
    try:
        return watch(args.run_dir, args.interval, args.once, base=args.base)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
