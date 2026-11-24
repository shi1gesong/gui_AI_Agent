"""第5周：--act vlm 接入（VLMClient、执行器坐标动作、GUIAgent vlm 模式、HTTP 服务、CLI 参数）—— mock 用例

VLM 本身用“看得见模拟桌面的替身”代替：它从子任务里取出引号中的目标，在模拟桌面上找到对应元素，
返回 0~1000 归一化坐标 —— 等价于一个完全正确的多模态模型，用来验证 坐标换算 → 执行 → 检查 的闭环。
"""
import base64
import io
import json
import re
import threading
import urllib.request

import pytest
from PIL import Image

from agent.basic_tasks import TaskContext, build_tasks, run_task
from agent.executor import ControllerExecutor, DryRunExecutor
from agent.gui_agent import ACT_MODES, GUIAgent, _describe_action
from agent.planner import HybridPlanner, RulePlanner
from agent.vlm_client import DEFAULT_VLM_URL, VLMClient, encode_image
from fakes import RecordingController
from finetune.serve import decode_image, make_server
from perception.screen_parser import Observation, ScreenPerceiver
from sim_desktop import SimDesktop

QUOTE_RE = re.compile(r"[“「\"]([^”」\"]+)[”」\"]")


class PointController(RecordingController):
    def click(self, x, y, button="left", clicks=1):
        self._rec("click", x, y, button)

    def move_to(self, x, y, duration=0.2):
        self._rec("move_to", x, y)

    def drag_to(self, sx, sy, ex, ey, duration=0.5):
        self._rec("drag_to", sx, sy, ex, ey)


def oracle_post(sim, log=None, wrong=False):
    """模拟 finetune/serve.py：根据子任务在模拟桌面上找目标，返回归一化坐标"""
    def post(url, payload, timeout):
        assert url.endswith("/act")
        Image.open(io.BytesIO(base64.b64decode(payload["image_b64"])))  # 必须是合法图片
        if log is not None:
            log.append(payload)
        st = payload["sub_task"]
        quoted = QUOTE_RE.findall(st)
        w, h = sim.logical
        for text, (x1, y1, x2, y2) in sim.elements_logical():
            if quoted and text == quoted[0]:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if wrong:
                    cx, cy = 5, h - 5
                acts = [{"action": "click", "x": round(cx / (w - 1) * 1000), "y": round(cy / (h - 1) * 1000)}]
                if "输入" in st and len(quoted) > 1:
                    acts.append({"action": "type", "text": quoted[1]})
                return {"ok": True, "actions": acts, "raw": json.dumps(acts, ensure_ascii=False), "sec": 0.1}
        return {"ok": False, "actions": [], "raw": "我不知道", "sec": 0.1, "error": "没有可识别的动作"}
    return post


def make(tmp_path, dpi=1.0, logical=(1920, 1080), post=None, **kw):
    sim = SimDesktop(tmp_path, logical=logical, dpi=dpi)
    apps = sim.app_manager()
    perceiver = ScreenPerceiver(capture=sim, ocr=sim, logical_size_fn=sim.logical_size, region_fn=apps.screen_regions)
    executor = ControllerExecutor(sim, app_manager=apps, sleep=lambda s: None)
    vlm = VLMClient(post=post or oracle_post(sim))
    agent = GUIAgent(perceiver, executor, planner=HybridPlanner("rule", RulePlanner(app_resolver=apps.resolve_app)),
                     act_mode="vlm", vlm=vlm, sleep=lambda s: None, log=lambda *a: None,
                     trace_dir=str(tmp_path / "trace"), **kw)
    ctx = TaskContext(apps, tmp_path / "work", sim.chat_log, sleep=lambda s: None)
    return agent, sim, ctx


def task(tid, ctx):
    return {x.id: x for x in build_tasks(ctx.work_dir)}[tid]


def obs_for(tmp_path, image_size=(1920, 1080), logical=(1920, 1080), **extra):
    p = tmp_path / "s.png"
    Image.new("RGB", image_size, "white").save(p)
    return Observation(elements=[], screenshot_path=str(p), image_size=image_size, logical_size=logical,
                       extra=extra)


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestVLMAgentLoop:
    def test_t4_send_message_with_vlm(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"], [s.get("error") for s in r["steps"]]
        assert sim.chat["sent"] == [("张三", "今晚7点开会")]

    def test_vlm_used_only_for_screen_steps(self, tmp_path):
        log = []
        agent, sim, ctx = make(tmp_path)
        agent.vlm.post = oracle_post(sim, log)
        r = agent.run("给张三发送消息：你好")
        sources = [s["source"] for s in r["steps"]]
        assert sources == ["rule", "rule", "vlm", "vlm", "vlm", "rule"]
        assert len(log) == 3

    def test_vlm_click_point_recorded(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("给张三发送消息：你好")
        click = r["steps"][2]
        assert click["attempts"][0]["detail"]["points"][0]["point"] == [205, 180]
        assert click["attempts"][0]["vlm"]["raw"].startswith("[")

    def test_trace_has_no_raw_in_action(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        r = agent.run("给张三发送消息：你好")
        assert "vlm_raw" not in r["steps"][2]["action"]

    def test_t1_no_vlm_call(self, tmp_path):
        log = []
        agent, sim, ctx = make(tmp_path)
        agent.vlm.post = oracle_post(sim, log)
        assert run_task(agent, task("T1", ctx), ctx)["success"] and log == []

    def test_instruction_forwarded(self, tmp_path):
        log = []
        agent, sim, ctx = make(tmp_path)
        agent.vlm.post = oracle_post(sim, log)
        agent.run("给张三发送消息：你好")
        assert log[0]["instruction"] == "给张三发送消息：你好" and log[0]["sub_task"] == "点击联系人“张三”"

    def test_vlm_failure_retried_then_fails(self, tmp_path):
        agent, sim, ctx = make(tmp_path, post=lambda u, p, t: {"ok": False, "actions": [], "raw": "??"})
        r = agent.run("给张三发送消息：你好")
        assert not r["success"] and len(r["steps"][2]["attempts"]) == 2

    def test_wrong_coordinates_caught_by_checker(self, tmp_path):
        agent, sim, ctx = make(tmp_path)
        agent.vlm.post = oracle_post(sim, wrong=True)
        r = run_task(agent, task("T4", ctx), ctx)
        assert not r["success"] and sim.chat["sent"] == []

    def test_describe_point_action(self):
        assert _describe_action({"action_type": "click", "point": [3, 4]}) == "点击「(3, 4)」"
        assert "右键" in _describe_action({"action_type": "click", "point": [3, 4], "button": "right"})

    def test_act_modes(self):
        assert "vlm" in ACT_MODES


@pytest.mark.multi_resolution
class TestVLMMultiResolution:
    @pytest.mark.parametrize("dpi", [1.0, 1.25, 1.5, 2.0])
    def test_t4_under_dpi_scaling(self, tmp_path, dpi):
        """截图像素 = 逻辑坐标 × DPI；VLM 坐标是归一化的，换算时用逻辑尺寸，所以任何缩放都能点准"""
        agent, sim, ctx = make(tmp_path, dpi=dpi)
        r = run_task(agent, task("T4", ctx), ctx)
        assert r["success"] and sim.chat["sent"] == [("张三", "今晚7点开会")]

    @pytest.mark.parametrize("logical", [(1366, 768), (2560, 1440), (3840, 2160)])
    def test_t4_resolutions(self, tmp_path, logical):
        agent, sim, ctx = make(tmp_path, logical=logical)
        assert run_task(agent, task("T4", ctx), ctx)["success"]

    @pytest.mark.parametrize("image,logical", [((3840, 2160), (2560, 1440)), ((2560, 1440), (2560, 1440)),
                                               ((1920, 1080), (1280, 720))])
    def test_client_maps_to_logical(self, tmp_path, image, logical):
        c = VLMClient(post=lambda u, p, t: {"actions": [{"action": "click", "x": 1000, "y": 1000}]})
        a = c.act("点击", "任务", obs_for(tmp_path, image, logical))
        assert a["point"] == [logical[0] - 1, logical[1] - 1]

    def test_client_downscales_image(self, tmp_path):
        sent = []
        c = VLMClient(post=lambda u, p, t: sent.append(p) or {"actions": [{"action": "wait", "seconds": 1}]})
        c.act("等", "任务", obs_for(tmp_path, (2560, 1440)))
        img = Image.open(io.BytesIO(base64.b64decode(sent[0]["image_b64"])))
        assert img.size == (1280, 720) and sent[0]["screen_size"] == [2560, 1440]


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestVLMClient:
    def test_url_and_payload(self, tmp_path):
        seen = {}

        def post(url, payload, timeout):
            seen.update(url=url, timeout=timeout, keys=sorted(payload))
            return {"actions": [{"action": "click", "x": 0, "y": 0}], "raw": "[]"}
        VLMClient("http://h:1/", timeout=9, post=post).act("点", "任务", obs_for(tmp_path))
        assert seen == {"url": "http://h:1/act", "timeout": 9,
                        "keys": ["image_b64", "instruction", "screen_size", "sub_task"]}

    def test_client_side_parse_of_raw(self, tmp_path):
        c = VLMClient(post=lambda u, p, t: {"actions": [], "raw": '```json\n[{"action":"key","keys":["enter"]}]\n```'})
        a = c.act("回车", "任务", obs_for(tmp_path))
        assert a["action_type"] == "key" and a["input_text"] == "enter"

    def test_offset_for_window_region(self, tmp_path):
        c = VLMClient(post=lambda u, p, t: {"actions": [{"action": "click", "x": 0, "y": 0}]})
        assert c.act("点", "任务", obs_for(tmp_path, offset=(100, 50)))["point"] == [100, 50]

    def test_last_record(self, tmp_path):
        c = VLMClient(post=lambda u, p, t: {"actions": [{"action": "wait", "seconds": 1}], "raw": "r", "sec": 0.5})
        c.act("等", "任务", obs_for(tmp_path))
        assert c.last["raw"] == "r" and c.last["sec"] == 0.5

    def test_default_url(self):
        assert VLMClient().base_url == DEFAULT_VLM_URL == "http://localhost:8765"

    def test_encode_image_roundtrip(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGB", (100, 50), "red").save(p)
        img = Image.open(io.BytesIO(base64.b64decode(encode_image(str(p)))))
        assert img.size == (100, 50)


@pytest.mark.exception
class TestVLMClientErrors:
    def test_no_screenshot(self):
        with pytest.raises(ValueError):
            VLMClient(post=lambda *a: {}).act("点", "任务", Observation(elements=[]))

    def test_no_size(self, tmp_path):
        o = obs_for(tmp_path)
        o.logical_size, o.image_size = (0, 0), (0, 0)
        with pytest.raises(ValueError):
            VLMClient(post=lambda *a: {}).act("点", "任务", o)

    def test_non_dict_response(self, tmp_path):
        with pytest.raises(RuntimeError):
            VLMClient(post=lambda *a: ["x"]).act("点", "任务", obs_for(tmp_path))

    def test_empty_actions_retryable_invalid(self, tmp_path):
        a = VLMClient(post=lambda *a: {"actions": [], "raw": "不知道"}).act("点", "任务", obs_for(tmp_path))
        assert a["_invalid"] and a["_retryable"]

    def test_server_down_becomes_decision_failure(self, tmp_path):
        def post(*a):
            raise ConnectionError("refused")
        agent, sim, ctx = make(tmp_path, post=post)
        r = agent.run("给张三发送消息：你好")
        assert not r["success"] and "决策失败" in r["steps"][2]["error"]

    def test_agent_requires_vlm(self):
        with pytest.raises(ValueError):
            GUIAgent(None, DryRunExecutor(), act_mode="vlm")


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestExecutorPoints:
    def _ex(self):
        ctl = PointController()
        return ControllerExecutor(ctl, sleep=lambda s: None), ctl

    def test_click_point_skips_grounding(self):
        e, ctl = self._ex()
        assert e({"action_type": "click", "point": [10, 20]}, [])["ok"] and ctl.calls == [("click", 10, 20, "left")]

    def test_right_click(self):
        e, ctl = self._ex()
        e({"action_type": "click", "point": [1, 2], "button": "right"}, [])
        assert ctl.calls == [("click", 1, 2, "right")]

    def test_double_click_point(self):
        e, ctl = self._ex()
        e({"action_type": "double_click", "point": [5, 6]}, [])
        assert ctl.calls == [("double_click", 5, 6)]

    def test_move_and_drag(self):
        e, ctl = self._ex()
        e({"action_type": "move", "point": [1, 1]}, [])
        e({"action_type": "drag", "point": [1, 1], "end_point": [9, 9]}, [])
        assert ctl.calls == [("move_to", 1, 1), ("drag_to", 1, 1, 9, 9)]

    def test_type_with_point_clicks_first(self):
        e, ctl = self._ex()
        e({"action_type": "type", "point": [3, 3], "input_text": "hi"}, [])
        assert ctl.calls == [("click", 3, 3, "left"), ("type_text", "hi")]

    def test_scroll_at_point(self):
        e, ctl = self._ex()
        e({"action_type": "scroll", "amount": -3, "point": [7, 8]}, [])
        assert ctl.calls == [("scroll", -3, 7, 8)]

    def test_follow_up_chain(self):
        e, ctl = self._ex()
        a = {"action_type": "click", "point": [1, 1],
             "follow_up": [{"action_type": "type", "input_text": "x"}, {"action_type": "key", "input_text": "enter"}]}
        assert e(a, [])["ok"] and [c[0] for c in ctl.calls] == ["click", "type_text", "press_key"]

    def test_float_point_rounded(self):
        e, ctl = self._ex()
        e({"action_type": "click", "point": [1.6, 2.4]}, [])
        assert ctl.calls == [("click", 2, 2, "left")]

    def test_dry_run_reports_point(self):
        r = DryRunExecutor()({"action_type": "click", "point": [4, 5]}, [])
        assert r["ok"] and r["detail"]["point"] == [4, 5]


@pytest.mark.exception
class TestExecutorPointErrors:
    @pytest.mark.parametrize("pt", [[1], "1,2", [1, "a"], [True, 1], {"x": 1}])
    def test_bad_point(self, pt):
        e = ControllerExecutor(PointController(), sleep=lambda s: None)
        r = e({"action_type": "click", "point": pt}, [])
        assert not r["ok"] and "坐标格式错误" in r["error"]

    def test_drag_needs_end(self):
        e = ControllerExecutor(PointController(), sleep=lambda s: None)
        assert not e({"action_type": "drag", "point": [1, 1]}, [])["ok"]

    def test_move_without_target(self):
        e = ControllerExecutor(PointController(), sleep=lambda s: None)
        assert not e({"action_type": "move", "target_text": "不存在"}, [])["ok"]


# --------------------------------------------------------------------------- #
class FakeActor:
    name, prompt = "lora", "v1_schema"

    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def predict(self, image, instruction, sub_task=None, prompt=None, screen_size=None):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.seen.append((image.size, instruction, sub_task, screen_size))
        return {"ok": True, "actions": [{"action": "click", "x": 1, "y": 2}], "raw": "[...]", "sec": 0.01}


@pytest.fixture
def server():
    started = []

    def start(actor):
        srv = make_server(actor, "127.0.0.1", 0)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"
    yield start
    for s in started:
        s.shutdown()
        s.server_close()


def _b64(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, "blue").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _req(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.mark.functional
class TestServe:
    def test_health(self, server):
        code, body = _req(server(FakeActor()) + "/health")
        assert code == 200 and body == {"ok": True, "name": "lora", "prompt": "v1_schema"}

    def test_act(self, server):
        actor = FakeActor()
        code, body = _req(server(actor) + "/act", {"image_b64": _b64(), "instruction": "任务", "sub_task": "点"})
        assert code == 200 and body["actions"] == [{"action": "click", "x": 1, "y": 2}]
        assert actor.seen == [((64, 48), "任务", "点", (64, 48))]

    def test_client_against_real_server(self, server, tmp_path):
        import requests  # noqa: F401  默认 post 用 requests
        url = server(FakeActor())
        a = VLMClient(url).act("点", "任务", obs_for(tmp_path, (1000, 1000), (1001, 1001)))
        assert a["point"] == [1, 2]

    def test_data_url_prefix_accepted(self):
        assert decode_image("data:image/png;base64," + _b64((3, 3))).size == (3, 3)


@pytest.mark.exception
class TestServeErrors:
    def test_missing_image(self, server):
        code, body = _req(server(FakeActor()) + "/act", {"instruction": "x"})
        assert code == 400 and "image_b64" in body["error"]

    def test_bad_base64(self, server):
        code, _ = _req(server(FakeActor()) + "/act", {"image_b64": "!!!", "instruction": "x"})
        assert code == 400

    def test_not_an_image(self, server):
        code, body = _req(server(FakeActor()) + "/act",
                          {"image_b64": base64.b64encode(b"hello").decode(), "instruction": "x"})
        assert code == 400 and "无法解码图片" in body["error"]

    def test_empty_instruction(self, server):
        code, _ = _req(server(FakeActor()) + "/act", {"image_b64": _b64()})
        assert code == 400

    def test_invalid_json(self, server):
        code, _ = _req(server(FakeActor()) + "/act", raw=b"{oops")
        assert code == 400

    def test_model_error_is_500(self, server):
        code, body = _req(server(FakeActor(fail=True)) + "/act", {"image_b64": _b64(), "instruction": "x"})
        assert code == 500 and "out of memory" in body["error"]

    def test_unknown_path(self, server):
        code, _ = _req(server(FakeActor()) + "/nope", {"a": 1})
        assert code == 404


# --------------------------------------------------------------------------- #
@pytest.mark.functional
class TestCLIAndFactory:
    def test_cli_accepts_vlm(self):
        from agent.cli import build_parser, make_config
        args = build_parser().parse_args(["--act", "vlm", "--vlm-url", "http://x:1", "打开记事本"])
        cfg = make_config(args)
        assert cfg.act_mode == "vlm" and cfg.vlm_url == "http://x:1" and cfg.vlm_ocr is False

    def test_null_ocr(self):
        from agent.factory import NullOCR
        assert NullOCR().detect("x.png") == []
