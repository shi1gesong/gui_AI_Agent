"""第3周：data/dataset_loader.py（Mind2Web / WebArena / ScreenAgent）—— mock 用例

HuggingFace datasets 与 git clone 均用替身，不联网；WebArena 用临时目录构造 config_files 结构。
"""
import importlib
import json
import os
import subprocess
import sys
import types

import pytest


@pytest.fixture(scope="module")
def dl(tmp_path_factory):
    """dataset_loader 在 import 时会在当前目录创建 ./data_cache，这里先切到临时目录再导入，避免污染仓库。"""
    old = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("import_cwd"))
    try:
        mod = importlib.import_module("data.dataset_loader")
    finally:
        os.chdir(old)
    return mod


@pytest.fixture
def root(dl, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DATA_ROOT", tmp_path)
    return tmp_path


def m2w_item(i, n_actions=2, **over):
    item = {
        "annotation_id": f"ann-{i}",
        "confirmed_task": f"Find flights {i}",
        "actions": [{
            "operation": {"op": "CLICK" if k % 2 == 0 else "TYPE", "value": "" if k % 2 == 0 else "NYC"},
            "pos_candidates": [{"tag": "button", "attributes": '{"id": "btn"}'}],
            "raw_html": "<html>" + "x" * 5000 + "</html>",
            "cleaned_html": "<html/>",
        } for k in range(n_actions)],
    }
    item.update(over)
    return item


@pytest.fixture
def hf(monkeypatch):
    state = types.SimpleNamespace(items=[], exc=None, calls=[])

    def load_dataset(name, split=None, streaming=False):
        state.calls.append((name, split, streaming))
        if state.exc:
            raise state.exc
        return iter(state.items)
    mod = types.ModuleType("datasets")
    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return state


@pytest.fixture
def git(monkeypatch, dl):
    state = types.SimpleNamespace(calls=[], returncode=0, stderr="")

    def run(cmd, capture_output=False, text=False):
        state.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, state.returncode, "", state.stderr)
    monkeypatch.setattr(dl.subprocess, "run", run)
    return state


def make_webarena(root, files):
    cfg = root / "webarena" / "config_files"
    for rel, content in files.items():
        p = cfg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return cfg


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.functional
class TestDatasetFunctional:
    def test_mind2web_requests_streaming_train_split(self, dl, root, hf):
        dl.download_mind2web(sample_limit=1)
        assert hf.calls == [("osunlp/Mind2Web", "train", True)]

    def test_mind2web_sample_count(self, dl, root, hf):
        hf.items = [m2w_item(i) for i in range(10)]
        assert len(dl.download_mind2web(sample_limit=5)) == 5

    def test_mind2web_unified_fields(self, dl, root, hf):
        hf.items = [m2w_item(0)]
        s = dl.download_mind2web(sample_limit=1)[0]
        assert set(s) == {"task_id", "instruction", "screenshot_path", "action_sequence", "source"}
        assert s["task_id"] == "ann-0" and s["instruction"] == "Find flights 0"
        assert s["source"] == "mind2web" and s["screenshot_path"] is None

    def test_mind2web_simplified_action(self, dl, root, hf):
        hf.items = [m2w_item(0)]
        acts = dl.download_mind2web(sample_limit=1)[0]["action_sequence"]
        assert acts[1] == {"op": "TYPE", "value": "NYC", "target_tag": "button", "target_attributes": '{"id": "btn"}'}

    def test_mind2web_drops_raw_html(self, dl, root, hf):
        hf.items = [m2w_item(0)]
        dl.download_mind2web(sample_limit=1)
        text = (root / "mind2web_processed.jsonl").read_text(encoding="utf-8")
        assert "raw_html" not in text and "xxxxx" not in text

    def test_mind2web_writes_jsonl(self, dl, root, hf):
        hf.items = [m2w_item(i) for i in range(3)]
        dl.download_mind2web(sample_limit=3)
        assert [r["task_id"] for r in read_jsonl(root / "mind2web_processed.jsonl")] == ["ann-0", "ann-1", "ann-2"]

    def test_mind2web_chinese_not_escaped(self, dl, root, hf):
        hf.items = [m2w_item(0, confirmed_task="查询北京天气")]
        dl.download_mind2web(sample_limit=1)
        assert "查询北京天气" in (root / "mind2web_processed.jsonl").read_text(encoding="utf-8")

    def test_webarena_extracts_intent(self, dl, root):
        make_webarena(root, {"examples/1.json": {"task_id": 1, "intent": "What is the top-1 best selling product?"}})
        s = dl.preprocess_webarena_tasks()
        assert s == [{"task_id": 1, "instruction": "What is the top-1 best selling product?",
                      "screenshot_path": None, "action_sequence": [], "source": "webarena"}]

    def test_webarena_recursive_search(self, dl, root):
        make_webarena(root, {"a.json": {"task_id": 1, "intent": "x"}, "examples/deep/2.json": {"task_id": 2, "intent": "y"}})
        assert sorted(s["task_id"] for s in dl.preprocess_webarena_tasks()) == [1, 2]

    def test_webarena_writes_jsonl(self, dl, root):
        make_webarena(root, {f"examples/{i}.json": {"task_id": i, "intent": f"t{i}"} for i in range(4)})
        dl.preprocess_webarena_tasks()
        assert len(read_jsonl(root / "webarena_processed.jsonl")) == 4

    def test_clone_screenagent_command(self, dl, root, git):
        target = dl.clone_screenagent_dataset()
        assert git.calls[0] == ["git", "clone", "--depth", "1",
                                "https://github.com/niuzaisheng/ScreenAgent.git", str(root / "ScreenAgent")]
        assert target == root / "ScreenAgent"

    def test_clone_webarena_command(self, dl, root, git):
        dl.clone_webarena_repo()
        assert git.calls[0][-2:] == ["https://github.com/web-arena-x/webarena.git", str(root / "webarena")]

    def test_clone_skips_when_exists(self, dl, root, git):
        (root / "ScreenAgent").mkdir()
        assert dl.clone_screenagent_dataset() == root / "ScreenAgent"
        assert git.calls == []

    def test_clone_webarena_skips_when_exists(self, dl, root, git):
        (root / "webarena").mkdir()
        dl.clone_webarena_repo()
        assert git.calls == []


@pytest.mark.boundary
class TestDatasetBoundary:
    def test_mind2web_limit_zero(self, dl, root, hf):
        hf.items = [m2w_item(0)]
        assert dl.download_mind2web(sample_limit=0) == []
        assert (root / "mind2web_processed.jsonl").read_text() == ""

    def test_mind2web_limit_exceeds_dataset(self, dl, root, hf):
        hf.items = [m2w_item(i) for i in range(3)]
        assert len(dl.download_mind2web(sample_limit=200)) == 3

    def test_mind2web_missing_annotation_id_fallback(self, dl, root, hf):
        item = m2w_item(0)
        del item["annotation_id"]
        hf.items = [m2w_item(9), item]
        assert dl.download_mind2web(sample_limit=2)[1]["task_id"] == "mind2web_1"

    def test_mind2web_no_actions(self, dl, root, hf):
        hf.items = [m2w_item(0, n_actions=0)]
        assert dl.download_mind2web(sample_limit=1)[0]["action_sequence"] == []

    def test_mind2web_empty_pos_candidates(self, dl, root, hf):
        item = m2w_item(0, n_actions=1)
        item["actions"][0]["pos_candidates"] = []
        hf.items = [item]
        a = dl.download_mind2web(sample_limit=1)[0]["action_sequence"][0]
        assert a["target_tag"] is None and a["target_attributes"] is None

    def test_mind2web_missing_operation(self, dl, root, hf):
        item = m2w_item(0, n_actions=1)
        del item["actions"][0]["operation"]
        hf.items = [item]
        a = dl.download_mind2web(sample_limit=1)[0]["action_sequence"][0]
        assert a["op"] is None and a["value"] is None

    def test_mind2web_missing_task_text(self, dl, root, hf):
        item = m2w_item(0)
        del item["confirmed_task"]
        hf.items = [item]
        assert dl.download_mind2web(sample_limit=1)[0]["instruction"] == ""

    def test_mind2web_long_action_sequence(self, dl, root, hf):
        hf.items = [m2w_item(0, n_actions=60)]
        assert len(dl.download_mind2web(sample_limit=1)[0]["action_sequence"]) == 60

    def test_webarena_excludes_test_raw(self, dl, root):
        make_webarena(root, {"test.raw.json": [{"task_id": 0}], "examples/1.json": {"task_id": 1, "intent": "x"}})
        assert [s["task_id"] for s in dl.preprocess_webarena_tasks()] == [1]

    def test_webarena_missing_task_id_uses_stem(self, dl, root):
        make_webarena(root, {"examples/42.json": {"intent": "x"}})
        assert dl.preprocess_webarena_tasks()[0]["task_id"] == "42"

    def test_webarena_task_id_zero_kept(self, dl, root):
        make_webarena(root, {"examples/0.json": {"task_id": 0, "intent": "x"}})
        assert dl.preprocess_webarena_tasks()[0]["task_id"] == 0

    def test_webarena_missing_intent(self, dl, root):
        make_webarena(root, {"examples/1.json": {"task_id": 1}})
        assert dl.preprocess_webarena_tasks()[0]["instruction"] == ""

    def test_webarena_empty_config_dir(self, dl, root):
        (root / "webarena" / "config_files").mkdir(parents=True)
        assert dl.preprocess_webarena_tasks() == []
        assert (root / "webarena_processed.jsonl").exists()

    def test_webarena_ignores_non_json(self, dl, root):
        make_webarena(root, {"README.md": "# x", "examples/1.json": {"task_id": 1, "intent": "x"}})
        assert len(dl.preprocess_webarena_tasks()) == 1

    @pytest.mark.xfail(strict=True, reason="已知问题：dataset_loader 在 import 时就在当前工作目录创建 data_cache/（模块级副作用）")
    def test_import_has_no_side_effect(self, tmp_path):
        from conftest import REPO_ROOT
        code = "import sys; sys.path.insert(0, %r); import data.dataset_loader" % str(REPO_ROOT)
        subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
        assert not (tmp_path / "data_cache").exists()

    @pytest.mark.xfail(strict=True, reason="已知问题：ScreenAgent 仅克隆仓库，没有实现数据解析，无法产出统一格式样本")
    def test_screenagent_preprocess_exists(self, dl):
        assert hasattr(dl, "preprocess_screenagent")


@pytest.mark.exception
class TestDatasetException:
    def test_mind2web_download_failure_returns_empty(self, dl, root, hf):
        hf.exc = ConnectionError("Network unreachable")
        assert dl.download_mind2web() == []

    def test_mind2web_failure_writes_no_file(self, dl, root, hf):
        hf.exc = OSError("SSL certificate mismatch")
        dl.download_mind2web()
        assert not (root / "mind2web_processed.jsonl").exists()

    def test_mind2web_without_datasets_package(self, dl, root, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError):
            dl.download_mind2web()

    def test_webarena_without_clone_returns_empty(self, dl, root):
        assert dl.preprocess_webarena_tasks() == []

    def test_webarena_malformed_json_skipped(self, dl, root):
        make_webarena(root, {"examples/bad.json": "{not json", "examples/1.json": {"task_id": 1, "intent": "x"}})
        assert [s["task_id"] for s in dl.preprocess_webarena_tasks()] == [1]

    def test_webarena_json_list_skipped(self, dl, root):
        make_webarena(root, {"examples/list.json": [1, 2, 3]})
        assert dl.preprocess_webarena_tasks() == []

    def test_webarena_non_utf8_skipped(self, dl, root):
        make_webarena(root, {"examples/gbk.json": '{"intent": "中文"}'.encode("gbk")})
        assert dl.preprocess_webarena_tasks() == []

    def test_clone_failure_returns_none(self, dl, root, git):
        git.returncode, git.stderr = 128, "fatal: unable to access"
        assert dl.clone_screenagent_dataset() is None

    def test_clone_webarena_failure_returns_none(self, dl, root, git):
        git.returncode = 1
        assert dl.clone_webarena_repo() is None
