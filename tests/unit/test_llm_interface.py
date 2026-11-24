"""第3周：agent/llm_interface.py（Ollama API 模式 / 本地 Qwen-VL 模式 / 工厂函数）—— mock 用例

Ollama 用 requests.post 替身模拟；本地 Qwen-VL 用 transformers 替身模拟，不下载权重、不占 GPU。
真实 Ollama 调用见 tests/real/test_real_llm.py（llm 标记）。
"""
import base64
import sys
import types

import pytest
import requests

import llm_interface as li
from llm_interface import BaseLLMInterface, LocalQwenVLInterface, OllamaAPIInterface, build_llm_interface


class FakeResp:
    def __init__(self, payload=None, status=200, json_error=None):
        self._payload = payload if payload is not None else {"response": "ok"}
        self.status_code = status
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.fixture
def post(monkeypatch):
    calls = []
    state = types.SimpleNamespace(resp=FakeResp(), exc=None, calls=calls)

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if state.exc:
            raise state.exc
        return state.resp
    monkeypatch.setattr(requests, "post", fake_post)
    return state


@pytest.fixture
def fake_transformers(monkeypatch):
    rec = types.SimpleNamespace(tok_args=None, model_args=None, chat_queries=[], list_format=[])

    class Tok:
        @classmethod
        def from_pretrained(cls, name, **kw):
            rec.tok_args = (name, kw)
            return cls()

        def from_list_format(self, items):
            rec.list_format.append(items)
            return "<img>" + items[0]["image"] + "</img>" + items[1]["text"]

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kw):
            rec.model_args = (name, kw)
            return cls()

        def eval(self):
            return self

        def chat(self, tokenizer, query, history=None):
            rec.chat_queries.append((query, history))
            return "模型回复", []

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer, mod.AutoModelForCausalLM = Tok, Model
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return rec


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "s.png"
    p.write_bytes(b"\x89PNG fake bytes")
    return p


@pytest.mark.functional
class TestLLMFunctional:
    def test_factory_api_mode(self):
        assert isinstance(build_llm_interface("api"), OllamaAPIInterface)

    def test_factory_default_is_api(self):
        assert isinstance(build_llm_interface(), OllamaAPIInterface)

    def test_factory_local_mode(self, fake_transformers):
        assert isinstance(build_llm_interface("local"), LocalQwenVLInterface)

    def test_factory_passes_kwargs(self):
        llm = build_llm_interface("api", model_name="qwen2.5:1.5b", base_url="http://172.20.192.1:11434")
        assert llm.model_name == "qwen2.5:1.5b" and llm.base_url == "http://172.20.192.1:11434"

    def test_ollama_defaults(self):
        llm = OllamaAPIInterface()
        assert llm.model_name == "qwen3:8b" and llm.base_url == "http://localhost:11434"

    def test_ollama_posts_to_generate_endpoint(self, post):
        OllamaAPIInterface(base_url="http://h:1").generate("hi")
        assert post.calls[0]["url"] == "http://h:1/api/generate"

    def test_ollama_payload_text_only(self, post):
        OllamaAPIInterface(model_name="m").generate("你好")
        assert post.calls[0]["json"] == {"model": "m", "prompt": "你好", "stream": False}

    def test_ollama_returns_response_field(self, post):
        post.resp = FakeResp({"response": "我是Qwen"})
        assert OllamaAPIInterface().generate("x") == "我是Qwen"

    def test_ollama_image_base64_encoded(self, post, png):
        OllamaAPIInterface().generate("看图", image_path=str(png))
        assert post.calls[0]["json"]["images"] == [base64.b64encode(png.read_bytes()).decode()]

    def test_ollama_timeout_120s(self, post):
        OllamaAPIInterface().generate("x")
        assert post.calls[0]["timeout"] == 120

    def test_ollama_stream_disabled(self, post):
        OllamaAPIInterface().generate("x")
        assert post.calls[0]["json"]["stream"] is False

    def test_local_loads_model_with_device(self, fake_transformers):
        LocalQwenVLInterface(model_name="Qwen/Qwen-VL-Chat", device="cpu")
        assert fake_transformers.model_args == ("Qwen/Qwen-VL-Chat", {"device_map": "cpu", "trust_remote_code": True})
        assert fake_transformers.tok_args[1] == {"trust_remote_code": True}

    def test_local_text_only_query(self, fake_transformers):
        out = LocalQwenVLInterface().generate("你好")
        assert out == "模型回复"
        assert fake_transformers.chat_queries[-1] == ("你好", None)
        assert fake_transformers.list_format == []

    def test_local_image_query_uses_list_format(self, fake_transformers):
        LocalQwenVLInterface().generate("描述", image_path="a.png")
        assert fake_transformers.list_format[-1] == [{"image": "a.png"}, {"text": "描述"}]

    def test_subclasses_share_base_interface(self, fake_transformers):
        assert issubclass(OllamaAPIInterface, BaseLLMInterface)
        assert issubclass(LocalQwenVLInterface, BaseLLMInterface)


@pytest.mark.boundary
class TestLLMBoundary:
    def test_empty_prompt(self, post):
        OllamaAPIInterface().generate("")
        assert post.calls[0]["json"]["prompt"] == ""

    def test_very_long_prompt(self, post):
        p = "屏幕元素" * 25000
        OllamaAPIInterface().generate(p)
        assert len(post.calls[0]["json"]["prompt"]) == 100000

    def test_missing_response_key_returns_empty(self, post):
        post.resp = FakeResp({"done": True})
        assert OllamaAPIInterface().generate("x") == ""

    def test_empty_image_file(self, post, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        OllamaAPIInterface().generate("x", image_path=str(p))
        assert post.calls[0]["json"]["images"] == [""]

    def test_image_path_empty_string_treated_as_no_image(self, post):
        OllamaAPIInterface().generate("x", image_path="")
        assert "images" not in post.calls[0]["json"]

    def test_base_url_trailing_slash(self, post):
        OllamaAPIInterface(base_url="http://h:1/").generate("x")
        assert post.calls[0]["url"] == "http://h:1/api/generate"

    def test_base_url_without_scheme_and_port(self):
        assert OllamaAPIInterface(base_url="172.20.192.1").base_url == "http://172.20.192.1:11434"

    def test_ollama_host_bind_all_maps_to_localhost(self, monkeypatch):
        """Windows 端为让 WSL2 访问会设 OLLAMA_HOST=0.0.0.0（服务端监听地址），客户端不能直接连 0.0.0.0"""
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
        assert OllamaAPIInterface().base_url == "http://localhost:11434"

    def test_base_url_env_priority(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://a:1")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://b:2")
        assert OllamaAPIInterface().base_url == "http://b:2"
        assert OllamaAPIInterface(base_url="http://c:3").base_url == "http://c:3"

    def test_response_with_think_tags_returned_verbatim(self, post):
        post.resp = FakeResp({"response": "<think>思考</think>{\"a\":1}"})
        assert OllamaAPIInterface().generate("x").startswith("<think>")

    def test_base_url_reads_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://172.20.192.1:11434")
        assert OllamaAPIInterface().base_url == "http://172.20.192.1:11434"


@pytest.mark.exception
class TestLLMException:
    def test_factory_invalid_mode(self):
        with pytest.raises(ValueError, match="不支持的模式"):
            build_llm_interface("openai")

    def test_factory_mode_case_sensitive(self):
        with pytest.raises(ValueError):
            build_llm_interface("API")

    def test_factory_unknown_kwarg(self):
        with pytest.raises(TypeError):
            build_llm_interface("api", top_k=5)  # 第4周起 temperature/think 是合法参数

    def test_base_interface_is_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMInterface()

    def test_http_500_raises(self, post):
        post.resp = FakeResp(status=500)
        with pytest.raises(requests.HTTPError):
            OllamaAPIInterface().generate("x")

    def test_http_404_model_not_found(self, post):
        post.resp = FakeResp({"error": "model not found"}, status=404)
        with pytest.raises(requests.HTTPError):
            OllamaAPIInterface().generate("x")

    def test_connection_refused(self, post):
        post.exc = requests.ConnectionError("Connection refused")
        with pytest.raises(requests.ConnectionError):
            OllamaAPIInterface().generate("x")

    def test_timeout(self, post):
        post.exc = requests.Timeout("read timed out")
        with pytest.raises(requests.Timeout):
            OllamaAPIInterface().generate("x")

    def test_invalid_json_body(self, post):
        post.resp = FakeResp(json_error=ValueError("not json"))
        with pytest.raises(ValueError):
            OllamaAPIInterface().generate("x")

    def test_missing_image_file(self, post, tmp_path):
        with pytest.raises(FileNotFoundError):
            OllamaAPIInterface().generate("x", image_path=str(tmp_path / "nope.png"))
        assert post.calls == []  # 图片读取失败时不应发出请求

    def test_local_without_transformers(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ImportError):
            LocalQwenVLInterface()

    def test_local_model_load_failure(self, fake_transformers, monkeypatch):
        def fail(*a, **k):
            raise OSError("CUDA not available")
        monkeypatch.setattr(sys.modules["transformers"].AutoModelForCausalLM, "from_pretrained", fail)
        with pytest.raises(OSError):
            LocalQwenVLInterface()
