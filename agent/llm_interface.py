"""
大模型调用接口
支持两种模式：
1. local  —— 本地部署开源多模态模型（Qwen-VL-Chat / GLM-4V-9B等），走transformers推理
2. api    —— 通过API调用（如接入你已有的Ollama/Qwen3服务，或其他兼容OpenAI格式的接口）

设计目标：上层Agent框架不关心具体用哪种模型，统一调用 generate() 方法
"""
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseLLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """输入文本prompt和可选的图片路径，返回模型生成的文本"""
        raise NotImplementedError


class LocalQwenVLInterface(BaseLLMInterface):
    """
    本地部署 Qwen-VL-Chat 模型（需要GPU，显存建议16GB+）
    模型首次调用会自动从HuggingFace下载权重，体积较大（约10-20GB），请确保磁盘空间充足
    """
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"正在加载模型 {model_name}（首次运行需下载权重，可能耗时较久）...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, trust_remote_code=True
        ).eval()
        print("模型加载完成")

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            query = self.tokenizer.from_list_format([
                {"image": image_path},
                {"text": prompt},
            ])
        else:
            query = prompt

        response, _ = self.model.chat(self.tokenizer, query=query, history=None)
        return response


class OllamaAPIInterface(BaseLLMInterface):
    """
    通过 Ollama 本地API调用模型（复用你在Microsoft实习medical-rag项目里已经跑通的Ollama服务）
    默认假设Ollama服务跑在本机 http://localhost:11434
    """
    def __init__(self, model_name: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import requests

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload["images"] = [img_b64]

        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")


def build_llm_interface(mode: str = "api", **kwargs) -> BaseLLMInterface:
    """工厂函数，根据mode返回对应的LLM接口实例"""
    if mode == "local":
        return LocalQwenVLInterface(**kwargs)
    elif mode == "api":
        return OllamaAPIInterface(**kwargs)
    else:
        raise ValueError(f"不支持的模式: {mode}，可选 'local' 或 'api'")


if __name__ == "__main__":
    # 单元测试：仅测试接口能否正常构造与调用（默认用api模式，避免强制下载大模型权重）
    # 需要本机已跑起 Ollama 服务并拉取过对应模型，例如: ollama run qwen3:8b
    try:
        llm = build_llm_interface(mode="api", model_name="qwen3:8b")
        response = llm.generate("用一句话介绍一下你自己")
        print(f"✅ API模式调用测试通过，返回: {response[:100]}")
    except Exception as e:
        print(f"⚠️ API模式测试未通过（可能是Ollama服务未启动或模型未拉取）: {e}")
        print("提示：可执行 `ollama serve` 启动服务，`ollama pull qwen3:8b` 拉取模型后重试")
