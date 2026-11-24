"""
把感知、控制、Agent 各模块组装成一个可运行的 GUIAgent（CLI 与基础任务测试共用）

组装顺序有讲究：先创建 ScreenCapture（mss 在 Windows 上会把进程设为 DPI 感知），
再创建 MouseKeyboardController，这样两者用的是同一套像素坐标；ScreenPerceiver 每次感知时
还会重新计算 截图尺寸 / pyautogui.size() 的比例，缩放 ≠ 100% 时自动换算坐标。
"""
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AgentConfig:
    planner: str = "auto"           # auto / rule / llm
    act_mode: str = "hybrid"        # hybrid / llm / rule
    reflect_mode: str = "auto"      # auto / llm / rule
    use_llm: bool = True
    model: str = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    base_url: Optional[str] = None
    think: Optional[bool] = False
    temperature: Optional[float] = 0.0
    ocr_engine: str = "easyocr"
    min_confidence: float = 0.3
    dry_run: bool = False
    step_retries: int = 1
    max_steps: int = 20
    chat_app: str = "模拟聊天"
    chat_log: Optional[str] = None
    run_dir: Optional[str] = None
    open_timeout: float = 10.0
    vlm_url: Optional[str] = None   # 第5周 --act vlm：finetune/serve.py 的地址
    vlm_ocr: bool = False           # vlm 模式下是否仍跑 OCR（默认不跑，省 4~6 秒/步）
    extra: dict = field(default_factory=dict)


class NullOCR:
    """vlm 模式不需要 OCR 元素：返回空列表，只保留截图"""

    def detect(self, image_path: str):
        return []


class LazyOCR:
    """第一次真正需要识别时才加载 OCR 模型（EasyOCR 加载要十几秒；只用规则动作的任务完全不需要）"""

    def __init__(self, engine: str = "easyocr"):
        self.engine = engine
        self._detector = None

    def detect(self, image_path: str):
        if self._detector is None:
            from perception.ocr_detector import OCRDetector
            print(f"[感知] 正在加载 OCR 引擎 {self.engine}（首次约 10~30 秒）...")
            self._detector = OCRDetector(engine=self.engine)
        return self._detector.detect(image_path)


def default_run_dir(base: str = "runs") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / base / stamp


def build_agent(cfg: AgentConfig, log=print):
    """返回 (agent, components)；components 里有 app_manager / controller 等，基础任务测试需要用到"""
    from perception.screenshot import ScreenCapture
    from perception.screen_parser import ScreenPerceiver
    from control.mouse_keyboard import MouseKeyboardController
    from control.app_manager import AppManager, PyGetWindowProvider
    from agent.executor import ControllerExecutor, DryRunExecutor
    from agent.gui_agent import GUIAgent
    from agent.planner import HybridPlanner, RulePlanner

    run_dir = Path(cfg.run_dir) if cfg.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    capture = ScreenCapture(save_dir=str(run_dir))           # 必须先于 controller 创建（DPI 感知）
    controller = MouseKeyboardController()
    windows = PyGetWindowProvider() if sys.platform.startswith("win") else None
    apps = AppManager(windows=windows, controller=controller, mock_chat_log=cfg.chat_log,
                      open_timeout=cfg.open_timeout)
    use_ocr = cfg.act_mode != "vlm" or cfg.vlm_ocr
    perceiver = ScreenPerceiver(capture=capture, ocr=LazyOCR(cfg.ocr_engine) if use_ocr else NullOCR(), min_confidence=cfg.min_confidence,
                                region_fn=apps.screen_regions if windows is not None else None)
    executor = DryRunExecutor() if cfg.dry_run else ControllerExecutor(controller, app_manager=apps, settle_sec=0.3)

    brain = None
    if cfg.use_llm:
        from agent.base_agent import BaseGUIAgent
        kw = {"model_name": cfg.model, "think": cfg.think, "temperature": cfg.temperature}
        if cfg.base_url:
            kw["base_url"] = cfg.base_url
        brain = BaseGUIAgent(llm_mode="api", **kw)

    vlm = None
    if cfg.act_mode == "vlm":
        from agent.vlm_client import DEFAULT_VLM_URL, VLMClient
        vlm = VLMClient(cfg.vlm_url or DEFAULT_VLM_URL)

    planner_mode = cfg.planner if brain is not None else "rule"
    act_mode = cfg.act_mode if (brain is not None or cfg.act_mode == "vlm") else "rule"
    planner = HybridPlanner(mode=planner_mode,
                            rule=RulePlanner(chat_app=cfg.chat_app, app_resolver=apps.resolve_app),
                            llm_plan=brain.plan if brain else None)
    agent = GUIAgent(perceiver=perceiver, executor=executor, planner=planner, brain=brain,
                     act_mode=act_mode, reflect_mode=cfg.reflect_mode if brain else "rule",
                     step_retries=cfg.step_retries, max_steps=cfg.max_steps, trace_dir=str(run_dir), log=log,
                     vlm=vlm)
    return agent, {"vlm": vlm, "app_manager": apps, "controller": controller, "capture": capture, "perceiver": perceiver,
                   "run_dir": run_dir, "brain": brain}
