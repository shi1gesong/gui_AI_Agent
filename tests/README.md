# 测试套件说明

覆盖第2周（感知 + 控制）、第3周（数据集、LLM 接口、Agent 框架）、第4周（端到端闭环 v1.0）、第5周（LoRA 微调）与第6周（系统 v2.0 鲁棒性）全部模块。
每个用例恰好属于一个类别：`functional` 功能 / `boundary` 边界 / `exception` 异常 / `multi_resolution` 多分辨率，
收集阶段会自动检查；运行结束时终端会打印“类别 × 结果”“模块 × 结果”“运行环境 × 结果”三张统计表。

## 目录

```
tests/
  conftest.py              全局配置：依赖桩、Xvfb 虚拟屏幕、类别检查、统计输出
  fakes.py                 pyautogui / mss / 剪贴板 / LLM 替身
  sim_desktop.py           第4周：内存中的模拟桌面（窗口/界面/截图/OCR/鼠标键盘），跑 5 个基础任务
  semantic_eval.py         Acting 阶段语义评测器
  data/semantic_cases.json 20 条 Acting 语义黄金用例
  data/week4_semantic_cases.json 12 条第4周扩展动作空间语义用例
  fault_desktop.py         第6周：可注入 14 种故障的模拟桌面（慢加载/弹窗/丢点击/OCR 抖动/抢焦点/截图失败/LLM 宕机…）
  robustness_eval.py       第6周：v1 vs v2 故障注入鲁棒性评测（python tests/robustness_eval.py --seeds 5）
  unit/                    mock 用例（任何机器都能跑，不动鼠标、不需要模型）
  gui/                     Xvfb 虚拟屏幕上的真实截图与真实鼠标键盘事件（Linux 自动启用）
  real/                    需要真机/真实服务的用例（默认跳过）
```

## 运行方式

```bash
pip install -r requirements-test.txt

# 1) 日常：全部 mock 用例 + （Linux 有 Xvfb 时）虚拟屏幕真实 GUI 用例
pytest

# 2) 只跑某个类别 / 某个模块
pytest -m boundary
pytest tests/unit/test_mouse_keyboard.py

# 3) 在不同分辨率的虚拟屏幕上重复真实控制用例（Linux）
XVFB_RES=2560x1440 pytest tests/gui/test_xvfb_control.py

# 4) Windows 真机（会真实操作鼠标键盘，运行前关闭记事本，运行时不要碰鼠标）
set GUI_AGENT_REAL_DEVICE=1
pytest -m real_device -v

# 5) 真实 LLM（Ollama 需已拉取模型；WSL2 中填 Windows 网关地址）
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=qwen3:8b
pytest -m llm -v

# 6) 真实 OCR 引擎
set GUI_AGENT_REAL_OCR=1
pytest -m ocr_real -v

# 7) 第4周：Windows 真机 5 个基础任务 / 真实 qwen3:8b
set GUI_AGENT_REAL_DEVICE=1
pytest tests\real\test_real_week4.py -m real_device -v -s
set OLLAMA_BASE_URL=http://localhost:11434
pytest tests\real\test_real_week4.py -m llm -v

# 8) 第4周：Xvfb 虚拟屏幕 + 真实 EasyOCR 端到端（Linux，需 pip install easyocr 与中文字体）
GUI_AGENT_REAL_OCR=1 pytest tests/gui/test_xvfb_e2e.py

# 9) 第5周：微型 Qwen2-VL 真实训练/推理（需要 torch+transformers+peft，CPU 即可；没装时自动跳过）
pytest -m torch_model

# 10) 第5周：真实 Qwen2-VL + LoRA（WSL2 + GPU）
GUI_AGENT_VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct GUI_AGENT_VLM_ADAPTER=outputs/lora_qwen2vl_2b/best \
  pytest tests/real/test_real_vlm.py -m gpu -v -s

# 11) 第6周：Xvfb 真实 Tk 窗口上的慢加载/弹窗/控件检测/效果校验（弹窗用例需要 pip install python-xlib）
pytest tests/gui/test_xvfb_week6.py
GUI_AGENT_REAL_OCR=1 pytest tests/gui/test_xvfb_week6.py -k easyocr -s      # 真实 EasyOCR：v1 全屏 vs v2 前台窗口

# 12) 第6周：Windows 真机 6 个复杂/干扰任务（v2）+ 感知速度
set GUI_AGENT_REAL_DEVICE=1
pytest tests/real/test_real_week6.py -m real_device -v -s

# 导出每个用例的结果为 CSV
GUI_AGENT_RESULTS_CSV=results.csv pytest
```

Linux 跑 Xvfb 用例需要系统包：`xvfb xclip python3-tk`。
在 Windows 上 `gui/` 下的用例会自动跳过，mock 用例会把 pyautogui 替换掉，不会移动真实鼠标。

## xfail 的含义

`xfail(strict=True)` 表示“期望行为已写成用例，但当前代码做不到”，即**已知缺陷**。
它们不会让 CI 变红，但会出现在报告的已知问题清单里；代码修好后用例会变成 XPASS 并因 strict 报错，
提醒把 xfail 标记去掉。
