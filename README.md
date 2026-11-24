# gui_AI_Agent

基于多模态大模型的桌面GUI智能体开发与优化项目。参考UI-TARS、Claude Computer Use、ScreenAgent等技术路线，目标是实现能够"看懂屏幕、操作电脑"的桌面GUI智能体原型。

## 项目结构

```
gui_AI_Agent/
├── perception/          # 感知：截图(screenshot.py)、OCR(ocr_detector.py)、感知管线+DPI换算(screen_parser.py)
├── control/             # 控制：鼠标键盘(mouse_keyboard.py)、程序启动/关闭(app_manager.py)
├── agent/               # Agent：LLM接口、Plan-Act-Reflect框架、动作校验、grounding、执行器、
│                        #        混合规划(planner.py)、端到端闭环(gui_agent.py)、命令行(cli.py)、基础任务(basic_tasks.py)
├── apps/mock_chat.py    # 模拟聊天程序（“发送消息”任务的测试目标）
├── data/                # 公开数据集预处理
├── finetune/            # 第5周：Qwen2-VL LoRA 微调（数据集、训练、推理、评测、HTTP 服务）
├── scripts/             # run_week5_wsl.sh：WSL2 一键 数据 → 基线 → 训练 → 评测 → 对比
├── tests/               # 1278 个测试用例（mock / Xvfb 虚拟屏幕 / Windows 真机 / 真实 LLM / 真实 OCR / 微型模型 / GPU）
└── report/              # 各周实验报告与测试报告
```

## 快速开始（第4周 v1.0）

```bat
conda activate gui-agent-win
cd %USERPROFILE%\Desktop\gui_AI_Agent
python -m agent.cli --list-tasks                       :: 查看 5 个基础任务
python -m agent.cli --plan-only "在浏览器中搜索天气预报"   :: 只看规划，不动鼠标
python -m agent.cli "在浏览器中搜索天气预报"              :: 真正执行（运行期间不要动鼠标）
python -m agent.cli                                    :: 交互模式
python -m agent.basic_tasks --repeat 3                 :: 5 个基础任务批量测试
```

不连 Ollama 时加 `--no-llm`（只能执行模板能识别的指令）。

## 各周进度

| 周次 | 核心内容 | 详情 |
|---|---|---|
| 第1周 | 技术调研 + WSL2/Windows双环境搭建 | [report/week1_README.md](report/week1_README.md) |
| 第2周 | 屏幕截图、OCR识别、鼠标键盘控制模块 | [report/week2_README.md](report/week2_README.md) |
| 第3周 | 公开数据集处理、Planning-Acting-Reflecting Agent框架 | [report/week3_README.md](report/week3_README.md) |
| 第2-3周补充 | 523 个测试用例 + 20 个缺陷修复 | [report/week2-3_测试报告.md](report/week2-3_测试报告.md) |
| 第4周 | 端到端 GUI 智能体 v1.0：闭环、CLI、5 个基础任务、469 个新用例 | [report/week4_README.md](report/week4_README.md)、[测试报告](report/week4_基础任务测试报告.md) |
| 第5周 | Qwen2-VL-2B LoRA 微调、提示词对比、`--act vlm` 接入闭环、285 个新用例 | [report/week5_README.md](report/week5_README.md)、[对比报告](report/week5_微调效果对比报告.md) |

## 技术栈

- **感知**：mss（截图）、EasyOCR / PaddleOCR（文字识别）
- **控制**：pyautogui、pynput（鼠标键盘操作）
- **Agent框架**：LangChain
- **大模型调用**：Ollama API / 本地Qwen-VL-Chat
- **环境**：WSL2（Ubuntu，GPU推理）+ Windows原生（桌面感知控制）

## 运行环境

项目采用双环境架构：
- WSL2端（`gui-agent`）：负责大模型推理、Agent框架
- Windows原生端（`gui-agent-win`）：负责屏幕截图、OCR、鼠标键盘控制

详见各周README中的环境配置说明。
