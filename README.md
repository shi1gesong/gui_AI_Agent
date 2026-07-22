# gui_AI_Agent

基于多模态大模型的桌面GUI智能体开发与优化项目。参考UI-TARS、Claude Computer Use、ScreenAgent等技术路线，目标是实现能够"看懂屏幕、操作电脑"的桌面GUI智能体原型。

## 项目结构

```
gui_AI_Agent/
├── week1/                          # 第1周：技术调研 + 环境搭建
│   ├── README.md                   # 第1周实验报告
│   ├── GUI智能体技术调研报告.docx
│   └── 环境配置文档.md
├── week2/                          # 第2周：桌面感知与控制模块
│   ├── README.md                   # 第2周实验报告
│   ├── USAGE.md                    # 模块使用说明
│   ├── perception/                 # 截图 + OCR识别
│   └── control/                    # 鼠标键盘控制
└── week3/                          # 第3周：数据处理 + Agent框架
    ├── README.md                   # 第3周实验报告
    ├── USAGE.md                    # 模块使用说明
    ├── data/                       # 公开数据集预处理
    └── agent/                      # LangChain Agent框架
```

## 各周进度

| 周次 | 核心内容 | 详情 |
|---|---|---|
| 第1周 | 技术调研 + WSL2/Windows双环境搭建 | [report/week1_README.md](report/week1_README.md) |
| 第2周 | 屏幕截图、OCR识别、鼠标键盘控制模块 | [report/week2_README.md](report/week2_README.md) |
| 第3周 | 公开数据集处理、Planning-Acting-Reflecting Agent框架 | [report/week3_README.md](report/week3_README.md) |

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
