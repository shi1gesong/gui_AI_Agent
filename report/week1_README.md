# 第1周实验报告

项目：基于多模态大模型的桌面GUI智能体开发与优化
第1周：行业技术调研与开发环境搭建 | 2026年7月20日

## 一、本周任务目标

按照项目周计划，第1周核心任务为：（1）调研GUI智能体三大主流技术路线（UI-TARS、Claude Computer Use、ScreenAgent），产出技术调研报告；（2）搭建完整的开发环境，包含WSL2端的大模型/Agent框架环境，以及Windows原生端的桌面感知与控制环境。

## 二、技术调研工作

完成了UI-TARS、Claude Computer Use、ScreenAgent三条技术路线在核心架构、感知方式、动作建模、推理机制、训练方式、公开程度、Benchmark表现等维度的系统对比（详见《GUI智能体技术调研报告》）。

核心结论：
- ScreenAgent的"规划-执行-反思"三阶段流程将作为第3-4周基础Agent框架的设计蓝本
- UI-TARS的统一动作空间与System-2推理思路用于指导动作接口设计
- Claude Computer Use的tool_use范式作为动作调用格式的参考

## 三、开发环境搭建过程

### 3.1 环境架构

采用双环境隔离策略：
- **WSL2端（gui-agent）**：负责大模型推理、Agent框架、模型微调
- **Windows原生端（gui-agent-win）**：负责屏幕截图、OCR识别、鼠标键盘控制

二者通过文件系统交换数据。这一分工是因为WSL2本身无法直接访问Windows图形界面，截图与控制类操作必须在Windows原生Python环境中执行。

### 3.2 关键问题与解决过程

| 问题现象 | 根本原因 | 解决方案 |
|---|---|---|
| import torch 时报 Bus error | C盘可用空间仅9.4G，WSL2虚拟磁盘存储在C盘，空间不足导致torch加载大型库文件时内存映射(mmap)失败 | 将整个WSL发行版导出→注销→重新导入到D盘（`wsl --export` / `--import`），迁移后C盘可用空间恢复至35G，问题消失 |
| 首次wsl --export提示路径找不到 | 目标备份文件夹`D:\wsl-backup`事先未创建 | 先手动mkdir建好目标文件夹，再执行export |
| torch.cuda提示sm_120不兼容 | RTX 5070 Ti为较新的Blackwell架构，最初安装的torch 2.2.2+cu121版本过旧，不支持该GPU的CUDA算力等级 | 卸载重装为 torch 2.7.0 + cu128（与既有medical-rag项目版本保持一致） |
| import peft 时报 undefined symbol: torch_library_impl | pip安装langchain等包过程中torchaudio与torch版本出现ABI不匹配 | 项目不涉及音频处理，直接`pip uninstall torchaudio -y`移除 |

**经验教训**：在处理WSL磁盘迁移问题时，因为先执行了`wsl --unregister`才发现对应的`--export`备份并未成功创建，导致此前WSL环境内的全部内容（包括早前搭建的medical-rag相关配置）被清空且无法恢复。所幸该项目核心代码此前已托管，环境配置类的内容通过后续重新执行标准安装步骤即可完整复现，未造成不可逆的实质性损失，但也确认了操作顺序应当是"确认备份文件已生成"之后再执行清除类操作。

### 3.3 环境验证结果

| 验证项 | 状态 | 备注 |
|---|---|---|
| WSL2 gui-agent conda环境 | ✅ 通过 | python 3.10，因WSL磁盘迁移中途重建过一次 |
| PyTorch识别RTX 5070 Ti GPU | ✅ 通过 | 最终版本 torch 2.7.0+cu128，cuda.is_available()=True |
| Windows端 pyautogui 截图 | ✅ 通过 | 分辨率2560×1440，截图与桌面一致 |
| Windows端 EasyOCR文字识别 | ✅ 通过 | 对测试截图识别出320处文字区域，含坐标框与置信度 |
| LangChain / transformers / peft import | ✅ 通过 | 卸载torchaudio后解决ABI冲突 |
| Git仓库初始化 | ✅ 通过 | 项目骨架目录：perception/control/agent/models/data/eval/reports/demo |
| PaddleOCR验证 | ⏳ 待办 | 本周优先用EasyOCR跑通验证，留到第2周开发时再装 |

## 四、最终环境配置

```
# WSL2端 (gui-agent)
Python 3.10 | torch 2.7.0+cu128 | CUDA可用 | GPU: RTX 5070 Ti
langchain, transformers, peft, accelerate, bitsandbytes, datasets 均正常导入

# Windows原生端 (gui-agent-win)
Python 3.10 | pyautogui, mss, pynput, opencv-python, easyocr, paddleocr, pillow
截图验证：分辨率 2560×1440，与桌面一致
OCR验证：EasyOCR识别测试截图，命中320处文字区域
```

## 五、本周交付物清单

- 《GUI智能体技术调研报告》
- 《环境配置文档》
- 本实验报告
- Git仓库骨架：`gui-desktop-agent/`（perception、control、agent、models、data、eval、reports、demo 八个模块目录）

## 六、下周计划（第2周）

开发屏幕感知与控制核心模块：
1. 在Windows端实现跨平台屏幕截图，支持多分辨率适配
2. 正式集成PaddleOCR（本周已验证EasyOCR可用，第2周对比两者精度/速度后确定主用OCR方案）
3. 开发鼠标键盘控制模块，支持点击、输入、滚动、拖拽等基本操作
4. 实现UI元素坐标定位与边界框绘制功能

预计交付物：桌面感知与控制模块完整代码 + 单元测试报告。
