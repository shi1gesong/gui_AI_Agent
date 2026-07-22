# 第3周实验报告

项目：基于多模态大模型的桌面GUI智能体开发与优化
第3周：公开GUI数据集处理与基础Agent框架搭建 | 2026年7月22日

## 一、本周任务目标

按项目周计划，第3周核心任务为：（1）下载并预处理ScreenAgent、WebArena、Mind2Web等公开GUI任务数据集；（2）基于LangChain搭建基础多模态Agent框架；（3）实现简单的任务拆解与规划能力；（4）开发大模型调用接口，支持开源多模态模型的本地部署与API调用。全部代码在WSL2 gui-agent环境中开发，数据集处理需要联网，Agent框架的大模型推理复用了此前Microsoft实习项目中已部署的Ollama服务。

## 二、数据集处理结果

| 数据集 | 处理方式 | 结果 | 备注 |
|---|---|---|---|
| Mind2Web | HuggingFace streaming加载 + 精简动作字段 | 20条样本，含指令+精简后的动作序列(op/value/target_tag/target_attributes) | 原始actions字段自带完整网页raw_html，体积巨大，已重构代码只提取关键字段 |
| ScreenAgent | git clone官方仓库 | 仓库克隆完成，含数据集与训练代码 | 具体数据集格式需后续按需查阅仓库README扩展预处理逻辑 |
| WebArena | git clone + 提取config_files/examples下任务json | 4条demo任务(task_id/instruction) | 仓库自带的是demo样例，非完整数据集；完整环境需额外部署Docker容器 |

## 三、Agent框架搭建与验证

### 3.1 框架设计

参照第1周调研结论，借鉴ScreenAgent的"Planning-Acting-Reflecting"三阶段设计，用LangChain的自定义LLM封装(LangChainLLMWrapper)接入本项目的大模型调用接口，实现了完整闭环：用户指令→Planning拆解子任务→逐个子任务Acting生成具体动作→全部执行后Reflecting判断任务完成情况。

### 3.2 大模型调用接口验证

大模型调用接口支持本地Qwen-VL-Chat和Ollama API两种模式。本周复用Microsoft实习项目里已部署的Ollama服务进行API模式验证，先用轻量模型qwen2.5:1.5b测试通了完整调用链路，再切换到qwen3:8b跑通Agent框架的完整三阶段流程。

### 3.3 端到端测试用例

以"打开浏览器并搜索天气预报"为测试指令，配合一份手写的模拟屏幕元素数据（未接入真实截图，仅用于验证代码逻辑），完整跑通了三阶段流程：

```
[Planning] 拆解出2个子任务: ['打开默认浏览器', "在浏览器的地址栏输入'天气预报'并按下回车键"]
[Acting] 子任务1 -> {'action_type': 'click', 'target_text': '设置', 'input_text': ''}
[Acting] 子任务2 -> {'action_type': 'click', 'target_text': '搜索', 'input_text': ''}
[Reflecting] {'is_complete': True, 'reason': '已在浏览器中执行了搜索操作，完成任务目标'}
```

## 四、问题排查记录

本周开发过程中遇到5类问题，均已定位并解决，记录如下：

| 问题现象 | 原因 | 解决方案 |
|---|---|---|
| Mind2Web下载报Network unreachable / SSL证书不匹配 | 本机运行的VPN(quickfox)干扰了WSL2网络路由 | 关闭VPN后网络恢复正常，Mind2Web下载成功 |
| WebArena任务提取结果为0条 | 任务json文件实际位于config_files/examples/子目录，原代码只扫描了config_files根目录 | 改用Path.rglob递归查找，并排除test.raw.json，修复后正确提取到4条任务 |
| Mind2Web样本数据体积巨大，终端刷屏 | 原始actions字段包含完整网页raw_html/cleaned_html及全部候选元素坐标属性 | 重构download_mind2web，只提取op/value/target_tag/target_attributes等关键字段，去除冗余HTML |
| WSL2无法通过localhost访问Windows端Ollama服务 | WSL2与Windows主机网络默认隔离，localhost在两端指向不同网络命名空间 | 在Windows端设置环境变量OLLAMA_HOST=0.0.0.0并重启Ollama服务；WSL端通过`ip route`获取网关IP(172.20.192.1)作为访问地址 |
| Agent的Acting阶段action_type输出为并列形式("click/type/scroll/key")而非具体值 | 初版Prompt只给出选项列表，未强制模型二选一并缺少示例 | 重写ACTING_PROMPT，加入四选一的明确说明与两个正反示例，修复后能稳定输出单一具体动作类型 |

## 五、本周测试结果总览

| 测试项 | 状态 | 结果摘要 |
|---|---|---|
| Mind2Web数据集下载与预处理 | ✅ 通过 | 20条样本，指令+精简动作序列，数据结构清晰可用 |
| ScreenAgent仓库克隆 | ✅ 通过 | 官方仓库完整克隆，作为第4周Agent框架设计参考 |
| WebArena任务提取 | ✅ 通过 | 4条demo任务，含真实任务指令文本 |
| LLM接口(Ollama API模式)调用 | ✅ 通过 | 跨WSL2/Windows网络配置后，qwen2.5:1.5b成功返回中文回复 |
| Agent框架Planning阶段 | ✅ 通过 | "打开浏览器并搜索天气预报"正确拆解为2个子任务 |
| Agent框架Acting阶段(JSON格式) | ✅ 通过（Prompt修复后） | action_type稳定输出单一具体值(click/type/scroll/key之一) |
| Agent框架Acting阶段(语义准确性) | ⚠️ 部分通过，已知限制 | "输入文字"类子任务仍可能被误判为click，需接入真实UI元素数据后进一步验证 |
| Agent框架Reflecting阶段 | ✅ 通过 | 正确判断任务完成状态，给出合理的完成依据 |

## 六、本周交付物清单

- `data/dataset_loader.py` —— 数据集下载与预处理脚本（Mind2Web/ScreenAgent/WebArena）
- `agent/llm_interface.py` —— 大模型调用接口（本地部署/Ollama API双模式）
- `agent/base_agent.py` —— 基础Agent框架（Planning-Acting-Reflecting三阶段）
- `data_cache/mind2web_processed.jsonl`、`webarena_processed.jsonl` —— 预处理后的数据集文件
- 本实验报告

## 七、已知限制

- Acting阶段的语义判断仍有局限：模型有时无法根据子任务描述正确选择对应的动作类型（如"输入文字"类任务被误判为click），这与当前测试用的是手写模拟UI元素而非真实截图识别结果有关，真实场景下UI元素信息更丰富，判断准确率预计会提升，但仍需在第6周专门验证和优化
- Mind2Web/WebArena目前只是小规模样本(20条/4条)，尚未达到用于模型微调训练所需的数据规模，第5周微调前需要视情况扩大采样量
- ScreenAgent数据集仅完成仓库克隆，具体数据格式与标注规范尚未深入解析，需在后续周次用到时再展开

## 八、下周计划（第4周）

将第2周开发的感知模块（截图+OCR+UI元素定位）与控制模块（鼠标键盘操作）跟本周搭建的Agent框架进行整合，实现"用户指令→屏幕感知→任务规划→动作执行→结果反馈"的完整闭环，并测试5个基础桌面任务（打开浏览器、搜索指定内容、打开指定文件、发送消息、关闭应用）。这一步会用真实截图+真实OCR识别结果替换本周使用的模拟数据，预计能同时验证并改善本周发现的Acting阶段语义判断问题。
