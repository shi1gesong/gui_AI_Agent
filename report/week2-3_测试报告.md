# 第2-3周 测试用例补充与缺陷修复报告

> 起因：第2、3周合计只有 16 个测试用例（第2周 8 个、第3周 8 个），且拖拽/滚动从未实测，Agent 语义判断“部分通过”。导师要求测试用例达到三位数（100+）。
> 本报告分两步：先补充测试用例并如实记录发现的 23 个缺陷（第一轮），再逐一修复这些缺陷并重跑测试（第二轮）。
> 所有数字来自 2026-11-23 的实际运行，逐条结果见 [test_results_week2_3.csv](test_results_week2_3.csv)。

## 一、结论

| 指标 | 第一轮（补用例） | 第二轮（修复后） |
|---|---|---|
| 测试用例总数 | 422 | **509** |
| 实际执行 | 378 | 463 |
| 通过 | 355 | **463** |
| 已知缺陷（strict xfail） | 23 | **0** |
| 意外失败 | 0 | 0 |
| 未执行（需 Windows 真机 / 真实 Ollama / 真实 OCR） | 44 | 46 |

- 第一轮发现的 20 个代码问题（23 个缺陷用例）**全部修复**，对应用例的 xfail 标记已去掉，改为普通断言并通过；修复时另补了 87 个用例覆盖新逻辑。
- 第2周未测的 **拖拽、滚动** 已在虚拟屏幕上用真实 pyautogui 事件验证，并在 1280×720、1920×1080、2560×1440、3840×2160 四种分辨率下重复运行，结果一致。
- 第3周 demo 中“打开默认浏览器 → 点击设置”等 3 处错误判断，已在第3周报告中更正为失败；修复后同一回放全部通过（见第五节）。

## 二、测试设计

### 2.1 三层运行环境

| 层 | 目录 | 说明 | 本次是否执行 |
|---|---|---|---|
| mock | `tests/unit/` | pyautogui、mss、EasyOCR、PaddleOCR、Ollama、transformers、HuggingFace datasets、git 全部用替身，任何机器都能跑，不动鼠标、不下载模型 | ✅ 405 个 |
| Xvfb 真实 GUI | `tests/gui/` | Linux 虚拟屏幕上用**真实 mss 截图**、**真实 pyautogui 鼠标键盘事件**；全屏“事件记录窗口”记录收到的点击/拖拽/滚轮/按键/文本，用例据此断言 | ✅ 58 个 |
| 真机 / 真实服务 | `tests/real/` | Windows 记事本端到端、真实 Ollama 语义准确率、真实 OCR 识别率 | ⏸ 46 个，需在本机运行 |

### 2.2 四个测试类别

每个用例必须且只能属于一个类别，`conftest.py` 在收集阶段自动检查。

| 类别 | 含义 | 举例 |
|---|---|---|
| functional 功能 | 正常输入下功能正确 | 中文经剪贴板输入、拖拽起止点、OCR 结果统一格式、Plan/Act/Reflect 调用顺序 |
| boundary 边界 | 极值、空值、临界条件 | 1×1 区域截图、四角像素点击、置信度恰好等于阈值、`max_steps=0`、`sample_limit=0` |
| exception 异常 | 错误输入、依赖失败 | Ollama 连接拒绝/超时/500、OCR 引擎未安装、截图区域越界、FAILSAFE 触发 |
| multi_resolution 多分辨率 | 不同屏幕尺寸/DPI/多显示器 | 800×600 至 3840×2160、竖屏 1080×1920、双屏/三屏布局、100%-200% DPI 缩放下的 bbox |

## 三、统计结果（修复后）

### 3.1 类别 × 结果

| 类别 | 通过 | 失败 | 未执行 | 合计 |
|---|---|---|---|---|
| functional 功能 | 195 | 0 | 30 | 225 |
| boundary 边界 | 121 | 0 | 2 | 123 |
| exception 异常 | 86 | 0 | 1 | 87 |
| multi_resolution 多分辨率 | 61 | 0 | 13 | 74 |
| **合计** | **463** | **0** | **46** | **509** |

### 3.2 模块 × 类别

| 周 | 模块 | 测试文件 | 功能 | 边界 | 异常 | 多分辨率 | 合计 | 结果 |
|---|---|---|---|---|---|---|---|---|
| 2 | 截图 screenshot.py | unit/test_screenshot.py | 14 | 7 | 12 | 20 | 53 | 53 通过 |
| 2 | 截图（真实 mss） | gui/test_xvfb_screenshot.py | 4 | 1 | 2 | 15 | 22 | 22 通过 |
| 2 | OCR ocr_detector.py | unit/test_ocr_detector.py | 16 | 17 | 9 | 8 | 50 | 50 通过 |
| 2 | OCR（真实引擎） | real/test_real_ocr.py | 0 | 0 | 0 | 12 | 12 | 未执行 |
| 2 | 控制 mouse_keyboard.py | unit/test_mouse_keyboard.py | 36 | 23 | 10 | 18 | 87 | 87 通过 |
| 2 | 控制（真实事件） | gui/test_xvfb_control.py | 21 | 10 | 5 | 0 | 36 | 36 通过 |
| 2 | 控制（Windows 真机） | real/test_real_device.py | 8 | 2 | 1 | 1 | 12 | 未执行 |
| 3 | 数据集 dataset_loader.py | unit/test_dataset_loader.py | 18 | 22 | 12 | 0 | 52 | 52 通过 |
| 3 | LLM 接口 llm_interface.py | unit/test_llm_interface.py | 15 | 11 | 12 | 0 | 38 | 38 通过 |
| 3 | Agent 框架 base_agent.py | unit/test_base_agent.py | 24 | 22 | 14 | 0 | 60 | 60 通过 |
| 3 | 动作校验与执行器（新增） | unit/test_action_validator.py | 19 | 8 | 10 | 0 | 37 | 37 通过 |
| 3 | Acting 语义回归 | unit/test_semantic_regression.py | 28 | 0 | 0 | 0 | 28 | 28 通过 |
| 3 | 真实 LLM 语义准确率 | real/test_real_llm.py | 22 | 0 | 0 | 0 | 22 | 未执行 |
| | **合计** | | **225** | **123** | **87** | **74** | **509** | |

第2周模块共 272 个用例，第3周模块共 237 个。

## 四、用例清单（按模块分组）

逐条用例及结果见 [test_results_week2_3.csv](test_results_week2_3.csv)（509 行，含类别、运行环境、结果）。各模块覆盖要点：

**截图（75 个）**：显示器列表；全屏/区域截图尺寸；BGRX→RGB 通道转换（mock 和真实纯红/绿/蓝屏幕各验证一次）；保存文件名与内容；区域截图与全屏裁剪逐像素一致；1×1 和右下角最后一个像素；同一秒内多次保存不覆盖；负数/非整数/越界显示器索引；越界、零宽、负高区域预先报错；第二块屏上的区域；mss 初始化失败；8 种分辨率（含竖屏）；4 种多显示器布局；真实 Xvfb 下 6 种分辨率。

**OCR（62 个）**：EasyOCR/PaddleOCR 语言参数；两种引擎返回格式统一为 `{text,bbox,confidence}`；PaddleOCR 多页合并；numpy 类型转换；倾斜四边形转轴对齐 bbox；浮点坐标取整；阈值恰好相等、略低、为 0；框超出图片、零面积框；最新截图查找（跳过标注图）；引擎未安装、推理报错、bbox 格式错；benchmark 记录单引擎失败；4K 右下角 bbox；100%-200% DPI。真实引擎用例：合成中英文图片，在 3 种分辨率下检查两种引擎的识别率和 bbox 重合度。

**控制（135 个）**：左/右/中键、多次点击、双击、平滑移动；**拖拽**（按下点=起点、抬起点=终点、中间有连续轨迹、反向、零距离、终点越界时不移动）；**滚动**（方向、格数、指定位置、0 格、位置越界）；ASCII 输入、**中文/中英混合/emoji/日韩文输入（`type_text` 自动走剪贴板）**、剪贴板恢复、换行、1 万字长文本；回车、组合键；未知按键名报错；四角像素自动内移、边缘非角点不内移；用户甩鼠标到角落时 FAILSAFE 仍生效；越界/负坐标报错；6 种分辨率。真实事件用例在 4 种分辨率下各跑一遍。

**数据集（52 个）**：Mind2Web 字段统一、动作精简、各种字段缺失、下载失败；WebArena 递归查找、排除 test.raw.json、格式错误/GBK 文件跳过；git clone 参数与失败；import 无副作用；**ScreenAgent 解析**（统一字段、点击/输入/按键/滚动/拖拽/等待转换、坐标比例、负样本过滤、纯规划样本跳过、缺图片名、屏幕尺寸为 0、坏 JSON）。

**LLM 接口（38 个）**：两种模式与参数透传；Ollama 请求地址、payload、图片 base64、超时；**地址优先级（参数 > OLLAMA_BASE_URL > OLLAMA_HOST > localhost）**、0.0.0.0 转 localhost、补协议和端口、去尾部斜杠；本地 Qwen-VL 加载与图文 query；HTTP 404/500、连接拒绝、超时、非 JSON、图片不存在时不发请求等。

**Agent 框架 + 动作校验（97 个）**：Plan/Act/Reflect 的 Prompt 与解析；JSON 提取（`<think>` 段、多个 JSON、数组、代码块、截断）；subtasks 类型校验；**动作校验**（类型、输入内容、grounding、目标相关性、按键名）；**带原因重试**、重试次数为 0；**规则兜底**；**执行器**（bbox 中心点击、先点输入框再输入、按键别名、follow_up、找不到目标、控制模块报错）；**Reflect 规则把关**（未执行、未通过校验、输出无法解析、空历史时强制未完成）；连续两次 run 历史隔离；包方式导入。

**语义回归（50 个）**：20 条黄金用例经 act() 管线判定；第3周 demo 原始输出回放；真实 Ollama 语义准确率（`-m llm`，本次未执行）。

## 五、第3周“语义判断部分通过”的更正与修复

第3周报告中的端到端 demo 原始输出：

```
[Planning] ['打开默认浏览器', "在浏览器的地址栏输入'天气预报'并按下回车键"]
[Acting] 子任务1 -> {'action_type': 'click', 'target_text': '设置', 'input_text': ''}
[Acting] 子任务2 -> {'action_type': 'click', 'target_text': '搜索', 'input_text': ''}
[Reflecting] {'is_complete': True, 'reason': '已在浏览器中执行了搜索操作，完成任务目标'}
```

| 检查项 | 第3周原记录 | 如实判定 | 修复后（同一输出回放） |
|---|---|---|---|
| 子任务1「打开默认浏览器」 | 部分通过 | ❌ 点击“设置”与打开浏览器无关 | ✅ 校验拦下“设置”；屏幕上没有浏览器图标，兜底为 按 Win → 输入“浏览器” → 回车 |
| 子任务2「在地址栏输入天气预报并回车」 | （未单列） | ❌ 输入类任务被判成 click，没有输入内容 | ✅ 校验拦下 click；兜底为 输入“天气预报” → 回车 |
| Reflect「任务已完成」 | 通过 | ❌ 动作全错且从未执行，却判定完成 | ✅ 未接执行器时动作未执行，强制判定未完成；接上执行器、动作真正执行后才采纳 LLM 的“完成” |

回放假设模型每次重试都重复同样的错误（最坏情况），证明即使模型不改正，Agent 也不会再把错误动作交给控制模块。另有一个端到端用例把修复后的 Agent、执行器和真实控制模块连起来（pyautogui 用替身），验证实际发出的是 Win → 粘贴“浏览器” → 回车 → 粘贴“天气预报” → 回车。

第3周报告第五节已加更正说明，两项状态改为“失败（2026-11-23 更正）”。

## 六、缺陷修复清单（20 个问题全部修复）

| # | 模块 | 问题 | 修复方式 | 验证用例 |
|---|---|---|---|---|
| 1 | Agent | “打开默认浏览器”被映射为点击“设置” | 新增 `agent/action_validator.py`：目标必须与“打开X”相关；不合格时带原因重试，仍不合格时规则兜底（屏幕上有该应用则点它，没有则 Win 键搜索）；Prompt 示例不再用“设置” | test_week3_subtask1_semantically_correct 等 |
| 2 | Agent | 输入类子任务被判为 click，没有输入内容 | 子任务含“输入X”时必须是 type 且 input_text 包含 X；兜底生成 type（含回车 follow_up） | test_week3_subtask2_semantically_correct |
| 3 | Agent | 动作错误且未执行时 Reflect 仍判定完成 | `run()` 可传入执行器（`agent/executor.py`），记录 executed/error；Reflect 在未执行、未通过校验、输出无法解析、无动作时强制 `is_complete=False`，保留 `llm_is_complete` 供排查 | test_week3_reflection_not_falsely_complete、test_reflect_keeps_llm_true_when_all_executed |
| 4 | Agent | 不校验 target_text 是否在屏幕上 | 解析 screen_context 中的元素，click 目标必须能匹配到元素（grounding） | test_act_rejects_target_not_on_screen |
| 5 | Agent | 不校验 action_type | 白名单校验，不合格重试，最终标记 `_invalid` 且不会被执行 | test_act_rejects_invalid_action_type、test_invalid_action_not_sent_to_executor |
| 6 | Agent | qwen3 `<think>` 段干扰 JSON 解析 | 解析前去掉 `<think>…</think>`（含只有结束标签的情况） | test_parse_qwen3_think_block_with_braces |
| 7 | Agent | 输出多个 JSON 时解析失败 | 用 `json.JSONDecoder.raw_decode` 取第一个合法 JSON | test_parse_two_json_objects_takes_first |
| 8 | Agent | LLM 返回 JSON 数组时 plan() 崩溃 | 兼容数组 | test_plan_json_array_response |
| 9 | Agent | subtasks 为字符串时被逐字拆开 | 类型校验，字符串包成单元素列表，过滤空值和非字符串 | test_plan_subtasks_string_rejected、test_plan_filters_empty_and_non_string |
| 10 | Agent | 连续 run() 时历史不清空 | run() 开头重置 | test_run_twice_history_isolated |
| 11 | Agent | 无法以包方式导入 | 新增 `agent/__init__.py`、`data/__init__.py`，导入兼容包方式与脚本方式 | test_import_as_package |
| 12 | LLM | Ollama 地址写死 localhost，WSL2 下不通 | 读取 OLLAMA_BASE_URL / OLLAMA_HOST，0.0.0.0 转 localhost，自动补协议与端口 | test_base_url_reads_env 等 5 个 |
| 13 | 控制 | `type_text` 输入中文被静默丢弃 | 含非 ASCII 字符时自动走剪贴板 | test_type_text_auto_routes_non_ascii_to_clipboard、test_type_text_chinese_auto_clipboard（真实事件） |
| 14 | 控制 | 中文输入后不恢复用户剪贴板 | 粘贴前保存、粘贴后等待 0.15 秒再恢复 | test_type_unicode_restores_clipboard、test_clipboard_restored_after_chinese_input（真实事件） |
| 15 | 控制 | 越界、负坐标不校验 | 所有坐标先校验，越界报 ValueError 且不执行任何动作 | test_click_outside_screen_rejected 等 6 个 |
| 16 | 控制 | 点击四角像素会触发 FAILSAFE（第一轮测试中发现） | 目标恰好是角点时向内偏移 1 像素；用户自己把鼠标甩到角落时 FAILSAFE 仍然有效 | test_click_exact_corner_pixel_nudged（真实事件，4 个角）、test_failsafe_top_left_corner |
| 17 | 控制 | 未知按键名被 pyautogui 静默忽略（第一轮测试中发现） | press_key / hotkey 按 pyautogui.KEYBOARD_KEYS 校验 | test_unknown_key_rejected（真实事件） |
| 18 | 截图 | 同一秒两次截图互相覆盖 | 文件名带毫秒 + 自增序号 | test_two_saves_in_same_second_keep_both_files |
| 19 | 截图 | 负数显示器索引被静默接受；越界区域在不同系统表现不一 | 校验显示器索引；区域尺寸和范围预先校验，统一报 ValueError；兼容 mss 10.x 的 `mss.MSS` | test_negative_monitor_index_rejected、test_region_outside_screen_rejected |
| 20 | OCR / 数据集 | 自动找截图会选中标注图；import 时创建 data_cache；ScreenAgent 未解析 | 跳过 annotated_*；目录改为写文件时才创建；新增 `preprocess_screenagent()` | test_find_latest_skips_annotated_images、test_import_has_no_side_effect、ScreenAgent 13 个用例 |

`preprocess_screenagent()` 已在真实的 ScreenAgent 官方仓库上运行：3005 个标注文件中解析出 **891 条**含可执行动作的样本（点击 520、按键 254、输入 216、双击 91、移动 74、等待 65、滚动 41、拖拽 25 等），截图路径均存在。

本次修复只改了业务代码和对应测试，没有为了通过而放宽任何断言；原本断言“当前行为”的几个用例（如 `type_text` 输入中文、越界坐标、未知按键）已改为断言修复后的正确行为。

## 七、未执行的 46 个用例

| 标记 | 数量 | 运行条件 | 说明 |
|---|---|---|---|
| `real_device` | 12 | Windows 桌面，`GUI_AGENT_REAL_DEVICE=1` | 记事本中真实输入中英文、拖拽选中、滚动、组合键、DPI 比例、开始按钮角点。**云端无法执行，代码未经真机验证**，首次运行可能需要调整窗口坐标 |
| `llm` | 22 | `OLLAMA_BASE_URL`、`OLLAMA_MODEL` | 20 条黄金用例的真实语义准确率 + 调用与规划冒烟测试（这里测的是模型本身的第一次输出，不含重试和兜底） |
| `ocr_real` | 12 | 安装 EasyOCR/PaddleOCR，`GUI_AGENT_REAL_OCR=1` | 合成图片在 3 种分辨率下的识别率与 bbox 重合度 |

建议在本机执行 `pytest -m "real_device or llm or ocr_real" -v`，把结果补进本报告。

## 八、局限性

- Xvfb 是 Linux 虚拟屏幕，可以验证 pyautogui 事件和 mss 截图的逻辑正确，但**不能替代 Windows 真机**（DPI 缩放、Win32 窗口、输入法等差异）。
- 动作校验与兜底基于规则（关键词、引号、“打开X/输入X/按X键”句式），能拦下第3周那类明显错误，但不保证覆盖所有表述；模型本身的语义准确率仍要用 `-m llm` 用例在真实模型上测量。
- 执行器的 grounding 直接用 OCR bbox 中心点，没有做 DPI 缩放换算。Windows 缩放比例不是 100% 时坐标会偏，留给第4周处理（真机用例 `test_device_screenshot_vs_pyautogui_scale` 会输出实际比例）。

## 九、运行环境

- 执行环境：Linux 云端容器，Python 3.12.3，pytest 9.2，mss 10.2.0，PyAutoGUI 0.9.54，langchain-core 1.8.4，Pillow 13.0
- 虚拟屏幕：Xvfb 24 位色深；默认 1920×1080，控制类真实用例另在 1280×720、2560×1440、3840×2160 下重复运行
- 运行命令：`pytest`（全量），`XVFB_RES=2560x1440 pytest tests/gui/test_xvfb_control.py`（换分辨率）
- 全量耗时约 63 秒
