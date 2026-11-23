# 第2-3周 测试用例补充报告

> 起因：第2、3周合计只有 16 个测试用例（第2周 8 个、第3周 8 个），且拖拽/滚动从未实测，Agent 语义判断“部分通过”。导师要求测试用例达到三位数（100+）。
> 本报告对应 `tests/` 目录下的 pytest 测试套件，所有数字来自 2026-11-23 的一次完整运行（原始结果见 [test_results_week2_3.csv](test_results_week2_3.csv)）。

## 一、结论

| 指标 | 数量 |
|---|---|
| 测试用例总数 | **422** |
| 本次实际执行 | 378（mock 321 + Xvfb 虚拟屏幕真实 GUI 57） |
| 通过 | **355** |
| 已知缺陷（xfail，用例写了期望行为但当前代码做不到） | **23** |
| 意外失败 | 0 |
| 未执行（需 Windows 真机 / 真实 Ollama / 真实 OCR 模型） | 44 |

- 第2周未测的 **拖拽、滚动** 已在虚拟屏幕上用真实 pyautogui 事件验证通过，并在 1280×720、1920×1080、2560×1440、3840×2160 四种分辨率下重复运行，结果一致（每次 34 通过 + 1 已知缺陷）。
- 第3周 demo 中 **“打开默认浏览器”被映射为点击“设置”** 的问题，这次按语义如实判定为失败；同一 demo 里的第2个子任务和 Reflect 结论也是错的（见第五节）。
- 23 个 xfail 归纳为 20 个代码问题，列在第六节，作为第4周的修复清单。

## 二、测试设计

### 2.1 三层运行环境

| 层 | 目录 | 说明 | 本次是否执行 |
|---|---|---|---|
| mock | `tests/unit/` | pyautogui、mss、EasyOCR、PaddleOCR、Ollama、transformers、HuggingFace datasets、git 全部用替身，任何机器都能跑，不动鼠标、不下载模型 | ✅ 321 个 |
| Xvfb 真实 GUI | `tests/gui/` | Linux 虚拟屏幕上用**真实 mss 截图**、**真实 pyautogui 鼠标键盘事件**；全屏“事件记录窗口”记录收到的点击/拖拽/滚轮/按键/文本，用例据此断言 | ✅ 57 个 |
| 真机 / 真实服务 | `tests/real/` | Windows 记事本端到端、真实 Ollama 语义准确率、真实 OCR 识别率 | ⏸ 44 个，需在本机运行 |

### 2.2 四个测试类别

每个用例必须且只能属于一个类别，`conftest.py` 在收集阶段自动检查。

| 类别 | 含义 | 举例 |
|---|---|---|
| functional 功能 | 正常输入下功能正确 | 中文经剪贴板输入、拖拽起止点、OCR 结果统一格式、Plan/Act/Reflect 调用顺序 |
| boundary 边界 | 极值、空值、临界条件 | 1×1 区域截图、四角像素点击、置信度恰好等于阈值、`max_steps=0`、`sample_limit=0` |
| exception 异常 | 错误输入、依赖失败 | Ollama 连接拒绝/超时/500、OCR 引擎未安装、截图区域越界、FAILSAFE 触发 |
| multi_resolution 多分辨率 | 不同屏幕尺寸/DPI/多显示器 | 800×600 至 3840×2160、竖屏 1080×1920、双屏/三屏布局、100%-200% DPI 缩放下的 bbox |

## 三、统计结果

### 3.1 类别 × 结果

| 类别 | 通过 | 已知缺陷(xfail) | 未执行 | 合计 |
|---|---|---|---|---|
| functional 功能 | 162 | 3 | 29 | 194 |
| boundary 边界 | 76 | 16 | 1 | 93 |
| exception 异常 | 56 | 4 | 1 | 61 |
| multi_resolution 多分辨率 | 61 | 0 | 13 | 74 |
| **合计** | **355** | **23** | **44** | **422** |

### 3.2 模块 × 类别

| 周 | 模块 | 测试文件 | 功能 | 边界 | 异常 | 多分辨率 | 合计 | 结果 |
|---|---|---|---|---|---|---|---|---|
| 2 | 截图 screenshot.py | unit/test_screenshot.py | 14 | 7 | 6 | 20 | 47 | 45 通过 / 2 xfail |
| 2 | 截图（真实 mss） | gui/test_xvfb_screenshot.py | 4 | 1 | 2 | 15 | 22 | 22 通过 |
| 2 | OCR ocr_detector.py | unit/test_ocr_detector.py | 16 | 16 | 9 | 8 | 49 | 48 通过 / 1 xfail |
| 2 | OCR（真实引擎） | real/test_real_ocr.py | 0 | 0 | 0 | 12 | 12 | 未执行 |
| 2 | 控制 mouse_keyboard.py | unit/test_mouse_keyboard.py | 34 | 16 | 7 | 18 | 75 | 70 通过 / 5 xfail |
| 2 | 控制（真实事件） | gui/test_xvfb_control.py | 21 | 10 | 4 | 0 | 35 | 34 通过 / 1 xfail |
| 2 | 控制（Windows 真机） | real/test_real_device.py | 7 | 1 | 1 | 1 | 10 | 未执行 |
| 3 | 数据集 dataset_loader.py | unit/test_dataset_loader.py | 14 | 16 | 9 | 0 | 39 | 37 通过 / 2 xfail |
| 3 | LLM 接口 llm_interface.py | unit/test_llm_interface.py | 15 | 8 | 12 | 0 | 35 | 34 通过 / 1 xfail |
| 3 | Agent 框架 base_agent.py | unit/test_base_agent.py | 20 | 18 | 11 | 0 | 49 | 41 通过 / 8 xfail |
| 3 | Acting 语义回归 | unit/test_semantic_regression.py | 27 | 0 | 0 | 0 | 27 | 24 通过 / 3 xfail |
| 3 | 真实 LLM 语义准确率 | real/test_real_llm.py | 22 | 0 | 0 | 0 | 22 | 未执行 |
| | **合计** | | **194** | **93** | **61** | **74** | **422** | |

第2周模块共 250 个用例，第3周模块共 172 个。

## 四、用例清单（按模块分组）

逐条用例及结果见 [test_results_week2_3.csv](test_results_week2_3.csv)（422 行，含类别、运行环境、结果）。下面是各模块覆盖的要点。

**截图（69 个）**：显示器列表；全屏/区域截图尺寸；BGRX→RGB 通道转换（mock 和真实纯红/绿/蓝屏幕各验证一次）；保存文件名与内容；区域截图与全屏裁剪逐像素一致；1×1 和右下角最后一个像素；同一秒两次保存；负数显示器索引；越界、零宽、负高区域；mss 初始化失败；8 种分辨率（含竖屏）全屏和中心区域；4 种多显示器布局；真实 Xvfb 下 6 种分辨率。

**OCR（61 个）**：EasyOCR/PaddleOCR 语言参数；两种引擎返回格式统一为 `{text,bbox,confidence}`；PaddleOCR 多页合并；numpy 类型转成 Python 类型；倾斜四边形转轴对齐 bbox；浮点坐标取整；阈值恰好相等、略低、为 0；框超出图片、零面积框；最新截图查找（目录不存在、为空、非 png、误选标注图）；引擎未安装、推理报错、bbox 格式错；benchmark 记录单引擎失败；4K 右下角 bbox；100%/125%/150%/200% DPI。真实引擎用例：合成中英文图片，在 3 种分辨率下检查两种引擎的识别率和 bbox 重合度。

**控制（120 个）**：左/右/中键、多次点击、双击、平滑移动；**拖拽**（先移到起点再拖、按下点=起点、抬起点=终点、中间有连续轨迹、反向拖拽、零距离）；**滚动**（上/下方向、格数、指定位置、0 格）；ASCII 输入、**中文/中英混合/emoji/日韩文经剪贴板输入**、换行、1 万字长文本；回车、组合键（Ctrl 修饰位真实验证）；清空后重输；FAILSAFE（左上角，以及四个角的像素本身）；越界坐标；未知按键名；剪贴板失败时不应粘贴；6 种分辨率下的屏幕尺寸、中心点击、整屏对角拖拽。真实事件用例在 4 种分辨率下各跑一遍。

**数据集（39 个）**：Mind2Web 流式读取参数、统一字段、动作精简（丢弃 raw_html）、中文不转义、`sample_limit` 为 0 或超过数据量、缺 annotation_id/operation/pos_candidates/confirmed_task、60 步长序列、下载失败返回空且不写文件、未安装 datasets；WebArena 递归查找、排除 test.raw.json、task_id 缺失或为 0、空目录、非 JSON、格式错误/列表/GBK 编码文件跳过；git clone 命令参数、已存在跳过、克隆失败返回 None；import 副作用；ScreenAgent 解析缺失。

**LLM 接口（35 个）**：工厂函数的两种模式和参数透传；Ollama 请求地址、payload、图片 base64、120 秒超时、关闭流式；缺 response 字段；空/超长 prompt；空图片文件；尾部斜杠；`<think>` 原样返回；本地 Qwen-VL 加载参数和图文 query 构造；非法模式、大小写、未知参数、抽象类实例化；HTTP 404/500、连接拒绝、超时、非 JSON 响应、图片不存在时不发请求；transformers 未安装、模型加载失败。

**Agent 框架（49 个）**：LangChain 包装器；Plan/Act/Reflect 的 Prompt 内容与解析；action_history 记录；run() 调用次数、每步刷新屏幕、结果结构；JSON 提取（夹带解释文字、markdown 代码块、多行、嵌套花括号、空串、只有右括号、截断 JSON）；`max_steps` 截断/为 0/刚好相等；空历史 Reflect；含花括号的 screen_context；qwen3 `<think>` 段干扰；多个 JSON；连续两次 run；非法 action_type；目标不在屏幕上；LLM 返回数组或字符串；LLM 中途报错保留部分历史；截图函数报错；包方式导入。

**语义回归（49 个）**：20 条覆盖浏览器、记事本、另存为、微信、登录表单的黄金用例（`tests/data/semantic_cases.json`），经 act() 管线后由语义评测器判定；第3周 demo 的真实输出回放；同样 20 条黄金用例在真实 Ollama 上的语义准确率（`-m llm`，本次未执行）。

## 五、第3周“语义判断部分通过”的如实结果

第3周报告中的端到端 demo 原始输出如下：

```
[Planning] ['打开默认浏览器', "在浏览器的地址栏输入'天气预报'并按下回车键"]
[Acting] 子任务1 -> {'action_type': 'click', 'target_text': '设置', 'input_text': ''}
[Acting] 子任务2 -> {'action_type': 'click', 'target_text': '搜索', 'input_text': ''}
[Reflecting] {'is_complete': True, 'reason': '已在浏览器中执行了搜索操作，完成任务目标'}
```

当时 Planning、Reflecting 记为通过，Acting 语义记为“部分通过”。这次把原始输出回放进 Agent，再用语义评测器逐项判定：

| 检查项 | 第3周记录 | 本次判定 | 原因 |
|---|---|---|---|
| 子任务1「打开默认浏览器」 | 部分通过 | ❌ 失败 | 点击了“设置”，与打开浏览器无关。屏幕上没有浏览器图标，正确做法是按 Win 键搜索或报告目标不存在 |
| 子任务2「在地址栏输入天气预报并回车」 | （未单列） | ❌ 失败 | 输入类任务被判成 click“搜索”，`input_text` 为空，“天气预报”根本没有输入 |
| Reflect「任务已完成」 | 通过 | ❌ 失败 | 两个动作都错，而且 Agent 当时没有接入控制模块，动作从未真正执行，却判定 `is_complete=True` |

对应用例：`test_semantic_regression.py::TestSemanticWeek3Regression` 中的 3 个 strict xfail。修复后它们会变成 XPASS 并报错，提醒把标记去掉。

根因有三点。第一，Acting 只校验 JSON 格式，不校验语义，也不校验目标是否真的在屏幕上（无 grounding）。第二，测试用的屏幕元素是手写的两条，里面根本没有浏览器。第三，Reflect 只看动作文本，不看执行结果和执行后的截图。

## 六、已知问题清单（23 个 xfail → 20 个问题）

| # | 模块 | 问题 | 对应用例 | 建议 |
|---|---|---|---|---|
| 1 | Agent | “打开默认浏览器”被映射为点击“设置” | test_week3_subtask1_semantically_correct | Act 后增加语义校验，目标必须来自屏幕元素 |
| 2 | Agent | 输入类子任务被判为 click，没有输入内容 | test_week3_subtask2_semantically_correct | Prompt 中按子任务动词给出动作类型约束，并做输出校验 |
| 3 | Agent | 动作错误且未执行时 Reflect 仍判定完成 | test_week3_reflection_not_falsely_complete | Reflect 时传入执行结果和执行后的截图/OCR |
| 4 | Agent | 不校验 target_text 是否出现在屏幕元素中 | test_act_rejects_target_not_on_screen | 实现 text→bbox→坐标的 grounding |
| 5 | Agent | 不校验 action_type 合法性 | test_act_rejects_invalid_action_type | 白名单校验，不合法时重试 |
| 6 | Agent | qwen3 `<think>` 段里有 `{` 时 JSON 解析失败 | test_parse_qwen3_think_block_with_braces | 先去掉 `<think>…</think>` 再解析 |
| 7 | Agent | LLM 输出多个 JSON 时解析失败 | test_parse_two_json_objects_takes_first | 用 `json.JSONDecoder.raw_decode` 逐个尝试 |
| 8 | Agent | LLM 返回 JSON 数组时 plan() 崩溃（AttributeError） | test_plan_json_array_response | 兼容 list，或判断类型后返回 [] |
| 9 | Agent | subtasks 为字符串时被逐字拆成子任务 | test_plan_subtasks_string_rejected | 类型校验 |
| 10 | Agent | 连续两次 run() 时 action_history 不清空 | test_run_twice_history_isolated | run() 开头重置历史 |
| 11 | Agent | `from llm_interface import` 导致无法以包方式导入 | test_import_as_package | 加 `agent/__init__.py`，改为相对导入 |
| 12 | LLM | Ollama 地址写死 localhost，不读 OLLAMA_HOST，WSL2 下不通 | test_base_url_reads_env | 从环境变量读取 |
| 13 | 控制 | `type_text` 输入中文时被静默丢弃（真实事件验证：只剩 ASCII） | test_type_text_auto_routes_non_ascii_to_clipboard、test_type_text_non_ascii_raises_or_warns、test_type_text_chinese_with_typewrite | 检测到非 ASCII 时自动走剪贴板 |
| 14 | 控制 | 中文输入后没有恢复用户原来的剪贴板 | test_type_unicode_restores_clipboard | 粘贴前保存，粘贴后恢复 |
| 15 | 控制 | 越界坐标、负坐标不校验 | test_click_outside_screen_rejected、test_click_negative_coordinate_rejected | 按屏幕尺寸校验并报错 |
| 16 | 截图 | 文件名只精确到秒，同一秒内两次截图会互相覆盖 | test_two_saves_in_same_second_keep_both_files | 文件名加毫秒或序号 |
| 17 | 截图 | `monitor_index=-1` 被负索引静默接受 | test_negative_monitor_index_rejected | 校验索引范围 |
| 18 | OCR | 自动查找最新截图时会选中标注图（第2周已记录） | test_find_latest_skips_annotated_images | 只匹配 `screenshot_*.png` |
| 19 | 数据集 | import 时就在当前目录创建 data_cache/ | test_import_has_no_side_effect | 改为在函数内部创建 |
| 20 | 数据集 | ScreenAgent 只克隆了仓库，没有解析数据 | test_screenagent_preprocess_exists | 补充解析函数 |

### 6.1 测试中发现、已按现状记录的行为（不算缺陷，但第4周需要注意）

- **屏幕四角像素点不能点击**：pyautogui 的 FAILSAFE 在点击四个角的像素时就会触发异常。Windows 开始按钮在左下角，Agent 定位时要避开角点（4 个用例，已通过）。
- **未知按键名被静默忽略**：`press_key("not_a_real_key")` 既不报错也不输入，Agent 侧需要自己校验按键名。
- **越界区域截图**：X11 下 mss 抛 `ScreenShotError`，鼠标越界会被限制在屏幕边缘。Windows 下的行为需要真机确认。
- **mss 10.x 已弃用 `mss.mss()`**，提示改用 `mss.MSS`。`requirements_win.txt` 没有锁定版本，建议锁定。
- **DPI 缩放**：Windows 缩放比例不是 100% 时，mss 截图是物理像素，pyautogui 是逻辑像素，OCR 坐标需要换算后才能点击。真机用例 `test_device_screenshot_vs_pyautogui_scale` 会输出实际比例。

## 七、未执行的 44 个用例

| 标记 | 数量 | 运行条件 | 说明 |
|---|---|---|---|
| `real_device` | 10 | Windows 桌面，`GUI_AGENT_REAL_DEVICE=1` | 记事本中真实输入中英文、拖拽选中、滚动、组合键、DPI 比例。**云端无法执行，代码未经真机验证**，首次运行可能需要调整窗口坐标 |
| `llm` | 22 | `OLLAMA_BASE_URL`、`OLLAMA_MODEL` | 20 条黄金用例的真实语义准确率 + 调用与规划冒烟测试 |
| `ocr_real` | 12 | 安装 EasyOCR/PaddleOCR，`GUI_AGENT_REAL_OCR=1` | 合成图片在 3 种分辨率下的识别率与 bbox 重合度 |

建议在本机执行 `pytest -m "real_device or llm or ocr_real" -v`，把结果补进本报告。

## 八、局限性

- Xvfb 是 Linux 虚拟屏幕，可以验证 pyautogui 事件和 mss 截图的逻辑正确，但**不能替代 Windows 真机**（DPI 缩放、Win32 窗口、输入法等差异），所以另写了 `real_device` 用例。
- mock 用例验证的是本项目代码对依赖的调用方式和数据处理，不验证 OCR 模型和 LLM 本身的效果。效果由 `ocr_real`、`llm` 用例覆盖。
- 本次只新增测试，没有修改任何业务代码。第六节的问题留给第4周修复，修好后对应 xfail 会自动变成 XPASS 并提醒。

## 九、运行环境

- 执行环境：Linux 云端容器，Python 3.12.3，pytest 9.2，mss 10.2.0，PyAutoGUI 0.9.54，langchain-core 1.8.4，Pillow 13.0
- 虚拟屏幕：Xvfb 24 位色深；默认 1920×1080，控制类真实用例另在 1280×720、2560×1440、3840×2160 下重复运行
- 运行命令：`pytest`（全量），`XVFB_RES=2560x1440 pytest tests/gui/test_xvfb_control.py`（换分辨率）
- 全量耗时约 61 秒
