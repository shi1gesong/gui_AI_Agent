# 第5周 微调效果对比报告（Qwen2-VL-2B + LoRA）

> 对比对象：同一测试集（ScreenAgent 官方测试划分 282 条）上的 常数基线 / 基座零样本 × 3 种提示词 / LoRA 微调后。
> 测试用例逐条结果见 [test_results_week5.csv](test_results_week5.csv)。
> **状态（2026-11-24）**：管线与测试已在云端完成；模型的数字要在本机 WSL2 + RTX 5070 Ti 上运行 `scripts/run_week5_wsl.sh` 后填入第二节（标 ⏳ 的格子）。

## 一、结论

| 指标 | 数值 |
|---|---|
| 第5周新增测试用例 | **285 个** |
| 云端已执行 | 275 个，**全部通过，0 失败**（另 10 个需要 GPU） |
| 全套用例 | 1278 个：云端 1194 通过 / 0 失败 / 84 跳过（真机、Ollama、GPU） |
| 由测试发现并修复的缺陷 | 3 个（见周报第五节） |
| 微调数据集 | train 808 / val 82 / test 282，三者会话不重叠 |
| 微调前后对比 | ⏳ 待本机运行 |

## 二、微调前后对比（ScreenAgent 测试集 282 条）

| 配置 | 格式合格率 | 动作类型准确率 | 首动作正确率 | 整步成功率 | 点击命中@50px | 坐标距离中位数 | 输入文字准确率 | 按键准确率 | 单条耗时 |
|---|---|---|---|---|---|---|---|---|---|
| 常数基线（总是点击训练集点击位置的中位数） | 100.0% | 63.1% | 3.2% | 1.8% | 4.3% | 310.3 px | 0.0% | 0.0% | — |
| 基座 + v0_minimal | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 基座 + v1_schema | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 基座 + v2_fewshot | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| **LoRA + v1_schema** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

常数基线的意义：测试集 63% 的首动作是点击，所以“总是点击”的类型准确率就有 63%；但它坐标几乎全错（中位距离 310px），首动作正确率只有 3.2%。**判断模型有没有学到东西，要看首动作正确率和点击命中率，而不是类型准确率。**

## 三、训练过程

| 项目 | 数值 |
|---|---|
| 基座 | Qwen/Qwen2-VL-2B-Instruct |
| LoRA | r=16，alpha=32，dropout=0.05，语言模型 q/k/v/o/gate/up/down，视觉编码器冻结 |
| 可训练参数 | ⏳（`train_summary.json` 的 params） |
| 优化 | AdamW，lr 2e-4，预热 5% + 余弦退火，等效批大小 8，3 epoch，梯度检查点 |
| 每 epoch 训练 / 验证损失 | ⏳ |
| 最佳 epoch | ⏳ |
| 训练用时 / 显存峰值 | ⏳ |

## 四、测试设计

### 4.1 四层运行环境

| 层 | 内容 | 用例数 | 本次执行 |
|---|---|---|---|
| mock | 动作格式转换与解析、提示词、数据集划分、评测指标、对比报表、VLM 客户端、执行器坐标动作、`--act vlm` 闭环（模拟桌面）、HTTP 服务 | 242 | ✅ 云端 |
| torch_model | 随机初始化的微型 Qwen2-VL（真实模型类、真实图像处理器、字节级分词器）：批处理与 label 掩码、LoRA 挂载位置、训练循环、保存/加载、合并权重、生成、训练/评测 CLI、HTTP 服务 | 33 | ✅ 云端 CPU |
| gpu | 真实 Qwen2-VL-2B（+ LoRA）：GPU 可用、3 种提示词、微调后格式合格率 ≥ 90%、微调后不差于基座、推理耗时、放大到真机分辨率 | 10 | ⏳ 本机 WSL2 |

微型模型的意义：云端没有 GPU，也下载不了真实权重，但训练循环、label 掩码、LoRA 只挂在语言模型、适配器保存后重新加载 logits 一致、合并权重后一致、`from_pretrained(本地目录)`、generate 解码这些**代码路径与真实 2B 模型完全相同**，只有权重是随机的。所以它验证的是“管线正确”，模型效果由 gpu 层和第二节的数字说明。

### 4.2 类别分布（第5周新增 285 个）

| 类别 | 通过 | 待 GPU | 合计 |
|---|---|---|---|
| functional 功能 | 140 | 7 | 147 |
| boundary 边界 | 39 | 1 | 40 |
| exception 异常 | 71 | 0 | 71 |
| multi_resolution 多分辨率 | 25 | 2 | 27 |
| **合计** | **275** | **10** | **285** |

### 4.3 模块分布

| 测试文件 | 用例 | 覆盖内容 |
|---|---|---|
| `test_ft_action_format.py` | 93 | ScreenAgent 动作 → VLM 动作（拖拽合并、右键、按键别名、坐标裁剪）；模型输出解析（代码块、思考标签、别名、bbox、Qwen 坐标标记、像素/小数坐标、坏 JSON）；0~1000 → 鼠标坐标（5 种分辨率、偏移） |
| `test_ft_dataset_eval.py` | 62 | 3 种提示词；数据集构建（会话不泄漏、可复现、去重、缺图、Windows 反斜杠路径、坏行定位）；评测指标、断点续跑、对比表与柱状图 |
| `test_ft_tiny_model.py` | 58 | LoRA 目标正则（两种 transformers 命名、排除视觉 MLP）、学习率曲线、配置校验；微型模型上的批处理、训练、保存加载、生成、CLI、HTTP |
| `test_ft_vlm_agent.py` | 62 | `--act vlm` 闭环：T4 发送消息、DPI 4 档、分辨率 3 档、服务异常/坐标错误被检查器发现；VLMClient；执行器 point/move/drag/右键；HTTP 服务 7 种异常；CLI 参数 |
| `test_real_vlm.py` | 10 | 真实 GPU 用例（见 4.1） |

### 4.4 关键用例

- **训练前缀与推理一致**：训练样本中提示词部分的 token 与推理时 `apply_chat_template(add_generation_prompt=True)` 的 token 逐个相同，且全部 label = -100。不一致的话模型学到的东西在推理时用不上。
- **适配器重新加载一致**：训练后保存的 LoRA 适配器，加载到新的基座上，logits 与训练结束时相同（误差 1e-5），合并权重后仍相同（误差 1e-4）。
- **端到端可学习**：微型模型训练后，对训练截图逐字生成 `[{"action": "click", "x": 127, "y": 156}]`，解析与评测给出整步成功率 100%、点击命中@25px 100%。
- **视觉编码器不被修改**：LoRA 只出现在 2 层 × 7 种语言模型线性层上（14 个），名字里没有 visual。
- **`--act vlm` 闭环**：用“完全正确的 VLM 替身”跑 T4，消息成功发给张三；故意让替身给出错误坐标时，每步执行都“成功”，但检查器发现消息没发出去，任务判为失败。

## 五、本机运行步骤

需要两个终端：**Anaconda Prompt**（更新代码）和 **Ubuntu / WSL**（训练，有 GPU）。全程约 1 小时，其中下载模型约 4.5GB。

### 第 1 步：在 Anaconda Prompt 里切到第5周分支

```bat
cd %USERPROFILE%\Desktop\gui_AI_Agent
git fetch origin
git checkout claude/project-thread-k1vrmc
git pull
```

### 第 2 步：在 WSL 里进入仓库、装依赖

打开“Ubuntu”（WSL 终端），逐行粘贴：

```bash
conda activate gui-agent
cd "$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")/Desktop/gui_AI_Agent"
pip install -U "transformers>=4.49,<4.58" peft accelerate pillow matplotlib pytest
```

> 如果报 `Failed to find C compiler`：新版 peft 会顺带导入 bitsandbytes，而它需要 gcc。默认训练用不到 bitsandbytes，脚本检测到这个错误会自动卸载它；也可以手动 `pip uninstall -y bitsandbytes`。只有加 `--load-4bit`（QLoRA）时才需要先 `sudo apt install -y build-essential`。

### 第 3 步：先跑 5 分钟冒烟，确认环境没问题

```bash
QUICK=1 bash scripts/run_week5_wsl.sh
```

最后看到“全部完成”就说明环境正常。

### 第 4 步：完整运行（约 1 小时，期间可以不管它）

```bash
bash scripts/run_week5_wsl.sh
```

### 第 5 步：把结果发回

需要的三个文件（都在仓库目录下）：

- `outputs/eval/compare.md`（对比表）
- `outputs/lora_qwen2vl_2b/train_summary.json`（训练过程）
- `outputs/week5_logs/pytest_gpu.txt`（GPU 用例结果）

### 可选：真机体验 `--act vlm`

WSL 里启动服务（保持窗口开着）：

```bash
python -m finetune.serve --adapter outputs/lora_qwen2vl_2b/best
```

Anaconda Prompt 里：

```bat
conda activate gui-agent-win
cd %USERPROFILE%\Desktop\gui_AI_Agent
python -m agent.cli --act vlm --task T4
```
