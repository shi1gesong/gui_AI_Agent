"""
第5周：多模态大模型 LoRA 微调与能力提升

模块：
  action_format  VLM 动作格式（坐标 0~1000 归一化）：序列化为训练目标、解析模型输出、转换成 Agent 动作
  prompts        提示词模板（v0 极简 / v1 结构化 / v2 结构化+示例），用于提示词工程对比
  build_dataset  ScreenAgent → 训练集 / 验证集 / 测试集（按会话划分，避免同一任务的截图跨集合泄漏）
  collator       多模态 SFT 批处理：只对助手回答部分计算损失
  modeling       加载 Qwen2-VL / Qwen2.5-VL，挂 LoRA（可选 4bit QLoRA）
  train_lora     LoRA 训练入口（WSL2 + GPU）
  infer          VLMActor：截图 + 指令 → 动作（基座 / 基座+LoRA）
  evaluate       测试集评测：格式合格率、动作类型准确率、点击命中率、文本/按键准确率、整步成功率
  serve          把 VLMActor 包成 HTTP 服务，Windows 端 Agent 通过 --act vlm 调用

除 train_lora / infer / serve 需要 torch + transformers + peft 外，其余模块只依赖标准库，
可以在 Windows 端（gui-agent-win，没有装 torch）直接 import。
"""
