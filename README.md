# Long-Horizon GRPO Project

基于端到端 Agentic RL 构建出行规划场景下的 Long-Horizon Agent。

基于 Qwen3-4B-Instruct-2507 构建多轮出行规划 Agent，使用工具查询地点、路线、天气、网页和交通信息，并通过 SFT 冷启动与 GRPO 后训练研究长程工具使用。

## 项目流程

```text
种子问题 (306 train / 100 test)
  -> DeepSeek 扩增训练问题 (1835 train prompts)
  -> MiniMax-M2.7 多轮工具轨迹蒸馏
  -> 规则、格式和 LLM 质量过滤
  -> 850 SFT train / 31 SFT val / 200 RL / 80 final test
  -> Qwen3-4B LoRA SFT
  -> vLLM 多轮 rollout + ms-swift GRPO
  -> 独立测试集上的工具轨迹与答案评估
```

训练数据保存在 `data/`，包括中间清洗结果和最终划分。`data/final/sft_train.jsonl`、`sft_val.jsonl`、`rl.jsonl`、`test_final.jsonl` 分别是训练和测试脚本使用的主要文件。数据为 JSONL，每条记录使用 `id` 与 `conversations` 字段；`conversations` 包含 `system`、`user` 和 `assistant` 消息，工具调用与工具返回分别以 `<tool_call>`、`<tool_response>` 标记。检索网页中疑似凭据的 URL 查询参数已替换为 `REDACTED`。

## 主要组件

| 组件 | 文件 | 作用 |
| --- | --- | --- |
| 问题扩增 | `data_expansion.ipynb` | 扩增种子问题 |
| 轨迹蒸馏 | `data_distill.py` | 使用教师模型生成多轮轨迹 |
| 数据清洗 | `data_clean/` | 格式与质量过滤 |
| SFT | `train_sft.sh` | ms-swift LoRA 微调 |
| 在线 GRPO | `rollout.sh`、`train_rl.sh` | vLLM rollout、奖励计算和策略更新 |
| 工具循环 | `run_tool_loop_infer.py`、`tools/` | 解析工具调用、执行工具、注入返回 |
| 课程式奖励 | `ms-swift/examples/train/grpo/plugin/tooluse_reward_parser_aligned.py` | 五项规则奖励与 LLM Judge |
| 多轮调度 | `ms-swift/examples/train/grpo/plugin/tooluse_multi_turn_scheduler.py` | GRPO 中的工具执行与终止控制 |

可用工具共八个：`search`、`visit`、`poi_search`、`around_search`、`route_planning`、`weather_search`、`train_tickets_search`、`flights_search`。网页搜索与访问使用 Firecrawl，地点、路线与天气使用高德 API；车票与航班工具通过配置的语言模型产生**模拟**返回，不是实时票务接口。

## 环境准备

建议在 Linux、CUDA GPU 与 Python 环境中运行。原始实验使用 4 张 GPU 进行 SFT，并使用 6 张训练 GPU 与 2 张 vLLM GPU 进行 GRPO。所需资源取决于模型大小和批量设置。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ./ms-swift
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写所需 API 凭据；不要提交这个文件。下载 Qwen3-4B-Instruct-2507 到 `pretrained/Qwen/Qwen3-4B-Instruct-2507/`，或通过 `MODEL_PATH` 指定其他可用的模型路径。第三方模型与服务需遵守各自许可和使用条款。

## 运行

以下命令从仓库根目录执行。训练产物会写入已被 Git 忽略的 `saved/`。

```bash
bash train_sft.sh

# 从 SFT 输出中选取需要合并的 LoRA checkpoint。
SFT_ADAPTER_DIR=/path/to/checkpoint bash merge_lora.sh

# 分别在两个终端启动 rollout 服务和 GRPO 训练。
bash rollout.sh
bash train_rl.sh

# MODEL_DIR 可以是 SFT 合并权重或 GRPO checkpoint。
MODEL_DIR=/path/to/model-or-checkpoint bash infer.sh
```

`train_sft.sh`、`train_rl.sh` 和 `rollout.sh` 支持环境变量覆盖模型、数据和输出路径。`train_rl.sh` 的默认模型路径与 `merge_lora.sh` 的默认输出路径一致。GRPO 训练仍需正确配置外部工具服务和 LLM Judge。

## 结果解释与已知限制

- 本仓库保留了历史实验使用的数据和训练流程。当前 `rl.jsonl` 存有完整教师轨迹；ms-swift 的预处理只移除最后一条教师 `assistant` 答案。因此原始 GRPO 运行是**在教师工具历史之后续写**，不应描述为仅从用户问题独立生成完整工具轨迹。
- 历史 rollout 日志曾出现工具路径加载失败。此公开副本已将 GRPO 插件中的本机绝对路径改为仓库相对路径，但**尚未重新训练和验证这些修改后的结果**。
- `test_output/` 保留本地模型输出与评估记录；模型权重、训练日志和含密钥的 `.env` 均未发布。Notebook 的执行输出已清空，代码单元仍在。

## 许可证

项目代码按 [Apache License 2.0](LICENSE) 发布。内置的 `ms-swift/` 保留其原有 [许可证](ms-swift/LICENSE)；其他第三方内容及数据说明见 [NOTICE.md](NOTICE.md)。
