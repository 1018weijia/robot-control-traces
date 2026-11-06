# chat：按角色分类导出原始聊天记录

状态：设计已确认，取代 `2026-11-05-curate-vqa-design.md`（VQA 整理方案不再做）。

## 目标

把 Codex 机器人控制会话的原始聊天记录原样导出，只做两件事：给每条内容标上类别，按决策轮次分组。
内容一律逐字来自记录（导出时已脱敏），不改写、不摘要、不截断、不划分训练集和测试集。

核心要求：内容之间的对应关系必须清楚，哪段分析对应哪次观测、哪条反馈属于哪个动作，都能从导出数据直接看出来。

## 命令与产物

```
rct.py chat --rollout R1 R2 --out DIR      # 或 --auto
DIR/manifest.json                          导出了哪些会话、跳过了哪些（原因）、脱敏计数、检查结果
DIR/sessions/<日期>_<会话前8位>/
    session.json                           任务（第一条用户消息）、模型、各类计数、检查结果
    rounds.jsonl                           每行一个决策轮次
    images/                                agent 当时看到的图，文件名是内容哈希
```

默认只导出有机器人控制证据的会话（与 `export` 相同的控制判定，`--min-control-steps 0` 关闭）；默认脱敏；
不导出系统提示和环境上下文；现有命令不变。

## 类别

| 类别 | 来源 |
|---|---|
| task | 用户消息（第一条是任务，后面的是中途插话） |
| observation | 观测类工具调用及其返回（看图、读状态、相机），用户发来的图 |
| analysis | 推理摘要、说明文字；控制命令里 `--note`/`--evidence`、MCP 参数 `reason` 等理由，单独拆成一条 |
| action | 控制类调用：命令或参数原文，加上还原出的代码 |
| feedback | 控制调用的返回：输出、退出码、返回的图；异步动作后续取回的结果 |
| other | 辅助命令（查文件、装包等）、上下文压缩、中断 |

类别直接沿用现有 profile 的角色判定（act / observe / result / aux）。

## 排序与轮次

- **排序按 rollout 文件里的原始行序。** 工具调用和它的返回在文件里是两条记录，各自有位置；并行调用、
  异步返回时，返回出现在哪些推理之后，按真实先后排。`seq` 是会话内的统一序号。
- **一轮** = 从上一轮最后一条反馈之后，到本轮所有动作的最后一条反馈为止（含）。第一轮从会话开头开始。
  异步动作等待期间的推理、轮询都留在本轮。连续发出、返回之前的多个控制调用算同一轮。
  最后一个动作之后的内容是 `final` 轮。
- 动作和它的反馈用同一个 `call_id` 对应；轮询或 `write_stdin` 取回的反馈另带 `action` 字段指向原动作。
- 从命令里拆出的理由（analysis）与命令同一个 `seq`、同一个 `call_id`，排在命令前面。命令本身保持原文。
- 上下文压缩所在轮标 `context_reset: true`。

示例（一行）：

```json
{"round": 1, "actions": ["c7"], "context_reset": false, "events": [
  {"seq": 13, "category": "analysis", "kind": "reasoning", "text": "..."},
  {"seq": 14, "category": "observation", "call_id": "c5", "tool": "view_image", "images": ["images/ab12cd34ef56ab78.png"]},
  {"seq": 15, "category": "analysis", "kind": "reasoning", "text": "..."},
  {"seq": 16, "category": "analysis", "kind": "--evidence", "call_id": "c6", "text": "..."},
  {"seq": 16, "category": "observation", "call_id": "c6", "command": "python robot_ctl.py ack_visual ...", "text": "{...}"},
  {"seq": 17, "category": "action", "call_id": "c7", "command": "...", "code": "..."},
  {"seq": 18, "category": "feedback", "call_id": "c7", "action": "c7", "text": "{...}", "exit_code": 0}
]}
```

## 自动检查（写完后读回导出文件逐项核对，不通过则命令返回非零）

1. 每个动作至少有一条反馈，且反馈都在动作所在的轮；没有反馈的列出来并注明原因（记录里没有返回）。
2. 每条反馈的 `action` 指向本轮的一个动作；每条拆出的理由指向本轮的一个调用。
3. 每张图的文件存在，内容哈希与文件名一致。
4. 原始记录里每条内容（系统提示和环境上下文除外）在导出里恰好出现一次，`seq` 严格递增、不重复（理由条目除外）。

## 不做

改写、润色、翻译、摘要、VQA 问答、训练集划分、外部模型标注。
