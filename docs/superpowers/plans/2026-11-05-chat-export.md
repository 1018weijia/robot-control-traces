# chat 导出实现计划

依据：`docs/superpowers/specs/2026-11-05-chat-export-design.md`。每步先写失败的测试，再实现，跑全量测试后提交。

## 1. 读取器记录原始位置（`rct_lib/codex.py`）

- `Event` 新增 `raw`（产生它的记录在文件中的行序）、`output_raw`（返回记录的行序）、`own_output` / `own_exit_code`
  （`write_stdin` 续读结果并入之前的原始输出）。
- `event_msg` 兜底消息、`exec_command_end`、`mcp_tool_call_end` 也记下自己的行序。
- 测试：并行调用（调用 A、调用 B、返回 A、返回 B）的行序；续读后 `own_output` 仍是首段输出。现有测试不变。

## 2. 分组与分类（新文件 `rct_lib/chat.py`）

- `items(rollout)`：事件与返回按行序合并成一条流，每项带类别；`seq` 从 1 开始。
- `rounds(items)`：按"本轮所有动作的最后一条反馈"切轮；最后剩余的是 `final`。
- `rationale` 复用 `steps.rationale` 的取值规则，拆成 analysis 条目。
- 图像按内容哈希存入 `images/`，复用 `Writer._save_image` 的命名规则。
- 控制判定复用 `export.analyse`（不跑 adapter），与 `export` 的跳过规则一致。
- 测试：轮次边界、并行动作同轮、异步续读留在本轮、反馈带图、压缩标记、理由拆分、脱敏、无控制证据的会话被跳过。

## 3. 自动检查（`chat.check(out)`）

- 读回 `rounds.jsonl` 和 `images/`，按设计文档第 1–4 条核对；`session.json` 的计数与原始记录比对"恰好一次"。
- 测试：正常导出无错误；手工篡改图片、删掉一条反馈后能报错。

## 4. 命令行与文档

- `rct.py chat`：参数沿用 `export`（`--rollout`/`--auto`/`--out`/`--overwrite`/`--images seen|none`/
  `--redact-path`/`--min-control-steps`/`--max-chars`，默认 0 表示不截断）。
- README（含中文部分）、SKILL.md、`reference/schema.md` 增加 chat 小节；示例目录增加 chat 输出；隐私扫描通过。
