# curate：从机器人控制会话整理高质量 VLM 训练数据（VQA）

状态：设计已确认，待用户审阅本文档后写实现计划。

## 1. 目标与约束

把 `rct.py export` 导出的 Codex 机器人控制会话整理成下游 VLM 的训练数据，提升模型对操作任务的理解和决策能力。产物包括：
成功控制案例、失败轨迹及原因分析、控制过程分析，以及由它们生成的多类 VQA 样本。

约束：

- **内容只来自聊天记录。** 问答和案例里的事实只能取自 rollout 中的推理摘要、说明文字、note/evidence 等理由、用户消息、
  工具输出、报错、代码与代码注释。框架运行目录里的判定结果（如 `result.json`）只能当标签，不进入问答内容。
- **agent 只负责编排。** 运行 skill 的 agent 可以划分阶段、选择、删减、合并、调整顺序、改写措辞，不能补充证据里没有的
  事实，不翻译（保持原文语言）。
- **每条内容可追溯。** 每句话标注证据编号，脚本校验；校验不过的内容回退或删除。
- **没有 agent 也能跑通。** 脚本自带模板初稿，`curate build` 可以直接用初稿出数据。
- 沿用现有约束：Python 3.9+、只用标准库、默认脱敏、只读不写原始数据。

不做：外部教师模型标注、人工标注流程、翻译、对图像内容做自动识别。

## 2. 流程与产物

```
rct.py export ... --out traces/                 现有：会话 -> 按步记录
rct.py curate prepare traces/ --out curated/    标签、证据包、模板初稿
  （agent 读 curated/packets/<episode>.json，写 curated/compose/<episode>.json）
rct.py curate check curated/                    校验编排文件，输出逐条结果
rct.py curate build curated/ --out dataset/     生成最终数据集
```

```
curated/
  packets/<episode>.json        证据、标签、模板初稿（脚本写）
  compose/<episode>.json        编排结果（agent 写，可缺省）
  check/<episode>.json          校验结果（脚本写）
dataset/
  vqa/train.jsonl, vqa/val.jsonl, vqa/dataset_info.json, vqa/images/<episode>/
  cases/success/<episode>.md    成功控制案例
  cases/failure/<episode>.md    失败轨迹分析
  process/<episode>.json        过程分析
  curation_report.md            筛选漏斗、分布、校验通过率、隐私扫描、抽查样本
  curation_manifest.json        选项、计数、剔除原因、schema 版本
```

现有 `sft` 命令保留；VQA 的决策题复用 `sft.py` 的提示拼装（任务、历史摘要、上一步结果、当前观测）。

## 3. 标签（`curate prepare`，确定性）

所有标签写入证据包，并作为可引用的证据条目（见第 5 节），让答案里的计数和结论有据可查。

### 3.1 episode 结果标签 `outcome`

`{label: success|failure|unknown, source, evidence[], conflicts[]}`，来源按可信度从高到低：

| source | 依据 |
|--------|------|
| `env` | 适配器读到的运行目录判定（如 cell_queue 的 `result.task_success` / `strict_success`） |
| `tool_output` | 聊天记录里环境打印的结果，按 profile 的 `outcome_patterns` 匹配，如 `"task_success": true` |
| `user` | 最后一个 act 之后的用户消息里的明确确认，按 `user_outcome_patterns` 匹配（"成功了"、"failed" 等） |
| `agent_claim` | 模型在最终消息里的自我声明，同样按模式匹配 |

取最高来源的结论。不同来源结论冲突时记入 `conflicts`，例如 `agent_claim=success` 而 `env=failure`。
只有 `agent_claim` 时，`label` 记为 `unknown`，`claimed` 字段保留声明内容。

决策题和成功案例要求结果来源不低于 `--min-outcome-source`，默认 `tool_output`（即 `env` 或 `tool_output`）；
没有环境输出、靠用户确认结果的框架可以设为 `user`。

### 3.2 每步标签

- `exec`：`ok | failed | unknown`，依据退出码和 profile 的 `failure_patterns`。默认模式包括 Traceback、`Error`、
  `"ok": false`、`IK failed`、`refused`、`not sent`，`exec_reason` 记录命中的文本。
- `hindsight`（根据后续记录判断）：
  - `progress`：执行成功，之后没有对同一调用键的立即重试；
  - `superseded`：下一次同一调用键的调用改了参数（调用键沿用 `adjustments` 的定义：函数名加简短选择参数；
    MCP/CLI 用 signature）；
  - `correction`：它改正了前一个 `superseded` 或失败步，并且自己执行成功；
  - `redundant`：与上一步动作完全相同且没有新的观测；
  - `user_corrected`：这一步之后、下一个 act 之前有用户消息。
  一步可以有多个标签；`role` 按 `correction > superseded > redundant > progress` 取一个主值，`exec=failed` 时为
  `failed_attempt`。
- `context_reset`：这一步之前刚发生过上下文压缩或回合中断。

### 3.3 纠偏链、关键失败步、原因

- **纠偏链** `chains[]`：同一调用键上连续的 `failed_attempt` / `superseded` 步，以第一个 `correction` 结束则
  `resolved=true`，否则 `resolved=false`。
- **关键失败步候选** `critical_candidates[]`（仅失败 episode）：最后一条未解决纠偏链的起点；不存在时取结束前最后 3 个
  act 步；加上之后再没有解决的 `user_corrected` 步。agent 只能在候选里选。
- **原因候选** `causes[]`：针对每条纠偏链和每个关键失败候选，收集
  (a) 失败之后、下一个 act 之前模型的推理和说明文字；(b) 同一区间的用户消息；(c) 报错原文。可信度：
  - `confirmed`：所在纠偏链 `resolved=true`，而且原因证据出现在解决它的那一步之前；
  - `stated`：有 (a) 或 (b)，但没有成功的修复；
  - `error_only`：只有 (c)。

### 3.4 episode 门槛与质量分

剔除或限制条件（原因写进 manifest 和报告）：

| 条件 | 影响 |
|------|------|
| 没有控制证据（现有 `flags.control`） | 整个 episode 不收录 |
| `outcome.label=unknown` | 不出决策题和成功案例；已证实的纠偏链仍可出诊断题和修正题 |
| 信息越界：aux/observe 调用读取了 profile 的 `forbidden_reads` 匹配的路径 | 不出决策题和场景理解题，报告标出 |
| 分析文字覆盖率低于 `--min-analysis` （默认 0.5） | 不出决策题 |
| 命中 profile 的 `exclude_if`（匹配任务提示或适配器 run_config，例如 `track=dev`） | 整个 episode 不收录 |

步级排除：动作在导出时被截断、需要图像的题型没有图、`redundant` 步、`context_reset` 后的第一步。

质量分 `quality{score, parts}`：`outcome_confidence`、`analysis_coverage`、`image_coverage`、`1 - redundant_ratio`、
`correction_success_rate`、`1 - human_intervention_ratio`，各项 0 到 1，`score` 取平均。用于排序和报告，阈值
`--min-quality` 默认 0（不过滤），报告列出各项。

profile 新增字段：`failure_patterns`、`outcome_patterns`、`user_outcome_patterns`、`forbidden_reads`、`exclude_if`
（均为正则列表，`default` 提供通用默认值，框架 profile 可以追加）。

## 4. VQA 题型与信息边界

| `qa_type` | 输入（提问侧） | 答案来源 | 入选条件 |
|-----------|----------------|----------|----------|
| `scene` | 任务、当前步的图像 | 这一步模型对图像的描述（看图确认的 evidence、推理里描述画面的句子） | 该步 `exec=ok` 且 `role=progress` 或 `correction` |
| `progress` | 任务、当前图像、历史摘要 | 阶段划分：当前阶段、已完成阶段 | 有通过校验的阶段划分（agent 编排）；仅模板时不出 |
| `decision` | 同 `sft` | 分析加动作 | `outcome=success`（来源不低于 `--min-outcome-source`），`role` 为 `progress` 或 `correction` |
| `outcome_check` | 上一步动作和 note、动作后的图像（不含工具输出） | 下一步模型对上一步结果的评估 | 上一步 `exec=ok`，且编排时选中了下一步里明确评估结果的句子；仅模板时不出 |
| `failure_diagnosis` | 失败步的上下文、图像、报错 | 原因及其证据 | 默认只用 `confirmed`；`--diagnosis stated` 放宽 |
| `correction` | 失败或被修正步的上下文 | 解决它的那一步的理由和参数变化（`adjustments`） | 纠偏链 `resolved=true` |
| `episode_review` | 任务、最多 6 张关键帧、步骤摘要 | 案例卡片的总结 | 每个 episode 最多 2 条，需通过校验的案例卡片 |

信息边界：

- 输入只包含模型在那个时刻已经拿到的内容。
- 事后信息（后来的修正、最终结果）只能出现在 `outcome_check`、`failure_diagnosis`、`correction`、`episode_review` 的答案里。
- `decision` 和 `scene` 的答案只能引用当前步及以前的证据，校验器强制检查。
- `<image>` 占位与图片一一对应；单条样本默认最多 4 张图（`episode_review` 6 张）。
- `episode_review` 的关键帧：每个阶段第一个带图步骤的第一张图，加上最后一个带图步骤的图；没有阶段划分时，在带图步骤中
  均匀取样。超过 6 张时均匀抽取。

问题文字由模板生成，语言默认跟随该 episode 分析文字的主要语言（`--question-lang auto|zh|en`）。

样本格式（`robot-control-vqa/v1`）：

```json
{"id": "<episode>#<qa_type>#<n>",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "...<image>..."},
              {"role": "assistant", "content": "..."}],
 "images": ["images/<episode>/<file>"],
 "meta": {"schema": "robot-control-vqa/v1", "qa_type": "decision", "episode_id": "...", "steps": [3],
          "outcome": {"label": "success", "source": "env"}, "confidence": "confirmed",
          "evidence": ["s3.reasoning.0", "s3.action"], "composed_by": "agent", "split": "train"}}
```

配比与切分：每个 episode 每类题型设上限（`decision` 30、`episode_review` 2、其它 10）；按归一化后的动作和答案文本去重；
训练集和验证集按 episode 切分（`--val-fraction 0.1`），或按任务切分（`--split-by task`），同一 episode 不跨集合。

## 5. 证据包、编排与校验

### 5.1 证据包 `curated/packets/<episode>.json`（`robot-control-curated/v1`）

- `evidence[]`：`{id, step, field, text}`。编号格式 `s<step>.<field>.<n>`，episode 级为 `ep.<field>.<n>`。字段包括
  `reasoning`、`message`、`rationale`、`user`、`output`、`error`、`code_comment`、`read`、`action`、`final`，以及标签
  事实 `label`（例如 `s3.label.0`："step 3 exec failed: IK failed"；`ep.label.outcome`）。文本来自已脱敏的导出，按
  `--evidence-chars`（默认 1500）截断。
- `labels`：第 3 节的全部标签。
- `drafts[]`：模板初稿，`{id, qa_type, steps, question, answer: [{text, evidence}]}`。
- `todo`：需要 agent 编排的项目：阶段划分、案例卡片、可改写的初稿、关键失败步选择、`outcome_check` 句子选择。

### 5.2 编排文件 `curated/compose/<episode>.json`（agent 写）

```json
{"phases": [{"name": "...", "steps": [0, 4], "summary": [{"text": "...", "evidence": ["s2.rationale.0"]}]}],
 "critical_step": 7,
 "case": {"kind": "success|failure", "title": "...", "sections": [{"heading": "...", "sentences": [...]}]},
 "qa": [{"draft": "<draft id>", "answer": [{"text": "...", "evidence": ["..."]}]}],
 "outcome_checks": [{"step": 5, "evidence": ["s6.reasoning.1"]}]}
```

agent 只能改写已有初稿，不能新增问题；可以不改写某条初稿（保留模板）或在 `qa` 里标 `"drop": true` 删除它。

### 5.3 校验 `curate check`

逐句检查，结果写入 `curated/check/<episode>.json`：

1. 结构：证据编号和初稿编号存在；`critical_step` 在候选里；阶段范围连续、不重叠、覆盖全部 act 步。
2. 引用：每句至少一条证据；`decision`、`scene` 只能引用不晚于当前步的证据。
3. 数字：句中每个数字（归一化正负号、全角负号、末尾的 0、单位）必须出现在所引证据中；引用标签事实里的步号和计数也算。
4. 专有名词：引号内文字、文件路径、形如 `a.b(` 的调用名必须逐字出现在所引证据中。
5. 防编造：句子与所引证据的实词重合度不低于 0.3（英文按去停用词的词，中文按字二元组）。

处理：问答句子不通过时，整条答案回退到模板初稿；案例卡片和阶段总结里不通过的句子删除；阶段划分结构不通过则视为
没有阶段划分。`check` 打印通过率，并随机抽 5 条，把改写文字和所引证据并排显示，供人工抽查。退出码：有结构错误时为 1。

## 6. 其它产物

- `cases/success/<episode>.md`：任务和结果来源、阶段列表、每阶段关键步（图像、判断、动作、参数调整）、已解决的纠偏链。
- `cases/failure/<episode>.md`：结果来源、关键失败步及其图像和报错、原因及可信度、尝试过的修正和结果、失败前达到的阶段。
- `process/<episode>.json`：阶段时间线、纠偏链、人工介入点、各阶段步数和重试次数、质量分各项。
- 没有编排文件时，案例卡片只包含脚本能确定的部分（结果、纠偏链、关键失败候选、报错），不写叙述。
- `curation_report.md`：rollout -> episode -> 收录的漏斗及每层剔除原因；各题型数量、可信度和结果来源分布、各任务数量；
  校验通过率；隐私扫描（复用 `audit`）；抽查样本。

## 7. 代码结构

- `scripts/rct_lib/labels.py`：第 3 节的标签、纠偏链、原因候选、门槛和质量分。
- `scripts/rct_lib/packets.py`：证据条目和模板初稿。
- `scripts/rct_lib/compose_check.py`：第 5.3 节的校验。
- `scripts/rct_lib/vqa.py`：样本生成、配比、去重、切分、案例卡片、过程分析、报告。复用 `sft.py` 的拼装函数和图片放置。
- `scripts/rct.py`：新增 `curate prepare|check|build` 子命令。
- `profiles/default.json`：新增第 3.4 节的默认模式。
- 文档：`SKILL.md` 新增 curate 步骤和 agent 编排规则（含校验失败后怎么改），`reference/schema.md` 新增
  curated/vqa 两个 schema，README 中英文段更新。

## 8. 测试

- 合成会话覆盖：成功 episode 里被修好的失败（`confirmed` 原因）、没修好的失败（关键失败步）、用户纠正、
  `agent_claim` 与 `env` 冲突、读取越界文件、原样重发的空转循环。
- 校验器：合格的编排文件全部通过；编造数字、引用不存在的证据、决策题引用未来证据、凭空句子、阶段重叠各自被拦下，
  并验证回退行为。
- `examples/` 增加一份手写编排文件（模拟 agent）和生成的 `dataset/`，`ExampleTests` 检查提交的产物是最新的，且隐私扫描
  为空。

## 9. 实现顺序

1. 标签与证据包（`labels.py`、`packets.py`、`curate prepare`）及测试。
2. 只用模板的 `curate build`：VQA、案例卡片骨架、过程分析、报告，端到端可用。
3. 编排与校验（`compose_check.py`、`curate check`、build 读取编排结果）及测试。
4. 文档、示例、隐私检查，提交。
