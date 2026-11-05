# curate：从机器人控制会话整理高质量 VLM 训练数据（VQA）

状态：设计已确认（第二版：内容一律原样摘取，数据集中没有引用），待用户审阅本文档后写实现计划。

## 1. 目标与约束

把 `rct.py export` 导出的 Codex 机器人控制会话整理成下游 VLM 的训练数据，提升模型对操作任务的理解和决策能力。产物包括：
成功控制案例、失败轨迹及原因分析、控制过程分析，以及由它们生成的多类 VQA 样本。

核心要求：数据集里的内容绝不是凭空捏造的。

- **实质内容只来自聊天记录，逐字复制。** 推理摘要、说明文字、note/evidence 等理由、用户消息、工具输出、报错、代码，
  以导出后（已脱敏）的文本为准。答案中使用的片段按整行或整句切分，不截断；过长的片段不用。
- **代码生成的只有格式。** 问题模板、案例卡片和过程分析的固定小标题与字段名、固定词表中的标签词（成功、失败、未知；
  已证实、仅陈述、仅报错；题型名）、从记录直接数出来的步号和计数、动作代码外的 Markdown 代码块标记。
- **agent 只做选择，不写任何文字。** 编排文件里只有片段编号、步号、步骤范围、取舍和顺序；格式不允许出现文本字段。
- **数据集中没有引用。** 正文不出现片段编号或任何标记；`meta` 只有筛选字段。溯源信息只保留在工作目录 `curated/`，
  供内部抽查，不随数据集发布。
- **框架运行目录的判定结果只当标签**（如 `result.json`），不进入问答内容。
- 沿用现有约束：Python 3.9+、只用标准库、默认脱敏、只读不写原始数据。

不做：改写、润色、翻译、外部教师模型标注、人工标注流程、对图像内容做自动识别。

代价：答案就是驱动模型当时的原话，可能中英混杂、不够精炼。

## 2. 流程与产物

```
rct.py export ... --out traces/                 现有：会话 -> 按步记录
rct.py curate prepare traces/ --out curated/    标签、原文片段、规则选出的初稿
  （agent 读 curated/packets/<episode>.json，写 curated/compose/<episode>.json：只有选择）
rct.py curate check curated/                    校验编排文件的结构和信息边界
rct.py curate build curated/ --out dataset/     按选择复制原文，生成最终数据集
```

```
curated/                        工作目录，不发布
  packets/<episode>.json        原文片段、标签、初稿（脚本写）
  compose/<episode>.json        选择结果（agent 写，可缺省）
  check/<episode>.json          校验结果（脚本写）
  provenance.jsonl              每条样本用到的片段编号及其出处（脚本写）
dataset/                        发布的数据集
  vqa/train.jsonl, vqa/val.jsonl, vqa/dataset_info.json, vqa/images/<episode>/
  cases/success/<episode>.md    成功控制案例
  cases/failure/<episode>.md    失败轨迹分析
  process/<episode>.json        过程分析
  curation_report.md            筛选漏斗、分布、校验结果、隐私扫描、抽查样本
  curation_manifest.json        选项、计数、剔除原因、schema 版本
```

现有 `sft` 命令保留；VQA 的决策题复用 `sft.py` 的提示拼装（任务、历史摘要、上一步结果、当前观测）。

## 3. 标签（`curate prepare`，确定性）

标签是结构化字段，用于筛选、分类和组织内容，本身不作为句子写进答案；答案里出现的标签只能是固定词表里的词。

### 3.1 episode 结果标签 `outcome`

`{label: success|failure|unknown, source, conflicts[]}`，来源按可信度从高到低：

| source | 依据 |
|--------|------|
| `env` | 适配器读到的运行目录判定（如 cell_queue 的 `result.task_success` / `strict_success`） |
| `tool_output` | 聊天记录里环境打印的结果，按 profile 的 `outcome_patterns` 匹配，如 `"task_success": true` |
| `user` | 最后一个 act 之后的用户消息里的明确确认，按 `user_outcome_patterns` 匹配（"成功了"、"failed" 等） |
| `agent_claim` | 模型在最终消息里的自我声明，同样按模式匹配 |

取最高来源的结论。不同来源结论冲突时记入 `conflicts`，例如 `agent_claim=success` 而 `env=failure`。
只有 `agent_claim` 时，`label` 记为 `unknown`，`claimed` 字段保留声明。

决策题和成功案例要求结果来源不低于 `--min-outcome-source`，默认 `tool_output`（即 `env` 或 `tool_output`）；
没有环境输出、靠用户确认结果的框架可以设为 `user`。

### 3.2 每步标签

- `exec`：`ok | failed | unknown`，依据退出码和 profile 的 `failure_patterns`。默认模式包括 Traceback、`Error`、
  `"ok": false`、`IK failed`、`refused`、`not sent`。
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
- **原因候选** `causes[]`：针对每条纠偏链和每个关键失败候选，列出可用的原文片段：
  (a) 失败之后、下一个 act 之前模型的推理和说明文字；(b) 同一区间的用户消息；(c) 报错原文。可信度：
  - `confirmed`：所在纠偏链 `resolved=true`，而且 (a) 或 (b) 出现在解决它的那一步之前；
  - `stated`：有 (a) 或 (b)，但没有成功的修复；
  - `error_only`：只有 (c)。

### 3.4 episode 门槛与质量分

| 条件 | 影响 |
|------|------|
| 没有控制证据（现有 `flags.control`） | 整个 episode 不收录 |
| `outcome.label=unknown` | 不出决策题和成功案例；已证实的纠偏链仍可出诊断题和修正题 |
| 信息越界：aux/observe 调用读取了 profile 的 `forbidden_reads` 匹配的路径 | 不出决策题和场景理解题，报告标出 |
| 分析文字覆盖率低于 `--min-analysis`（默认 0.5） | 不出决策题 |
| 命中 profile 的 `exclude_if`（匹配任务提示或适配器 run_config，例如 `track=dev`） | 整个 episode 不收录 |

步级排除：动作在导出时被截断、需要图像的题型没有图、`redundant` 步、`context_reset` 后的第一步。

质量分 `quality{score, parts}`：`outcome_confidence`、`analysis_coverage`、`image_coverage`、`1 - redundant_ratio`、
`correction_success_rate`、`1 - human_intervention_ratio`，各项 0 到 1，`score` 取平均。用于排序和报告；
`--min-quality` 默认 0（不过滤），报告列出各项。

profile 新增字段：`failure_patterns`、`outcome_patterns`、`user_outcome_patterns`、`forbidden_reads`、`exclude_if`
（均为正则列表，`default` 提供通用默认值，框架 profile 可以追加）。

## 4. VQA 题型与信息边界

答案全部由原文片段按固定格式拼成；下表"答案"一列说明片段从哪里选。

| `qa_type` | 输入（提问侧） | 答案 | 入选条件 |
|-----------|----------------|------|----------|
| `scene` | 任务、当前步的图像 | 这一步对图像的描述原文：看图确认的 evidence、推理里描述画面的句子 | 该步 `exec=ok` 且 `role` 为 `progress` 或 `correction` |
| `progress` | 任务、当前图像、历史摘要 | 当前阶段和已完成阶段的阶段名（阶段名本身是原文片段，见 5.2） | 有通过校验的阶段划分；仅规则初稿时不出 |
| `decision` | 同 `sft` | 这一步的推理和说明文字原文，加原始动作（代码、命令或工具参数） | `outcome=success`（来源不低于 `--min-outcome-source`），`role` 为 `progress` 或 `correction` |
| `outcome_check` | 上一步动作和 note、动作后的图像（不含工具输出） | 下一步里评估上一步结果的原句 | 上一步 `exec=ok`，且编排时选中了评估原句；仅规则初稿时不出 |
| `failure_diagnosis` | 失败步的上下文、图像、报错 | 原因片段原文（固定小标题"原因"），可附报错原文 | 默认只用 `confirmed`；`--diagnosis stated` 放宽 |
| `correction` | 失败或被修正步的上下文 | 解决它的那一步的理由原文，加该步的原始动作 | 纠偏链 `resolved=true` |
| `episode_review` | 任务、最多 6 张关键帧、步骤摘要 | 固定小标题下的阶段名、各阶段选中的关键片段、结果标签词 | 每个 episode 最多 2 条，需有通过校验的阶段划分 |

不使用计算出来的数值：修正题不写参数差值，模型从输入里的原动作和答案里的新动作自己对比。

信息边界：

- 输入只包含模型在那个时刻已经拿到的内容。
- 事后信息（后来的修正、最终结果）只能出现在 `outcome_check`、`failure_diagnosis`、`correction`、`episode_review` 的答案里。
- `decision` 和 `scene` 的答案只能使用当前步及以前的片段，`check` 强制检查。
- 输入侧的长输出可以按现有规则截断（保留首尾）；答案侧不截断。
- `<image>` 占位与图片一一对应；单条样本默认最多 4 张图（`episode_review` 6 张）。
- `episode_review` 的关键帧：每个阶段第一个带图步骤的第一张图，加上最后一个带图步骤的图；超过 6 张时均匀抽取。

问题文字由固定模板生成，语言默认跟随该 episode 分析文字的主要语言（`--question-lang auto|zh|en`）。

样本格式（`robot-control-vqa/v1`）：

```json
{"id": "<episode>#<qa_type>#<n>",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "...<image>..."},
              {"role": "assistant", "content": "..."}],
 "images": ["images/<episode>/<file>"],
 "meta": {"schema": "robot-control-vqa/v1", "qa_type": "decision", "episode_id": "...", "steps": [3],
          "outcome": {"label": "success", "source": "env"}, "confidence": "confirmed", "split": "train"}}
```

`meta` 只有上面这些筛选字段，没有片段编号或出处。

配比与切分：每个 episode 每类题型设上限（`decision` 30、`episode_review` 2、其它 10）；按归一化后的动作和答案文本去重；
训练集和验证集按 episode 切分（`--val-fraction 0.1`），或按任务切分（`--split-by task`），同一 episode 不跨集合。

## 5. 原文片段、选择与校验

### 5.1 证据包 `curated/packets/<episode>.json`（`robot-control-curated/v1`，只在工作目录）

- `spans[]`：`{id, step, field, text}`。把每步的推理、说明文字、理由、用户消息、工具输出、报错、代码注释、读取结果按行
  或句切成片段；动作（代码、命令、工具参数）整体作为一个片段。编号格式 `s<step>.<field>.<n>`，episode 级为
  `ep.<field>.<n>`。
- `labels`：第 3 节的全部标签。
- `drafts[]`：规则选出的初稿，`{id, qa_type, steps, spans: [...]}`，只有片段编号。
- `todo`：需要 agent 选择的项目：阶段划分和阶段名、案例的关键片段、初稿的片段取舍、关键失败步、`outcome_check` 评估原句。

### 5.2 编排文件 `curated/compose/<episode>.json`（agent 写，只有选择）

```json
{"phases": [{"steps": [0, 4], "title": "s0.rationale.0", "key": ["s2.reasoning.1", "s3.message.0"]}],
 "critical_step": 7,
 "drafts": [{"id": "<draft id>", "spans": ["s5.reasoning.0", "s5.reasoning.2"]}, {"id": "<draft id>", "drop": true}],
 "outcome_checks": [{"step": 5, "spans": ["s6.reasoning.1"]}]}
```

- 阶段名 `title` 必须是该阶段内某一步的 note、理由或说明文字片段。
- 只能调整已有初稿的片段取舍和顺序，或删除初稿，不能新增题目。
- 格式中没有任何文本字段；出现未知字段或字符串值不是片段编号的，整个文件判为无效。

### 5.3 校验 `curate check`

结果写入 `curated/check/<episode>.json`：

1. 结构：只含允许的字段；片段编号、初稿编号存在；`critical_step` 在候选里；阶段范围连续、不重叠、覆盖全部 act 步；
   阶段名片段属于该阶段。
2. 信息边界：`decision`、`scene` 只能选不晚于当前步的片段；`failure_diagnosis` 的原因片段来自 3.3 列出的区间；
   `correction` 的片段来自解决它的那一步。
3. 片段字段：每类题型只能选允许的字段，比如 `scene` 只能选描述画面的字段，不能选工具输出。

编排文件无效或某项不通过时，该项回退到规则初稿；阶段划分不通过则视为没有阶段划分。退出码：有无效编排文件时为 1。

### 5.4 生成时的原文核对（`curate build`）

`build` 按片段编号从证据包复制文本，再逐段核对：每段内容必须能在该 episode 的导出记录（`steps.jsonl`、`episode.json`）
中逐字找到（只忽略首尾空白）。任何一段找不到，整条样本或整句丢弃，并在报告里计数。这一步防的是程序自身的错误。

生成后再检查一遍发布目录：样本正文和 `meta` 里不得出现片段编号格式的字符串，`meta` 只能有第 4 节列出的字段。

溯源写入 `curated/provenance.jsonl`：`{sample_id, spans: [{id, step, field}]}`，不复制到 `dataset/`。

## 6. 其它产物

- `cases/success/<episode>.md`：固定小标题下的内容：任务（原文）、结果标签词及来源、各阶段的阶段名和关键片段、
  关键步的图像和原始动作、已解决的纠偏链（失败动作、原因片段、修正动作，均为原文）。
- `cases/failure/<episode>.md`：结果标签词及来源、关键失败步的图像、原始动作和报错原文、原因片段及可信度、
  尝试过的修正（原始动作及其执行结果）、失败前到达的阶段。
- `process/<episode>.json`：阶段（步骤范围和阶段名片段文本）、纠偏链、人工介入点（用户消息原文）、各阶段步数和重试次数、
  质量分各项。
- 没有编排文件时，案例卡片只包含规则能确定的部分（结果、纠偏链、关键失败候选、报错），没有阶段。
- `curation_report.md`：rollout -> episode -> 收录的漏斗及每层剔除原因；各题型数量、可信度和结果来源分布、各任务数量；
  编排校验和原文核对的结果；隐私扫描（复用 `audit`）；抽查样本。

## 7. 代码结构

- `scripts/rct_lib/labels.py`：第 3 节的标签、纠偏链、原因候选、门槛和质量分。
- `scripts/rct_lib/packets.py`：片段切分和规则初稿。
- `scripts/rct_lib/compose_check.py`：第 5.3 节的校验。
- `scripts/rct_lib/vqa.py`：按选择复制原文、原文核对、配比、去重、切分、案例卡片、过程分析、报告。复用 `sft.py` 的拼装
  函数和图片放置。
- `scripts/rct.py`：新增 `curate prepare|check|build` 子命令。
- `profiles/default.json`：新增第 3.4 节的默认模式。
- 文档：`SKILL.md` 新增 curate 步骤和 agent 的选择规则（只选编号、不写文字、校验失败后怎么改），`reference/schema.md`
  新增 curated/vqa 两个 schema，README 中英文段更新。

## 8. 测试

- 合成会话覆盖：成功 episode 里被修好的失败（`confirmed` 原因）、没修好的失败（关键失败步）、用户纠正、
  `agent_claim` 与 `env` 冲突、读取越界文件、原样重发的空转循环。
- 编排校验：合格文件通过；含文本字段、引用不存在的片段、决策题选未来片段、阶段重叠、阶段名不在阶段内各自被拦下，
  并验证回退行为。
- 原文核对：篡改证据包里某个片段的文本后，对应样本被丢弃。
- 发布目录检查：数据集中没有片段编号，`meta` 字段集合固定；每条答案的每段都能在导出记录中逐字找到。
- `examples/` 增加一份手写编排文件（模拟 agent）和生成的 `dataset/`，`ExampleTests` 检查提交的产物是最新的，且隐私扫描
  为空。

## 9. 实现顺序

1. 标签与证据包（`labels.py`、`packets.py`、`curate prepare`）及测试。
2. 只用规则初稿的 `curate build`：VQA、案例卡片、过程分析、原文核对、报告，端到端可用。
3. 编排与校验（`compose_check.py`、`curate check`、build 读取选择结果）及测试。
4. 文档、示例、隐私检查，提交。
