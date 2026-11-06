# robot-control-traces

An agent skill and a dependency-free CLI that turn **Codex conversation logs of AI-driven robot control** into a
step-level dataset:

```
observation (images the model saw, state reads, previous feedback)
  -> analysis (reasoning summaries, messages, stated rationale, code comments, user corrections)
  -> action   (command / code, extracted control calls, parameter adjustments vs. earlier steps)
  -> outcome  (output, exit code, returned images)
```

It works with any control framework: whatever drives the robot (Python cells, a CLI, ROS 2, an MCP server), the
Codex rollout under `~/.codex/sessions` records what the model looked at, said and executed. JSON profiles say
which calls act and which observe; adapters can add ground truth from run directories (built in: file-queue
"cell-queue" sessions with commands, responses and observation frames).

## Example output

[`examples/`](examples/) holds two fictional sessions and everything the exporter produces from them:

- [episode.md of a cell-queue session](examples/output/episodes/2026-01-15_0199c0de_00/episode.md): a block is
  located from the head camera, corrected from the wrist camera, a failed carry is retried lower and slower after a
  user request, and the run directory adds native action ranges, primitive calls and the success label;
- [episode.md of an MCP session](examples/output/episodes/2026-01-16_0199c0de_00/episode.md): `robot__move_arm`
  calls refined from camera frames, then an SDK script, transcript only;
- `steps.jsonl`, `timeline.jsonl`, `episode.json`, `images/` and `manifest.json` next to them;
- [`sft/train.jsonl`](examples/sft/train.jsonl): the same steps as training samples, one per action;
- [`chat/`](examples/chat/): the raw conversations, every piece labelled and grouped by decision round.

One step of the first episode, as rendered in `episode.md`:

> **Step 1 · `python robot_ctl.py cell`**
>
> **Observation** ![cam_wrist](examples/output/episodes/2026-01-15_0199c0de_00/images/9039e148d017bd44.png)
> read `view_image`: cam_wrist.png; state: native_action_index=48 …
>
> **Analysis** "The wrist view shows the block left of the finger centre line, so the head estimate was off in y
> by about 1.5 cm." · evidence: Wrist: block ~9 px (about 1.5 cm) left of the finger centre · note: correct y and
> descend
>
> **Action** `robot.move_to([0.42, -0.135, 0.13], speed=0.15)`, `robot.move_to([0.42, -0.135, 0.045], speed=0.05)`;
> adjustment vs step 0: args[0] Δ [0.0, -0.015, 0.0]
>
> **Outcome** exit 0 · native actions 48 → 90 (+42) · 2 primitive calls

Regenerate them with `python3 examples/make_examples.py`.

## Install as a skill

Copy or symlink `skills/robot-control-traces/` into your agent's skill directory:

| agent | location |
|-------|----------|
| Codex CLI | `~/.codex/skills/robot-control-traces/` |
| Claude Code | `~/.claude/skills/robot-control-traces/` |
| Cursor | `.cursor/skills/robot-control-traces/` in a project |

```bash
git clone https://github.com/1018weijia/robot-control-traces.git
ln -s "$PWD/robot-control-traces/skills/robot-control-traces" ~/.codex/skills/robot-control-traces
```

Then ask the agent, e.g. "export my robot control sessions from Codex as a dataset". The skill walks it through
scanning, checking roles, writing a profile if needed, exporting and validating.

## Use the CLI directly

Python 3.9+, standard library only.

```bash
RCT="python3 skills/robot-control-traces/scripts/rct.py"
$RCT scan                                   # rollouts in $CODEX_HOME (default ~/.codex) that controlled something
$RCT scan --detail ~/.codex/sessions/2026/01/15/rollout-....jsonl   # role of every command / tool
$RCT export --auto --since 2026-01-01 --out traces/                 # write + validate
$RCT export --rollout a.jsonl --profile my_framework.json --out traces/
$RCT export --auto --out traces/ --remap-path /old/runs=/data/runs  # run directories moved since recording
$RCT validate traces/
$RCT sft traces/ --out sft/                 # one training sample per action (LLaMA-Factory sharegpt layout)
$RCT chat --auto --out chat/                # the raw conversation, labelled and grouped by decision round
$RCT audit sft/ some_other_dataset/         # privacy scan of any JSON / JSONL / Markdown / TSV files
$RCT profiles
```

A rollout is exported only if at least one act step carries robot-control evidence (`--min-control-steps`): a call
matched by a framework profile, a robot MCP server or ROS / robot CLI, or code with recognised control calls. Agent
infrastructure (sub-agents, Codex app tools), GitHub or file-system MCP tools and research scripts are not control.

Output (`robot-control-trace/v1`):

```
traces/
  manifest.json
  episodes/<date>_<thread>_<NN>/
    steps.jsonl      one record per action: observation / analysis / action / outcome
    episode.md       the same steps as a readable report with images
    timeline.jsonl   every normalized event, for custom segmentation
    episode.json     task prompt, models, profile, coverage stats, adapter results (e.g. success labels)
    images/          frames the model received, named by content hash
```

`sft` writes `robot-control-sft/v1`:

```
sft/
  train.jsonl          {"messages": [system, user, assistant], "images": [...], "meta": {...}}
  dataset_info.json    LLaMA-Factory entry "robot_control_sft" (sharegpt, columns messages / images)
  sft_manifest.json    options, counts, skipped steps and why, redaction counts
  images/<episode>/    the observation frames the samples reference
```

Each sample is one decision: the user turn holds the task, a one-line summary of the last `--history` steps, the
result of the previous action and the current observation (`<image>` placeholders, state reads, user messages);
the assistant turn is the recorded analysis followed by the exact command or code. Long episodes therefore stay
within normal context lengths instead of becoming one sample of hundreds of images. Failed actions are kept as
targets by default (their failure is shown in the next prompt); `--drop-failed` and `--success-only` filter them.

`chat` writes `robot-control-chat/v1`, the conversation itself without any reshaping:

```
chat/
  manifest.json                 sessions written / skipped and why, redaction counts, check result
  sessions/<date>_<thread>/
    rounds.jsonl                one decision round per line
    session.json                task, models, counts, actions that never returned
    images/                     frames the model received, named by content hash
```

Every piece of the rollout appears once, verbatim, in file order, labelled `task`, `observation`, `analysis`,
`action`, `feedback` or `other`. A round runs from the end of the previous round to the last feedback of its own
actions, so the images and reasoning an action was based on, the action and what it returned are on one line.
An action and its feedback share a `call_id`; feedback fetched later (`write_stdin`, a poll command) names its
action in `action`; `--note` / `--evidence` values and MCP `reason` arguments are repeated as `analysis` entries
next to their call. The export is read back and checked (order, pairing, image hashes, nothing lost or
duplicated); the command exits non-zero on any error. Example:
[`examples/chat/`](examples/chat/).

Field reference: [skills/robot-control-traces/reference/schema.md](skills/robot-control-traces/reference/schema.md).
Profiles for other frameworks (ROS 2, MCP servers, SDK scripts) and writing adapters:
[skills/robot-control-traces/reference/profiles.md](skills/robot-control-traces/reference/profiles.md).

## Privacy

- Only `sessions/` and `archived_sessions/` are read; `auth.json` and config are never touched. Rollouts and run
  directories are never modified.
- Redaction is on by default: home directories become `~`, external volume names `/Volumes/<volume>`, private
  IPv4 addresses `<private-ip>`, emails and API keys / tokens / passwords are replaced, system prompts and git
  remotes are not exported. Add `--redact-path OLD=NEW` for project or customer names.
- `validate` and `sft` include a privacy scan; `rct.py audit PATH...` runs it on any dataset, including ones
  converted by other tools, and exits non-zero when something the redaction rules would change remains.
- Images are part of the data. Review them (`episode.md` shows them inline) before publishing, or export with
  `--images none`.

## Development

```bash
python3 -m unittest discover -s tests
```

Tests build synthetic rollouts and run directories; one test regenerates `examples/` and checks the committed
output is current. No Codex installation is needed.

## 中文说明

把 Codex 里和 AI 对话控制机器人的记录（`~/.codex/sessions` 下的 rollout）整理成按步组织的数据集。每一步包含
观测（模型看过的图像、状态读取、上一步反馈）、分析（推理摘要、说明文字、`--note` / `--evidence` 等理由、代码注释、
用户纠正）、动作（命令或代码、提取出的控制调用、与之前同类调用相比的参数调整量）和结果。每个 episode 同时生成
`episode.md`，可以直接阅读和复核。完整示例（输入和产物）见 `examples/`。任何控制框架都可以用：运行
`rct.py scan --detail` 查看每类调用被判成什么角色，判错了就写一个 JSON profile（见 `reference/profiles.md`）。
文件队列式（cell-queue）会话会自动和运行目录对齐，补充原生动作区间、primitive 调用、观测状态和成功标签。
只有含机器人控制证据的 rollout 才会导出，子 agent、GitHub、文件系统等工具调用不算控制。
`rct.py sft` 把导出结果转成按步的训练样本（每个动作一条，LLaMA-Factory sharegpt 格式，带 `<image>` 占位），
`rct.py chat` 原样导出聊天记录，不做任何改写：每条内容标上类别（任务、观测、分析、动作、反馈、其他），
按决策轮次分组，一轮一行。一轮从上一轮的最后一条反馈之后开始，到本轮动作的最后一条反馈为止，所以某个动作之前看过的图、
做过的分析，这个动作本身，以及它的返回都在同一行里。动作和反馈用同一个 `call_id` 对应，异步取回的结果用 `action`
指回原动作。导出后会读回文件逐项核对（顺序、配对、图像哈希、内容不丢不重），有错误时命令返回非零。
`rct.py audit` 可以对任意数据集（包括别的工具转换的）做隐私扫描。默认开启脱敏（home 路径、外接卷名、内网 IP、
邮箱、密钥），从不读取 `auth.json`。

## License

Apache-2.0
