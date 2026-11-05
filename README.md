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
- `steps.jsonl`, `timeline.jsonl`, `episode.json`, `images/` and `manifest.json` next to them.

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
$RCT profiles
```

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

Field reference: [skills/robot-control-traces/reference/schema.md](skills/robot-control-traces/reference/schema.md).
Profiles for other frameworks (ROS 2, MCP servers, SDK scripts) and writing adapters:
[skills/robot-control-traces/reference/profiles.md](skills/robot-control-traces/reference/profiles.md).

## Privacy

- Only `sessions/` and `archived_sessions/` are read; `auth.json` and config are never touched. Rollouts and run
  directories are never modified.
- Redaction is on by default: home directories become `~`, emails and API keys / tokens / passwords are replaced,
  system prompts and git remotes are not exported. Add `--redact-path OLD=NEW` for project or customer names.
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
默认开启脱敏，从不读取 `auth.json`。

## License

Apache-2.0
