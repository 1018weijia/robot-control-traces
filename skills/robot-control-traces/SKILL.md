---
name: robot-control-traces
description: Converts Codex conversation logs (rollout JSONL under ~/.codex/sessions) of AI-driven robot, simulator or embodied-agent control into a step-level dataset of observation, analysis, action (calls and parameter adjustments) and outcome, including the camera images the model saw, step-level SFT samples from it, and a verbatim export of the conversation labelled by role and grouped by decision round. Works for any control framework through JSON profiles; file-queue (cell-queue) sessions also get ground truth from their run directories. Use when the user wants to collect, export, organize, clean or share robot control sessions, agent control traces, interaction or action logs, or Codex chat history as training or evaluation data.
---

# Robot control traces

`scripts/rct.py` (Python 3.9+, standard library only) reads Codex rollouts and writes a step dataset, one folder
per episode: `steps.jsonl` (the dataset), `episode.md` (the same steps as a readable report with images),
`timeline.jsonl` (every event in order), `episode.json` and `images/`. With `chat` it writes the conversation itself
instead, verbatim and labelled by role.

## How a transcript becomes steps

Every tool call in a rollout gets a role:

| role | meaning | examples |
|------|---------|----------|
| `act` | changes the robot or world; **one step per act call** | run a control cell, `ros2 topic pub`, MCP `move_arm` |
| `observe` | reads images or state before acting | `view_image`, camera / state queries, `ack_visual` |
| `result` | delivers the output of the previous act | `write_stdin` polling, `poll` |
| `aux` | everything else (editing files, `ls`, git) | `apply_patch`, `sed -n` |

A step's observation and analysis are everything between the previous act and this one: images and state reads,
reasoning summaries, assistant messages, user interjections, rationale options such as `--note`, and comments in the
executed code. The action holds the command, the executed code (recovered from heredocs, `python -c`, or files
written earlier in the transcript), the extracted control calls with literal arguments, and the numeric changes
against the previous step with the same call. The outcome holds output, exit code and images returned.

Roles come from profiles: explicit `--profile`, then auto-detected built-ins (`cell-queue`), then `default`
keyword heuristics. Unmatched calls are `aux`. Text typed into a running process (`write_stdin` with `chars`) is
classified like a command; typed text no rule knows is an `act` when an act call started the process, while an empty
`write_stdin` polls and is the `result` of the latest input.

An act step has `flags.control` when it is evidence of robot control: matched by a non-default profile, by a robot
MCP server / ROS / robot CLI rule, or code with extracted control calls. Generic `default` matches (an MCP tool
named `move_file`, a research script) are acts without that flag, and rollouts with no control step are skipped.
Sub-agent and Codex app tools (`collaboration.*`, `multi_agent*.*`, `mcp__codex*`) are `aux` unless a profile says
otherwise.

## Workflow

Copy this checklist and track progress:

```
- [ ] 1. scan: find rollouts that controlled something
- [ ] 2. scan --detail on 1-2 representative rollouts: check roles
- [ ] 3. if roles are wrong: write a profile, re-check with --profile
- [ ] 4. export steps, and/or chat (4b) if the user wants the raw conversation
- [ ] 5. validate and spot-check 2-3 steps or rounds
- [ ] 6. if training samples are wanted: sft on the step export
- [ ] 7. report to the user
```

Below, `RCT` is `python3 <this skill directory>/scripts/rct.py`.

**1. Scan.** `RCT scan` (add `--since 2026-01-01`, `--codex-home DIR`, `--all` to include rollouts without act
calls). Each entry shows act/observe counts, images, models, the most common act signature and the rollout's path
(for `--detail` and `--rollout`).

**2. Check roles.** `RCT scan --detail <rollout.jsonl>` prints every command/tool signature with its role, the rule
that assigned it and an example. Judge it against what the agent was doing:

- the calls that move the robot or send commands must be `act`; camera/state reads should be `observe`;
- `act 0` means the framework is unknown: write a profile;
- file edits, builds or log reads marked `act` mean the default heuristics are too broad: override them;
- rollouts listed by scan but skipped by export with "no robot-control evidence" are usually not robot sessions
  (coding, research, agent orchestration); if one is, write a profile for its control calls;
- if the agent runs control code from files, check that `code_file_options` covers the option naming the file;
  after export, `action.code_source` shows where each step's code came from.

**3. Write a profile** when needed, as a JSON file in the user's project (format and examples in
[reference/profiles.md](reference/profiles.md)). Minimal shape:

```json
{"name": "mybot",
 "rules": [{"role": "act", "kind": "shell", "command": "\\bmybot_ctl\\b[^\\n]*\\s(move|grip)\\b"},
           {"role": "observe", "kind": "shell", "command": "\\bmybot_ctl\\b[^\\n]*\\s(snap|state)\\b"}],
 "rationale_options": ["--why"],
 "action_calls": "^bot\\.(move\\w*|grip)$"}
```

Re-run `RCT scan --detail <rollout> --profile path/to/mybot.json` until the table is right. Explicit profiles take
precedence; `default` still classifies the rest. `--no-auto-profile` turns off built-in detection.

**4. Export.**

```bash
RCT export --auto --out traces/               # every rollout scan lists (filtered by --since)
RCT export --rollout a.jsonl b.jsonl --out traces/ --profile mybot.json
```

Useful options: `--images seen|all|none` (default `seen`: images the model received; `all` also copies adapter
run-directory frames), `--redact-path /data/acme=<data>`, `--max-chars N` (output clip, default 4000),
`--no-adapters`, `--read-local-files` (read code files missing from the transcript from disk; their current content
may differ), `--remap-path RECORDED=LOCAL` (run directories copied or moved since the session; RECORDED is the
prefix the commands and image paths in `scan --detail` show), `--min-act-steps N`,
`--min-control-steps N` (default 1; 0 keeps rollouts without control evidence), `--overwrite` (only replaces a
previous export of the same kind). A selected rollout whose content is the start of another rollout of the same
session (an identical file, or a copy taken while it was running) is skipped as a copy.

**4b. Raw chat.** When the user wants the conversation itself rather than reshaped steps:
`RCT chat --auto --out chat/` (or `--rollout ...`). Shared options: `--codex-home`, `--since`, `--profile`,
`--redact-path`, `--min-control-steps`, `--overwrite`; `--images seen|none`; `--max-chars N` (default 0: texts stay
whole). It writes `sessions/<id>/rounds.jsonl`, one decision round per line: every piece of the rollout verbatim
(terminal colour codes and progress-bar redraws removed), in file order, labelled `task` / `observation` /
`analysis` / `action` / `feedback` / `other`; a round ends with the last feedback of its actions. Actions and
feedback share `call_id`; late results carry `action`. A call's rationale values (`--note`, a `reason` argument) are
also listed as `analysis` right before it, with its `call_id` and the option name as `kind`; filter on that `kind`
when only the recorded pieces are wanted.

**5. Validate.** Export runs validation automatically; `RCT validate traces/` re-runs it. It checks the schema,
step order, adjustment references and image hashes, runs a privacy scan, then prints coverage and sample steps.
Chat export checks itself (order, action/feedback pairing, image hashes, every recorded piece exactly once, images
against the raw rollout); `RCT validate chat/` re-runs all of it except the rollout comparison. Fix any reported
error before using the data, then spot-check:

- steps (`episode.md` of 1-2 episodes, or `steps.jsonl`): `analysis` contains the reasoning that led to the action
  and `observation.images` the frames it was based on; `action.calls` lists the control calls, and
  `action.adjustments` shows sensible deltas;
- chat (one `rounds.jsonl`): the analysis before each action is about the images before it;
- both: no names, hosts, tokens or private paths remain. The privacy scan finds home paths, IPs, emails and keys,
  not project or customer names, nor user names in paths outside home (`/mnt/<user>/…`): ask the user which names
  are private, search for them (`grep -ril acme traces/`) and add `--redact-path`.

Low `with_images` usually means the model rarely viewed images; low `with_code` means control code was in files the
transcript never showed (try `--read-local-files`); `enriched 0%` for a cell-queue run means the run directory was
not found (moved: use `--remap-path`).

**6. SFT samples.** From a step export: `RCT sft traces/ --out sft/` writes `train.jsonl` with one sample per
control step (prompt: task, short history, previous result, current images and reads; target: the analysis
written after that result, then the exact command or code), plus `dataset_info.json` for LLaMA-Factory and
`sft_manifest.json`. Options: `--history N`, `--max-images N`, `--drop-failed`, `--success-only` (needs adapter
success labels), `--keep-no-control`. It checks that every `<image>` placeholder has a file and runs the privacy
scan. Do not train on whole episodes as single samples: long sessions hold hundreds of images and far exceed usual
cutoffs.

**7. Report** the output path, episode and step counts, coverage, profile used, rollouts skipped and why, and
anything that stayed unmatched. For chat: sessions, rounds with actions (plus a `"final"` line where text follows
the last action), actions without feedback, the check result, and that rationale values are repeated as
`analysis` entries.

## Privacy rules

- Never read, copy or print `auth.json` or other credentials in `$CODEX_HOME`. The script only reads
  `sessions/` and `archived_sessions/`.
- Keep redaction on (default): home paths become `~`, external volume names `/Volumes/<volume>`, private IPv4
  addresses `<private-ip>`, emails `<email>`, API keys / tokens / passwords `<redacted>` (or `<redacted:kind>`).
  Add `--redact-path OLD=NEW` for project or customer names. `--no-redact` is for private use only.
- Before sharing any dataset, including one produced by another converter, run `RCT audit <dir or files>`; it
  decodes JSON, masks what it prints and exits 1 if findings remain.
- Images are exported because they are the observations; tell the user to review them before sharing
  (`--images none` drops them).
- Only export Codex homes the user owns or has consent for. The script is read-only: it never modifies
  rollouts or run directories.

## Framework adapters

Adapters add ground truth from run directories the transcript points to. Built-in: `cell_queue` (enabled by the
`cell-queue` profile, detected from `<controller>.py ... ack_visual` or `cell --file` commands) matches each
controller call to its session command and adds native action ranges, primitive calls, the observation state the
action was based on, visual acknowledgements, timestamped event logs and the episode result. Missing run
directories leave steps transcript-only. Writing a new adapter: [reference/profiles.md](reference/profiles.md).

## Limitations

- Argument values computed at runtime stay symbolic (`{"expr": "approach"}`); only literals, arithmetic on
  literals and earlier assignments are resolved.
- Reasoning is what Codex stored: reasoning summaries and messages, not hidden chain of thought.
- Steps follow tool calls, so control loops inside one script are one step; their calls are flagged `in_loop`.

## Additional resources

- Field-by-field description of `steps.jsonl`, `episode.json`, `timeline.jsonl`, `manifest.json`, the SFT
  output and the chat output: [reference/schema.md](reference/schema.md)
- Profile format, rule matching, examples for ROS 2 / MCP / SDK scripts, adapters:
  [reference/profiles.md](reference/profiles.md)
