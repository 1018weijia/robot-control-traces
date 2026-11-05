---
name: robot-control-traces
description: Converts Codex conversation logs (rollout JSONL under ~/.codex/sessions) of AI-driven robot, simulator or embodied-agent control into a step-level dataset of observation, analysis, action (calls and parameter adjustments) and outcome, including the camera images the model saw. Works for any control framework through JSON profiles; file-queue (cell-queue) sessions also get ground truth from their run directories. Use when the user wants to collect, export, organize, clean or share robot control sessions, agent control traces, interaction or action logs, or Codex chat history as training or evaluation data.
---

# Robot control traces

`scripts/rct.py` (Python 3.9+, standard library only) reads Codex rollouts and writes one folder per episode:
`steps.jsonl` (the dataset), `episode.md` (the same steps as a readable report with images), `timeline.jsonl`
(every event in order), `episode.json` and `images/`. A complete synthetic example, input and output, is in the
repository's `examples/` directory.

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
keyword heuristics. Unmatched calls are `aux`.

## Workflow

Copy this checklist and track progress:

```
- [ ] 1. scan: find rollouts that controlled something
- [ ] 2. scan --detail on 1-2 representative rollouts: check roles
- [ ] 3. if roles are wrong: write a profile, re-check with --profile
- [ ] 4. export
- [ ] 5. validate and spot-check 2-3 steps
- [ ] 6. report to the user
```

Below, `RCT` is `python3 <this skill directory>/scripts/rct.py`.

**1. Scan.** `RCT scan` (add `--since 2026-01-01`, `--codex-home DIR`, `--all` to include rollouts without act
calls). Each line shows act/observe counts, images, models and the most common act signature.

**2. Check roles.** `RCT scan --detail <rollout.jsonl>` prints every command/tool signature with its role, the rule
that assigned it and an example. Judge it against what the agent was doing:

- the calls that move the robot or send commands must be `act`; camera/state reads should be `observe`;
- `act 0` means the framework is unknown: write a profile;
- file edits, builds or log reads marked `act` mean the default heuristics are too broad: override them;
- if the agent runs control code from files, check that `code_file_options` covers the option naming the file.

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
precedence; `default` still classifies the rest.

**4. Export.**

```bash
RCT export --auto --out traces/               # every rollout scan lists (filtered by --since)
RCT export --rollout a.jsonl b.jsonl --out traces/ --profile mybot.json
```

Useful options: `--images seen|all|none` (default `seen`: images the model received; `all` also copies adapter
run-directory frames), `--redact-path /data/acme=<data>`, `--max-chars 4000` (output clip), `--no-adapters`,
`--read-local-files` (read code files missing from the transcript from disk; their current content may differ),
`--remap-path RECORDED=LOCAL` (run directories copied or moved since the session), `--min-act-steps N`,
`--overwrite` (only replaces a previous export).

**5. Validate.** Export runs validation automatically; `RCT validate traces/` re-runs it. It checks the schema,
step order, adjustment references and image hashes, then prints coverage and sample steps. Then open the
`episode.md` of 1-2 episodes (or read steps in `steps.jsonl`) and confirm:

- `analysis` contains the reasoning that led to the action and `observation.images` the frames it was based on;
- `action.calls` lists the control calls, and `action.adjustments` shows sensible deltas;
- no names, hosts, tokens or private paths remain (search the export for the user's name and project paths).

Low `with_images` usually means the model rarely viewed images; low `with_code` means control code was in files the
transcript never showed (try `--read-local-files`); `enriched 0%` for a cell-queue run means the run directory was
not found (moved: use `--remap-path`).

**6. Report** the output path, episode and step counts, coverage, profile used and anything that stayed unmatched.

## Privacy rules

- Never read, copy or print `auth.json` or other credentials in `$CODEX_HOME`. The script only reads
  `sessions/` and `archived_sessions/`.
- Keep redaction on (default): home paths become `~`, emails `<email>`, API keys / tokens / passwords
  `<redacted>` (or `<redacted:kind>`). Add `--redact-path OLD=NEW` for project or customer names. `--no-redact` is for private use only.
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

- Field-by-field description of `steps.jsonl`, `episode.json`, `timeline.jsonl` and `manifest.json`:
  [reference/schema.md](reference/schema.md)
- Profile format, rule matching, examples for ROS 2 / MCP / SDK scripts, adapters:
  [reference/profiles.md](reference/profiles.md)
