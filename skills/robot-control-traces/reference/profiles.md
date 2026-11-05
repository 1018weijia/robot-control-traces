# Profiles and adapters

A profile is a JSON file that tells the exporter which tool calls of a control framework act, observe, deliver
results or are auxiliary, and how to read rationale and control calls out of them. Built-in profiles live in
`profiles/`; user profiles can live anywhere and are passed with `--profile path.json` (repeatable).

## Fields

| field | type | meaning |
|-------|------|---------|
| `name` | string, required | identifier, shown in `episode.json` and rule origins (`name:label`) |
| `description` | string | shown by `rct.py profiles` |
| `detect` | regex list | auto-select this built-in profile when a command or tool name in the rollout matches |
| `rules` | list | role rules, see below |
| `rationale_options` | list | shell options whose values are the agent's stated reasons (`--note`, `--why`) |
| `rationale_keys` | list | argument keys of MCP / builtin tools with the same meaning (`reason`, `intent`) |
| `code_file_options` | list | options that name a code file the command executes (`--file`, `--script`) |
| `action_calls` | regex | fully-qualified function names that are control calls (`^bot\.(move\w*\|grip)$`) |
| `probe_calls` | regex | function names that query state without acting (`^bot\.(state\|ik)$`) |
| `episode_key` | regex | first group splits a rollout into episodes (`--(?:session\|output)[= ]+(\S+)`) |
| `persistent_namespace` | bool | values assigned in one step's code stay known in later steps (Jupyter-like cells) |
| `adapters` | list | adapter modules to run (`cell_queue`) |

Regexes are Python `re` syntax inside JSON strings, so backslashes are doubled: `"\\bros2 topic pub\\b"`.

## Rules

```json
{"label": "move", "role": "act", "kind": "shell", "command": "\\bmybot_ctl\\b[^\\n]*\\smove\\b"}
```

- `role`: `act`, `observe`, `result`, `aux` or `ignore` (dropped from steps, kept in the timeline).
- `kind`: `shell` (calls with a command: `exec_command`, `shell`, `local_shell`, …), `mcp` (tool names containing
  `__` or `.`, e.g. `robot__move_arm`), `builtin` (other Codex tools such as `view_image`, `apply_patch`).
- Match fields, all optional but at least one required; every given field must match (`re.search`):
  - `tool`: tool name;
  - `command`: the shell script's **skeleton**: heredoc bodies removed and every quoted string replaced by `'…'`,
    so a note such as `--note 'move closer'` never triggers a rule. Set `"raw": true` to match the full script;
  - `code`: the executed code (heredoc, `python -c`, or a file written earlier in the transcript);
  - `args`: the JSON-encoded arguments of non-shell tools.
- `label`: name used in `rule` fields of the export (defaults to the rule index).

Rules are tried in order: explicit profiles (in `--profile` order), auto-detected built-ins, then `default`. The
first match wins; calls nothing matches become `aux` (`rule: "unmatched"`). To stop a default heuristic from firing,
add an earlier rule for the same calls with role `aux`.

When several profiles apply, list fields are merged, `action_calls` / `probe_calls` are OR-ed, the first
`episode_key` wins and `persistent_namespace` is on if any profile sets it.

`write_stdin` calls that continue a running command are handled without rules: they deliver results of an `act`
and are ignored otherwise.

## The default profile and control evidence

`default` is a fallback for frameworks without a profile. Its rules, in order:

| label | role | matches |
|-------|------|---------|
| `view_image` | observe | Codex `view_image` |
| `codex_builtin` | aux | Codex tools such as `apply_patch`, `update_plan`, `send_message`, `wait_agent` |
| `agent_tools` | aux | sub-agent and Codex app servers: `collaboration.*`, `multi_agent*.*`, `mcp__codex*` |
| `mcp_observe` | observe | MCP camera, image, screenshot, state and observation tools |
| `robot_server_act` | act | a tool of a server named like a robot or simulator (`robot`, `arm`, `ros2`, `isaac_sim`, `mujoco`, `xarm`, `lerobot`, …) whose name contains a control verb (`execute`, `move`, `grasp`, `step`, `set`, `publish`, `finish`, …) |
| `robot_server_other` | observe | the remaining tools of such a server (`task_demos`, `zoom_head`) |
| `mcp_act` | act | other MCP tools whose name ends in robot vocabulary (`set_gripper`, `move_joints`, `pick_and_place`) |
| `act_code`, `act_command` | act | Python calling `robot.move…` / `env.step` and similar; ROS publish / service / action commands |
| `observe_code`, `observe_command` | observe | Python or commands reading observations, state, cameras, images |

A step's `flags.control` is true when the rule that made it an act comes from a non-default profile, or is
`robot_server_act`, `mcp_act` or `act_command`, or is `act_code` with extracted `action.calls`. Export skips rollouts
with fewer than `--min-control-steps` (default 1) such steps, so coding, research and agent-orchestration sessions
that only look like control to keyword rules stay out of the dataset. `rct.py sft` also leaves out act steps
without the flag unless `--keep-no-control`. If a real robot framework is skipped, write a profile for it: any
explicit profile rule counts as control evidence.

## Developing a profile

1. `rct.py scan --detail ROLLOUT` and note the signatures that should be `act` / `observe`.
2. Write rules for those signatures; anchor subcommands with `\\btool\\b[^\\n]*\\s(sub)\\b` so paths and option
   values do not match.
3. Re-run `rct.py scan --detail ROLLOUT --profile my.json` until every control call has the right role.
4. Export one rollout and read `action.calls`; widen `action_calls` if control calls are missing.

## Examples

**ROS 2 command line.** The agent publishes commands and echoes topics:

```json
{"name": "ros2-cli",
 "rules": [
   {"label": "pub", "role": "act", "kind": "shell", "command": "\\bros2 (topic pub|action send_goal|service call)\\b"},
   {"label": "echo", "role": "observe", "kind": "shell", "command": "\\bros2 topic echo\\b"},
   {"label": "list", "role": "aux", "kind": "shell", "command": "\\bros2 (topic|node|service) list\\b"}]}
```

Numeric options of an act command (`--rate 10`) become kwargs of a pseudo-call and are diffed between steps; message
payloads stay in `action.command`.

**MCP robot server.** Tools `robot__move_arm`, `robot__get_camera`, `robot__plan`:

```json
{"name": "robot-mcp",
 "rules": [
   {"label": "plan", "role": "aux", "kind": "mcp", "tool": "^robot__plan$"},
   {"label": "camera", "role": "observe", "kind": "mcp", "tool": "^robot__(get_camera|get_state)$"},
   {"label": "act", "role": "act", "kind": "mcp", "tool": "^robot__"}],
 "rationale_keys": ["reason"]}
```

The arguments of each act become one pseudo-call; nested objects are flattened (`target.x`), so a change from
`{"target": {"x": 0.30}}` to `{"target": {"x": 0.32}}` is the adjustment `target.x: +0.02`.

**Python SDK scripts.** The agent writes `step.py` with `apply_patch` and runs `python step.py --why '...'`:

```json
{"name": "mysdk",
 "rules": [{"label": "run", "role": "act", "kind": "shell", "code": "\\barm\\.(move_\\w+|gripper)\\("},
           {"label": "look", "role": "observe", "kind": "shell", "command": "\\bpython\\S* \\S*snap\\.py\\b"}],
 "rationale_options": ["--why"],
 "action_calls": "^arm\\.(move_\\w+|gripper)$",
 "probe_calls": "^arm\\.(pose|ik)$"}
```

A `code` rule sees the file content because writes earlier in the transcript (heredoc, `tee`, `apply_patch` add and
update) are tracked per path.

## Built-in: cell-queue

For file-queue control: the agent drives a controller script (any `<name>.py`) whose subcommands submit Python
cells and visual acknowledgements to a runner, and the runner writes one session directory per episode. Detected
when a command runs `<name>.py ... ack_visual` or `<name>.py ... cell --file`.

| controller op | role | step content |
|---------------|------|--------------|
| `observe`, `ack_visual` | observe | `--note` and `--evidence` go to `analysis.rationale`; viewed frames to `observation.images` |
| `cell` / `python`, `finish` | act | cell code from `--file` (written earlier in the transcript) or from the session |
| `poll` | result | appended to the previous act's `outcome.results` |
| `where` | aux | |

`action_calls` matches `<object>.move_to / move_by / move_path / move_joints / set_gripper / home / wait` and
`*execute*` / `*_step` methods; `probe_calls` matches state, reachability, projection, annotation, proposal and
diagnosis helpers. Cells share one namespace, so constants defined in an earlier cell resolve later. Episodes are
split by `--session` / `--output`. The `cell_queue` adapter reads the session directory (see
[schema.md](schema.md), `enrichment.cell_queue`); `examples/` in the repository contains a complete session.

## Adapters

An adapter adds ground truth from files a framework writes next to the transcript. It is a module in
`scripts/rct_lib/adapters/` registered in `REGISTRY` of `adapters/__init__.py` and enabled by a profile's `adapters`
list. Interface:

```python
NAME = "myframework"

def enrich(steps, rollout, options) -> dict:
    """steps: transcript-derived step dicts (step["_act"] is the act event; step["_observe"] and
    step["_results"] are the observe/result events, each with .command, .args, .output, .ts, .workdir).
    May set step["enrichment"][NAME], step["observation"]["state"], a missing step["action"]["code"] and
    step["_episode_hint"] (a key that groups steps into episodes).
    Returns {"episodes": {hint: per-episode info}, "stats": {...}}."""
```

Rules for adapters:

- read only; never write into run directories or command queues;
- honour `options.remap_paths` when resolving recorded paths, and report the recorded paths, not local ones;
- never raise for missing or partial data: count it in `stats` and leave the step transcript-only
  (exceptions are caught and reported in `adapter_stats`, but leave that rollout partly enriched);
- match transcript calls to framework records by explicit ids first, then by content, then by nearest time;
  record the method used so dataset users can filter;
- keep large arrays compact (`cell_queue.compact`) and put absolute paths only in fields the redactor sees
  (everything in the step is redacted on write);
- add a fixture-based test in `tests/` that builds both a rollout and a run directory.
