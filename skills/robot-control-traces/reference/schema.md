# Export schema (`robot-control-trace/v1`)

```
OUT/
  manifest.json                      export options, sources, episode list, redaction counts
  episodes/<date>_<thread8>_<NN>/
    episode.json                     source, agent, task, profile, stats, final turn, adapter info
    episode.md                       readable report of the steps (observation, analysis, action, outcome)
    steps.jsonl                      one JSON object per act call (the dataset)
    timeline.jsonl                   every normalized event of the episode, in order
    images/<sha16>.png               images the model received; session_<sha16>.png with --images all
```

An episode is one rollout, or one run directory inside a rollout when the profile's `episode_key` or an adapter
splits it (e.g. several `--session` directories driven from one Codex thread). `<NN>` numbers episodes of the same
rollout. Timestamps are ISO 8601 UTC strings from the rollout. All text is redacted unless `--no-redact`.

## steps.jsonl

| field | content |
|-------|---------|
| `schema`, `episode_id`, `step_index` | schema tag, episode folder name, 0-based index within the episode |
| `time.start / action / end` | first event of the step, the act call, end of its output |
| `turn` | Codex turn (user prompt) index |
| `observation` | what the model had seen since the previous act (below) |
| `analysis` | what the model said or wrote about it (below) |
| `action` | the act call and the control calls in it (below) |
| `outcome` | what came back (below) |
| `context.aux` | auxiliary calls in between: `tool`, `signature`, `exit_code`, `command` (clipped) |
| `context.notes` | context compaction or aborted turns that happened inside the step |
| `flags.human_input` | a user message arrived since the previous act (correction or new instruction) |
| `flags.action_failed` | exit code was non-zero |
| `flags.control` | the act is evidence of robot control: a non-default profile rule, a robot MCP server / ROS / robot CLI rule, or code with extracted control calls (false for generic `default` matches) |
| `source` | `call_id` of the act call, `seq_start` / `seq_action` into `timeline.jsonl` |
| `enrichment.<adapter>` | adapter ground truth (see cell_queue below), `{}` otherwise |

### observation

- `images[]`: image references, in the order the model received them:
  `id` (sha256 of the bytes), `file` (relative path in the episode, or null), `source` (`tool_output` or `user`),
  `via` (the signature of the call that returned it, or `user`), `path_hint` (path passed to `view_image`, redacted),
  `camera` (file stem when it looks like a camera name), `mime`, `available` (bytes were inline in the rollout).
- `reads[]`: observe calls: `tool`, `signature`, `rule`, `command` or `args`, `output` (clipped), `exit_code`,
  `images` (ids).
- `prior_feedback`: `{step, excerpt}` from the previous step's output (the feedback the model reacted to).
- `state`: structured state from an adapter (robot state, native action index, …), else null.

### analysis

- `reasoning[]`: reasoning summaries Codex stored for this span.
- `messages[]`: assistant messages (commentary) in this span.
- `rationale[]`: `{source: observe|act, signature, key, text}` from rationale options (`--note`, `--evidence`, …)
  or rationale keys in MCP arguments (`reason`, `intent`, …).
- `code_comments[]`: comments in the executed Python code.
- `user_messages[]`: user text in this span (the task prompt for step 0, corrections later).

### action

- `tool`: `exec_command`, `shell`, MCP `server__tool`, …; `signature`: stable label such as
  `python robot_ctl.py cell`, `ros2 topic pub` or `python -` (quoted values, option values, redirections and file
  writes removed); `rule`: profile rule that made it an act.
- `command`: full shell script (clipped at 20 000 chars) or null; `args`: tool arguments for non-shell calls.
- `code`, `code_source`: executed code and where it came from: `heredoc_stdin` (`python - <<EOF`), `python_-c`,
  `<writer>:<path>` for a file written earlier in the transcript (`heredoc` for `cat`/`tee` heredocs,
  `apply_patch_add` for a new file, `apply_patch` for an update applied to known content; an update to unknown
  content leaves the code unknown), `local_file_now:<path>` (`--read-local-files`), `session:commands.jsonl` (adapter).
- `calls[]`: control calls matching the profile's `action_calls`:
  `{func, line, args[], kwargs{}, in_loop?, in_def?}`. Literals, arithmetic on literals, `np.array` / `np.radians`
  and names assigned earlier are resolved (with `persistent_namespace`, across steps); anything else is
  `{"expr": "<source>"}`. MCP acts become one pseudo-call `{func: tool, kwargs: flattened arguments}` (rationale keys such as `reason` excluded); CLI acts one
  pseudo-call with numeric options as kwargs.
- `probes[]`: same format for `probe_calls` (state queries, IK checks, projections) that are not action calls.
- `adjustments[]`: for each call, the difference to the most recent earlier call with the same key (function plus
  short selector strings such as `arm="right"`):
  `{func, key, previous_step, line, changes{arg: change}}` with change one of
  `{from, to, delta}` (numbers), `{from, to, delta[], norm}` (vectors), `{from, to, items}` (other lists),
  `{from, to, symbolic: true}` (an `expr` on either side) or `{from, to}`. Free text and `*_id` arguments are skipped.

### outcome

`output` (ANSI codes and carriage-return progress removed, clipped to `--max-chars` keeping head and tail),
`output_chars` and `output_sha256` (of the raw output), `exit_code`, `duration_s`, `images[]` (returned by the act
call), `results[]` (`result`-role calls such as polling, same format as `observation.reads`).

### enrichment.cell_queue

`session`, `command_id`, `op`, `note`, `match` (`command_id`, `observation_index`, `note_and_time`, `note_only`),
`native_actions{before, after, count}`, `response` (error, native_actions, episode_ended), `primitives[]` (the
framework's primitive call records), `events` (rows with a numeric `time` from any other `*.jsonl` in the session
directory, by file name, inside the command's time window), `acks[]` (visual acknowledgements: `observation_index`,
`image_sha256`, `evidence`, `accepted`), `commands[]` (every matched controller invocation in the step),
`session_images[]` (`camera`, recorded `path`, `sha256`, `seen_by_model`, plus `file` with `--images all`).
Paths are the recorded ones; `--remap-path` only changes where the exporter looks for them.

## episode.json

- `schema`, `episode_id`.
- `source`: `rollout_file`, `rollout_sha256`, `thread_id`, `parent_thread_id`, `originator`, `cli_version`,
  `session_source`, `started_at`, `cwd`, `episode_key`. System prompts and git remotes are never exported.
- `agent`: `models[]`, `efforts[]` from turn contexts.
- `task`: `prompt` (first user message), `instruction` (adapter, e.g. the environment's task text).
- `profile`: `names[]`, `adapters[]`.
- `stats`: `steps`, `images_seen`, `images_saved`, coverage fractions (`with_images`, `with_observation`,
  `with_reasoning`, `with_analysis_text`, `with_code`, `with_calls`, `with_adjustments`, `enriched`),
  `human_input_steps`, `failed_actions`, `control_steps`.
- `final`: analysis after the last act (`messages`, `reasoning`, `user_messages`, `observation`); last episode of a
  rollout only.
- `enrichment.<adapter>`: per-episode adapter data (cell_queue: `session`, `run_config`, `result` with success
  labels). `adapter_stats`: match counts. `warnings`: reader warnings (truncated lines, …).

## timeline.jsonl

One normalized event per line, null fields dropped: `seq`, `ts`, `turn`, `kind` (`user_message`,
`assistant_message`, `reasoning`, `tool_call`, `image`, `compaction`, `turn_aborted`), `role`, `rule`, `tool`,
`signature`, `text`, `phase`, `command` or `args`, `code_source`, `output`, `exit_code`, `images`,
`continuation_of` (a `write_stdin` call continuing an earlier command). Use it to re-segment differently.

## manifest.json

`schema` (`robot-control-trace-manifest/v1`), `generator{name, version}`, `created_at`, `options` (redacted; custom
redaction and remap paths are only counted), `sources[]` (`rollout_file`, `rollout_sha256`, `lines`, `warnings`, `profiles`, `episodes[]`, or
`skipped` with the reason, e.g. too few act steps or no robot-control evidence), `episodes[]` (`id` = folder name,
`steps`, `rollout`), `redaction{enabled, counts, custom_paths}`. Redaction count kinds: `home_path`, `volume_path`,
`private_ip`, `email`, `custom_path`, `secret_field` (a JSON field such as `"api_key"`) and one per secret pattern
(`assignment`, `bearer`, `openai_key`, …). Token counts (`max_tokens`) and placeholders (`$API_KEY`, `...`,
`args.api_key`) are not treated as secrets.

## episode.md

The steps of the episode rendered for people: task prompt and result, then per step the images the model saw
(linked from `images/`), state and reads, reasoning and rationale, the executed code or arguments, control calls,
adjustments against earlier steps, and the outcome (JSON output shown as `key: value` lines). It is rendered from
the redacted records, so it is safe to review instead of the JSONL before sharing.

## SFT output (`robot-control-sft/v1`, written by `rct.py sft`)

```
SFT/
  train.jsonl          one sample per act step
  dataset_info.json    LLaMA-Factory dataset entry "robot_control_sft" (sharegpt; columns messages, images)
  sft_manifest.json    schema, created_at, source, options, counts, skipped_steps, image_files, redaction
  images/<episode_id>/<file>   observation frames referenced by samples (hard links when possible)
```

A `train.jsonl` row:

- `id`: `<episode_id>#<step_index>`.
- `messages`: `[system, user, assistant]`. The user turn holds `Task:`, `Earlier steps:` (one line per step for the
  last `--history` steps: signature, control calls, why, ok/failed and result), `Result of your last action`,
  `User:` messages of this span (not repeated when equal to the task), and `Current observation:` with one
  `<camera or image N>: <image>` line per image (latest `--max-images`), reads and adapter state. The assistant
  turn is the reasoning and messages of the span, then the action: the executed file
  (`` File `path`: ```python``` ``) when the code is not inline, then the command in a `bash` block, or the
  tool and its arguments in a `json` block. A literal `<image>` in recorded text is written as `< image>`.
- `images`: paths relative to the SFT directory, one per `<image>` placeholder, in order.
- `meta`: `schema`, `episode_id`, `step_index`, `steps_in_episode`, `task_success` (adapter label or null),
  `action_failed`, `human_input`, `control`, `signature`, `has_analysis`, `prompt_chars`, `target_chars`.

`skipped_steps` counts steps left out and why: `no_control_evidence` (unless `--keep-no-control`),
`action_failed` (`--drop-failed`), `episode_not_successful` (`--success-only`), `action_clipped` (the command was
cut at export, so the target would be incomplete). Text is redacted again when samples are built.

## Loading

```python
import json, pathlib
for ep in sorted(pathlib.Path("traces/episodes").iterdir()):
    for line in (ep / "steps.jsonl").open():
        step = json.loads(line)
        frames = [ep / i["file"] for i in step["observation"]["images"] if i["file"]]
        text = step["analysis"]["reasoning"] + [r["text"] for r in step["analysis"]["rationale"]]
        action = step["action"]["code"] or step["action"]["command"] or step["action"]["args"]
```
