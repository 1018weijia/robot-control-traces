"""Export directory → step-level SFT samples for vision-language robot agents.

One sample per act step. The prompt carries the task, a one-line summary of the previous steps, the full result of
the previous action and the current observation (images as ``<image>`` placeholders, text reads, state, user
messages). The target is the agent's own analysis followed by the exact action it issued: the shell command and
the code it ran, or the tool name and its JSON arguments. Nothing is synthesised, and each piece of feedback
appears once. Rows follow the LLaMA-Factory sharegpt layout (``messages`` + ``images``) and ``dataset_info.json``
registers them; ``meta`` carries labels for filtering (task success, failed action, human input, control).
"""
from __future__ import annotations

import collections
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from .redact import Redactor
from .steps import clip, written_texts

SCHEMA = "robot-control-sft/v1"
DATASET_NAME = "robot_control_sft"
CUT_MARK = " chars cut]…"
SYSTEM = ("You control a robot through tools. Each turn shows the task, a summary of your earlier steps, the result "
          "of your last action and the current observation. Describe what you observe and what you conclude, then "
          "issue the next action exactly as the tool expects it.")
STATE_SKIP = {"images", "image_sha256", "image_paths"}
PLACEHOLDER = "<image>"
_SLOT = "\x00image\x00"


@dataclasses.dataclass
class SftOptions:
    history: int = 4            # earlier steps summarised in each prompt
    max_images: int = 4         # most recent observation images per sample
    read_chars: int = 1500      # per observation read and for the previous result
    task_chars: int = 4000
    success_only: bool = False  # only episodes whose recorded result says the task succeeded
    drop_failed: bool = False   # do not use failed actions as targets (they stay in later prompts' history)
    control_only: bool = True   # only act steps with robot-control evidence (flags.control)
    redact: bool = True         # re-apply the current redaction rules (fixes exports made by older versions)


def episode_success(episode):
    """The task outcome an adapter recorded (result.task_success / success), or None when unknown."""
    for info in (episode.get("enrichment") or {}).values():
        result = info.get("result") if isinstance(info, dict) else None
        if isinstance(result, dict):
            for key in ("task_success", "success", "succeeded"):
                if isinstance(result.get(key), bool):
                    return result[key]
    return None


def _drop_noise(value):
    if isinstance(value, dict):
        return {k: _drop_noise(v) for k, v in value.items() if k not in STATE_SKIP}
    if isinstance(value, list):
        return [_drop_noise(v) for v in value]
    return value


def _compact(text, limit):
    """Tool output for a prompt: JSON on one line without image paths and hashes, anything else as recorded."""
    text = (text or "").strip()
    if text[:1] in "{[":
        try:
            text = json.dumps(_drop_noise(json.loads(text)), ensure_ascii=False)
        except ValueError:
            pass
    return clip(text, limit)


def _first_line(texts, limit=160):
    for text in texts:
        for line in _compact(text, 0).splitlines():
            line = line.strip()
            if line:
                return line if len(line) <= limit else line[:limit].rstrip() + "…"
    return ""


def _analysis_texts(step):
    """What the agent wrote once the previous action's result was in, in written order."""
    return written_texts(step["analysis"])[step["analysis"].get("waiting", 0):]


def _summary(step):
    action, outcome = step["action"], step["outcome"]
    calls = ", ".join(c["func"] for c in action["calls"][:4])
    why = _first_line(_analysis_texts(step) + [r["text"] for r in step["analysis"]["rationale"]])
    status = "failed" if step["flags"].get("action_failed") else "ok"
    result = _first_line([outcome.get("output")] + [r.get("output") for r in outcome["results"]], 120)
    line = f"- step {step['step_index']}: {action['signature']}" + (f" ({calls})" if calls else "")
    return line + (f" — {why}" if why else "") + f" → {status}" + (f": {result}" if result else "")


def _result_text(step, limit):
    outcome = step["outcome"]
    parts = []
    if outcome.get("exit_code") not in (None, 0):
        parts.append(f"exit code {outcome['exit_code']}")
    if outcome.get("output"):
        parts.append(_compact(outcome["output"], limit))
    for result in outcome["results"]:
        if result.get("output"):
            parts.append(f"result from `{result['signature']}`: " + _compact(result["output"], limit))
    return "\n".join(parts) or "(no output)"


def _state_text(state, limit):
    if not isinstance(state, dict) or not state:
        return None
    kept = _drop_noise(state)
    return clip(json.dumps(kept, ensure_ascii=False), limit) if kept else None


def _action_text(action):
    """The action exactly as issued; code that a command runs from a file is shown first when it is not inline."""
    parts = []
    command = action.get("command")
    code = action.get("code")
    if code and code.strip() not in (command or ""):
        source = (action.get("code_source") or "").split(":", 1)[-1] or "code"
        parts.append(f"File `{source}`:\n```python\n{code.rstrip()}\n```")
    if command is not None:
        parts.append(f"Action (`{action['tool']}`):\n```bash\n{command.rstrip()}\n```")
    else:
        args = json.dumps(action.get("args"), ensure_ascii=False)
        parts.append(f"Action (`{action['tool']}`):\n```json\n{args}\n```")
    return "\n\n".join(parts)


class _Images:
    def __init__(self, export_dir, out):
        self.export_dir, self.out = export_dir, out
        self.copied = 0

    def place(self, episode_id, ref):
        source = self.export_dir / "episodes" / episode_id / ref["file"]
        if not source.is_file():
            return None
        relative = Path("images") / episode_id / Path(ref["file"]).name
        target = self.out / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
            self.copied += 1
        return relative.as_posix()


def _sample(episode, steps, index, images, options, redactor):
    step = steps[index]
    episode_id = episode["episode_id"]
    task = episode.get("task") or {}
    task_text = task.get("instruction") or task.get("prompt") or ""
    lines = [f"Task: {clip(task_text, options.task_chars)}"]
    if index:
        shown = steps[max(0, index - options.history):index]
        lines += ["", f"Earlier steps ({index} so far, last {len(shown)} shown):"] + [_summary(s) for s in shown]
        lines += ["", f"Result of your last action (step {index - 1}):", _result_text(steps[index - 1],
                                                                                     options.read_chars)]
    for message in step["analysis"]["user_messages"]:
        if message.strip() != task_text.strip():
            lines += ["", f"User: {clip(message, options.read_chars)}"]
    observation = step["observation"]
    files = []
    refs = [r for r in observation["images"] if r.get("file")][-options.max_images:] if options.max_images else []
    for ref in refs:
        path = images.place(episode_id, ref)
        if path:
            files.append((ref.get("camera") or f"image {len(files) + 1}", path))
    reads = [r for r in observation["reads"] if (r.get("output") or "").strip()]
    state = _state_text(observation.get("state"), options.read_chars)
    if files or reads or state:
        lines += ["", "Current observation:"]
        lines += [f"{camera}: {_SLOT}" for camera, _ in files]
        lines += [f"- `{r['signature']}`: {_compact(r['output'], options.read_chars)}" for r in reads]
        if state:
            lines.append(f"- state: {state}")
    lines += ["", "Decide the next action."]
    analysis = "\n".join(t.strip() for t in _analysis_texts(step) if t and t.strip())
    target = (analysis + "\n\n" if analysis else "") + _action_text(step["action"])
    # A literal "<image>" in recorded text would be read as an extra image slot by the trainer.
    prompt = redactor.text("\n".join(lines)).replace(PLACEHOLDER, "< image>").replace(_SLOT, PLACEHOLDER)
    target = redactor.text(target).replace(PLACEHOLDER, "< image>")
    success = episode_success(episode)
    return {
        "id": f"{episode_id}#{step['step_index']}",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                     {"role": "assistant", "content": target}],
        "images": [path for _, path in files],
        "meta": {"schema": SCHEMA, "episode_id": episode_id, "step_index": step["step_index"],
                 "steps_in_episode": len(steps), "task_success": success,
                 "action_failed": bool(step["flags"].get("action_failed")),
                 "human_input": bool(step["flags"].get("human_input")),
                 "control": bool(step["flags"].get("control")), "signature": step["action"]["signature"],
                 "has_analysis": bool(analysis), "prompt_chars": len(prompt), "target_chars": len(target)},
    }


def build(export_dir, out, options: SftOptions, overwrite=False):
    """Write <out>/train.jsonl, images/, dataset_info.json and sft_manifest.json; returns the manifest."""
    export_dir, out = Path(export_dir), Path(out)
    manifest_in = json.loads((export_dir / "manifest.json").read_text())
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} is not empty; choose a new directory or pass --overwrite")
        if not (out / "sft_manifest.json").exists():
            raise SystemExit(f"refusing to overwrite {out}: it does not look like a previous sft export")
        shutil.rmtree(out / "images", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    redactor = Redactor(options.redact)
    images = _Images(export_dir, out)
    skipped = collections.Counter()
    counts = collections.Counter()
    with open(out / "train.jsonl", "w", encoding="utf-8") as handle:
        for entry in manifest_in.get("episodes", []):
            folder = export_dir / "episodes" / entry["id"]
            episode = json.loads((folder / "episode.json").read_text())
            steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
            if options.success_only and episode_success(episode) is not True:
                skipped["episode_not_successful"] += len(steps)
                continue
            counts["episodes"] += 1
            for index, step in enumerate(steps):
                if options.control_only and step["flags"].get("control") is False:
                    skipped["no_control_evidence"] += 1
                    continue
                if options.drop_failed and step["flags"].get("action_failed"):
                    skipped["action_failed"] += 1
                    continue
                if CUT_MARK in (step["action"].get("command") or ""):
                    skipped["action_clipped"] += 1
                    continue
                row = _sample(episode, steps, index, images, options, redactor)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["samples"] += 1
                counts["images"] += len(row["images"])
                counts["with_analysis"] += row["meta"]["has_analysis"]
    info = {DATASET_NAME: {"file_name": "train.jsonl", "formatting": "sharegpt",
                           "columns": {"messages": "messages", "images": "images"},
                           "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                                    "assistant_tag": "assistant", "system_tag": "system"}}}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": {"generator": manifest_in.get("generator"), "episodes": len(manifest_in.get("episodes", []))},
                "options": dataclasses.asdict(options), "counts": dict(counts), "skipped_steps": dict(skipped),
                "image_files": images.copied, "redaction": redactor.counts}
    (out / "sft_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def check(out):
    """Errors in an sft directory: placeholder/image count mismatches and missing image files."""
    out = Path(out)
    errors = []
    for number, line in enumerate((out / "train.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        placeholders = sum(m["content"].count(PLACEHOLDER) for m in row["messages"])
        if placeholders != len(row["images"]):
            errors.append(f"train.jsonl:{number}: {placeholders} <image> placeholders, {len(row['images'])} images")
        errors += [f"train.jsonl:{number}: missing {p}" for p in row["images"] if not (out / p).is_file()]
        if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
            errors.append(f"train.jsonl:{number}: unexpected roles")
    return errors
