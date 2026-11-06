"""Turn a classified rollout timeline into control steps: observation → analysis → action → outcome.

Every ``act`` tool call closes a step. Everything since the previous act belongs to that step: images and reads
from ``observe`` calls are its observation; reasoning summaries, agent messages, rationale arguments, user messages
and code comments are its analysis; the act call (with the code it actually ran) is its action; the act's output
and any later ``result`` calls (polls, stdin reads) are its outcome.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import re
import shlex

from . import calls as calls_mod
from . import shell
from .codex import typed_input
from .profile import tool_kind

SCHEMA = "robot-control-trace/v1"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CAMERA_RE = re.compile(r"(?i)(cam|camera|rgb|depth|wrist|head|front|side|top|left|right|overhead|ego)")


def _terminal(text):
    text = ANSI_RE.sub("", text)
    if "\r" in text:
        # Progress bars redraw a line with carriage returns; keep what a terminal would show.
        text = "\n".join(line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n"))
    return text


def _clean_strings(value):
    if isinstance(value, str):
        return _terminal(value)
    if isinstance(value, list):
        return [_clean_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_strings(v) for k, v in value.items()}
    return value


def clip(text, limit):
    if text is None:
        return None
    text = str(text)
    if text[:1] in "{[" and ("\\r" in text or "\\u001b" in text):
        try:
            text = json.dumps(_clean_strings(json.loads(text)), ensure_ascii=False, indent=2)
        except ValueError:
            pass
    text = _terminal(text)
    if limit and len(text) > limit:
        half = limit // 2
        return text[:half] + f"\n…[{len(text) - limit} chars cut]…\n" + text[-half:]
    return text


def _sha(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest() if text else None


# ---- preparation ------------------------------------------------------------------------------------------------

def apply_update(content, patch):
    """Apply an apply_patch ``Update File`` body to known content; None when a hunk does not match."""
    lines = content.rstrip("\n").split("\n")
    hunks, current = [], None
    for line in patch.rstrip("\n").split("\n"):
        if line.startswith("@@"):
            current = ([], [])
            hunks.append(current)
            continue
        if line.startswith("*** End of File"):
            continue
        if current is None:
            current = ([], [])
            hunks.append(current)
        tag, body = (line[:1], line[1:]) if line[:1] in " +-" else (" ", line)
        if tag in " -":
            current[0].append(body)
        if tag in " +":
            current[1].append(body)
    position = 0
    for old, new in hunks:
        if not old:
            lines[position:position] = new
            position += len(new)
            continue
        for strict in (True, False):
            norm = (lambda s: s) if strict else (lambda s: s.rstrip())
            found = next((i for i in range(position, len(lines) - len(old) + 1)
                          if all(norm(lines[i + j]) == norm(o) for j, o in enumerate(old))), None)
            if found is not None:
                lines[found:found + len(old)] = new
                position = found + len(new)
                break
        else:
            return None
    return "\n".join(lines) + "\n"


class FileState:
    """What the transcript says each file contains, as far as writes in it can tell."""

    def __init__(self, read_local=False):
        self.files = {}
        self.read_local = read_local

    def write(self, path, content, source, seq):
        if source == "apply_patch_update":
            known = self.files.get(path)
            updated = apply_update(known[0], content) if known else None
            if updated is None:
                self.files.pop(path, None)  # content unknown from here on; a diff is not the executed code
            else:
                self.files[path] = (updated, "apply_patch", seq)
            return
        self.files[path] = (content, source, seq)

    def lookup(self, path, workdir):
        full = shell.resolve_path(path, workdir)
        if full in self.files:
            return self.files[full]
        suffix = "/" + str(path).lstrip("./")
        matches = [(v[2], k) for k, v in self.files.items() if k.endswith(suffix)]
        if matches:
            return self.files[max(matches)[1]]
        if self.read_local and full and os.path.isfile(full) and os.path.getsize(full) < 2_000_000:
            try:
                with open(full, encoding="utf-8") as handle:
                    return handle.read(), "local_file_now", None
            except (OSError, UnicodeDecodeError):
                return None
        return None


def _writes(event):
    base = (event.tool or "").split(".")[-1]
    if event.command is not None:
        return shell.file_writes(event.command)
    if base == "apply_patch":
        text = event.args if isinstance(event.args, str) else (event.args or {}).get("input") \
            if isinstance(event.args, dict) else None
        return shell.patch_writes(text or "")
    return []


def prepare(rollout, profile, read_local=False):
    """Resolve skeleton, signature, code and role of every tool call in timeline order."""
    files = FileState(read_local)
    by_seq = {}
    for event in rollout.events:
        by_seq[event.seq] = event
        if event.kind != "tool_call":
            continue
        # A script usually writes its code file before running it, so writes are applied first.
        for path, content, source in _writes(event):
            files.write(shell.resolve_path(path, event.workdir), content, source, event.seq)
        if event.command is not None:
            event.skeleton = shell.skeleton(event.command)
            event.signature = shell.signature(event.command)
            code, source = shell.inline_code(event.command)
            if code is None:
                for path in shell.referenced_files(event.command, profile.code_file_options):
                    hit = files.lookup(path, event.workdir)
                    if hit:
                        code, source = hit[0], f"{hit[1]}:{path}"
                        break
            event.code, event.code_source = code, source
        else:
            event.signature = event.tool
        base = (event.tool or "").split(".")[-1]
        if base == "write_stdin" and event.continuation_of in by_seq:
            target = by_seq[event.continuation_of]
            if typed_input(event):
                event.code, event.code_source = event.args["chars"], "stdin"
                event.role, event.rule = _typed_role(event, target, by_seq, profile)
            else:
                event.role, event.rule = ("result" if target.role == "act" else "ignore"), "stdin_continuation"
        else:
            event.role, event.rule = profile.classify(event, tool_kind(event))


def _typed_role(event, target, by_seq, profile):
    """Typed input is classified like a command that runs it; input no rule knows acts when an act call started the
    process."""
    typed = dataclasses.replace(event, command=event.code, skeleton=shell.skeleton(event.code))
    role, rule = profile.classify(typed, "shell")
    if rule != "unmatched":
        return role, rule
    while target.continuation_of in by_seq:
        target = by_seq[target.continuation_of]
    return ("act", "default:stdin_input") if target.role == "act" else (role, rule)


# ---- segmentation -----------------------------------------------------------------------------------------------

class _Buffer:
    def __init__(self):
        self.events = []
        self.waiting = 0

    def of(self, *kinds, role=None):
        return [e for e in self.events if e.kind in kinds and (role is None or e.role == role)]

    def texts(self):
        return [e for e in self.of("reasoning", "assistant_message") if e.text]


def image_ref(rollout, key, via):
    image = rollout.images.get(key)
    hint = image.path_hint if image else None
    stem = os.path.splitext(os.path.basename(hint))[0] if hint else None
    return {"id": key, "source": image.source if image else None, "via": via, "path_hint": hint,
            "camera": stem if stem and CAMERA_RE.search(stem) else None, "mime": image.mime if image else None,
            "available": bool(image and image.data is not None), "file": None}


def call_record(event, limit):
    record = {"tool": event.tool, "signature": event.signature, "rule": event.rule, "exit_code": event.exit_code,
              "output": clip(event.output, limit), "images": list(event.images)}
    if event.command is not None:
        record["command"] = clip(event.command, limit)
    else:
        record["args"] = _clip_json(event.args, limit)
    return record


def _clip_json(value, limit):
    if isinstance(value, str):
        return clip(value, limit)
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= limit else clip(text, limit)


def rationale(events, source, profile):
    found = []
    keys = {k.lower() for k in profile.rationale_keys}
    for event in events:
        if event.command is not None:
            for option, values in shell.option_values(event.command, profile.rationale_options).items():
                for value in values:
                    found.append({"source": source, "signature": event.signature, "key": option, "text": value})
        elif isinstance(event.args, dict):
            for key, value in event.args.items():
                if key.lower() in keys and isinstance(value, str) and value.strip():
                    found.append({"source": source, "signature": event.signature, "key": key, "text": value})
    return found


def _texts(events):
    return [e.text for e in events if e.text]


def written_texts(analysis):
    """Reasoning and messages in the order the agent wrote them; reasoning first for exports without ``order``."""
    order = analysis.get("order") or (["reasoning"] * len(analysis["reasoning"])
                                      + ["messages"] * len(analysis["messages"]))
    lists = {"reasoning": iter(analysis["reasoning"]), "messages": iter(analysis["messages"])}
    return [next(lists[name]) for name in order]


def make_step(rollout, act, buffer, profile, limit, previous):
    observe = buffer.of("tool_call", role="observe")
    images = []
    for event in buffer.events:
        for key in event.images:
            via = event.signature if event.kind == "tool_call" else "user" if event.kind in ("image", "user_message") \
                else event.kind
            images.append(image_ref(rollout, key, via))
    first = next((e.ts for e in buffer.events if e.ts), act.ts)
    prior = None
    if previous is not None and previous["outcome"]["output"]:
        prior = {"step": previous["step_index"], "excerpt": clip(previous["outcome"]["output"], min(limit, 1200))}
    step = {
        "schema": SCHEMA, "episode_id": None, "step_index": None,
        "time": {"start": first, "action": act.ts, "end": act.end_ts},
        "turn": act.turn,
        "observation": {"images": images, "reads": [call_record(e, limit) for e in observe],
                        "prior_feedback": prior, "state": None},
        "analysis": {"reasoning": _texts(buffer.of("reasoning")),
                     "messages": _texts(buffer.of("assistant_message")),
                     "order": ["reasoning" if e.kind == "reasoning" else "messages" for e in buffer.texts()],
                     "waiting": buffer.waiting,
                     "rationale": rationale(observe, "observe", profile) + rationale([act], "act", profile),
                     "code_comments": [], "user_messages": _texts(buffer.of("user_message"))},
        "action": {"tool": act.tool, "signature": act.signature, "rule": act.rule,
                   "command": clip(act.command, max(limit, 20000)) if act.command is not None else None,
                   "args": _clip_json(act.args, limit) if act.command is None else None,
                   "code": act.code, "code_source": act.code_source, "calls": [], "probes": [], "adjustments": []},
        "outcome": {"output": clip(act.output, limit), "output_chars": len(act.output or ""),
                    "output_sha256": _sha(act.output), "exit_code": act.exit_code, "duration_s": act.duration_s,
                    "images": [image_ref(rollout, k, act.signature) for k in act.images], "results": []},
        "context": {"aux": [{"tool": e.tool, "signature": e.signature, "exit_code": e.exit_code,
                             "command": clip(e.command, 400) if e.command else None}
                            for e in buffer.of("tool_call", role="aux")],
                    "notes": [{"kind": e.kind, "text": clip(e.text, 600)}
                              for e in buffer.of("compaction", "turn_aborted")]},
        "flags": {"human_input": bool(buffer.of("user_message")) and previous is not None,
                  "action_failed": act.exit_code not in (None, 0)},
        "source": {"call_id": act.call_id, "seq_start": buffer.events[0].seq if buffer.events else act.seq,
                   "seq_action": act.seq},
        "enrichment": {},
        "_act": act, "_observe": observe,
    }
    return step


def segment(rollout, profile, limit=4000):
    steps, buffer = [], _Buffer()
    for event in rollout.events:
        if event.kind == "tool_call" and event.role == "act":
            steps.append(make_step(rollout, event, buffer, profile, limit, steps[-1] if steps else None))
            buffer = _Buffer()
        elif event.kind == "tool_call" and event.role == "result" and steps:
            record = call_record(event, limit)
            if event.rule == "stdin_continuation":
                record["output"] = None  # already appended to the output of the call it polled
            steps[-1]["outcome"]["results"].append(record)
            steps[-1]["outcome"]["images"] += [image_ref(rollout, k, event.signature) for k in event.images]
            steps[-1]["_results"] = steps[-1].get("_results", []) + [event]
            buffer.waiting = len(buffer.texts())
        elif event.kind == "tool_call" and event.role == "ignore":
            continue
        else:
            buffer.events.append(event)
    tail = buffer
    final = {"messages": _texts(tail.of("assistant_message")), "reasoning": _texts(tail.of("reasoning")),
             "user_messages": _texts(tail.of("user_message")),
             "observation": {"images": [image_ref(rollout, k, e.signature or e.kind) for e in tail.events
                                        for k in e.images],
                             "reads": [call_record(e, limit) for e in tail.of("tool_call", role="observe")]}}
    return steps, final


# ---- per-episode action analysis --------------------------------------------------------------------------------

def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: value} if prefix else {"value": value}


def _is_python(code, source):
    if not code or (source and source.endswith(".sh")):
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def annotate_actions(steps, profile, persistent_namespace=False):
    """Fill calls, probes, code comments and adjustments; values and history are scoped to the episode."""
    tracker = calls_mod.AdjustmentTracker()
    env = {}
    for step in steps:
        action = step["action"]
        act = step.get("_act")
        calls = []
        if _is_python(action["code"], action["code_source"]):
            scope = env if persistent_namespace else {}
            found, _ = calls_mod.extract(action["code"], profile.action_calls, dict(scope))
            probes, _ = calls_mod.extract(action["code"], profile.probe_calls, scope)
            calls = found
            action["probes"] = [p for p in probes if not profile.action_calls.search(p["func"])]
            step["analysis"]["code_comments"] = calls_mod.comments(action["code"])
        elif act is not None and act.command is None and isinstance(act.args, dict):
            keys = {k.lower() for k in profile.rationale_keys}
            params = {k: v for k, v in act.args.items() if k.lower() not in keys}  # already in analysis.rationale
            calls = [{"func": act.tool, "args": [], "kwargs": _flatten(params)}]
        elif act is not None and act.command is not None:
            for argv in shell.commands(act.command):
                params = shell.numeric_options(argv)
                if params:
                    calls.append({"func": shell.signature(shlex.join(argv)), "args": [], "kwargs": params})
        action["calls"] = calls
        action["adjustments"] = tracker.step(step["step_index"], calls)
        step["flags"]["control"] = control_evidence(action)
    return steps


ROBOT_SPECIFIC_DEFAULT_RULES = {"default:robot_server_act", "default:mcp_act", "default:act_command"}


def control_evidence(action):
    """True when the act step demonstrably commands a robot: matched by a framework profile, by a robot-specific
    default rule, or by the generic code rule with at least one extracted control call."""
    rule = action.get("rule") or ""
    if not rule.startswith("default:") or rule in ROBOT_SPECIFIC_DEFAULT_RULES:
        return True
    return rule == "default:act_code" and bool(action["calls"])
