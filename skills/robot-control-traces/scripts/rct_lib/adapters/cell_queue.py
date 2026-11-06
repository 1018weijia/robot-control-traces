"""Enrichment for file-queue control sessions driven by a controller script.

The agent submits work through ``<controller>.py [options] observe | ack_visual | cell | finish | poll | where`` and
a runner writes a session directory: commands.jsonl, responses/<id>.json, observations/NNNN/observation.json and,
optionally, primitive_calls.jsonl, run_config.json, result.json and further timestamped *.jsonl event logs. Each
controller invocation in the transcript is matched to one session command by, in order: the command id printed in
its output, the new observation index printed in its output, or op + note + nearest time.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from .. import shell

NAME = "cell_queue"
OPS = {"observe", "ack_visual", "cell", "python", "finish", "poll", "where"}
VALUE_OPTIONS = {"--session", "--output", "--note", "--timeout-s", "--file", "--observation-index",
                 "--native-action-index", "--camera", "--camera-sha256", "--evidence", "--id"}
FLAG_OPTIONS = {"--async", "--brief"}
# Files read explicitly, or too fine-grained (one row per simulator tick) to attach to a step.
NOT_EVENT_LOGS = {"commands.jsonl", "primitive_calls.jsonl", "visual_acknowledgements.jsonl", "python_cells.jsonl",
                  "native_actions.jsonl"}
MAX_LOG_BYTES = 50_000_000
ID_RE = re.compile(r'"id":\s*"([0-9a-f]{32})"')
BRIEF_OBS_RE = re.compile(r"\[observation (\d+), native")
MAX_TIME_GAP_S = 900


def read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def compact(value, max_list=24, depth=0):
    """Drop bulky arrays (action chunks, joint trajectories) but keep their length."""
    if depth > 8:
        return "…"
    if isinstance(value, list):
        if len(value) > max_list:
            return {"_list_len": len(value), "head": [compact(v, max_list, depth + 1) for v in value[:2]]}
        return [compact(v, max_list, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: compact(v, max_list, depth + 1) for k, v in value.items()}
    return value


def epoch(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def remap(path, pairs, reverse=False):
    """Apply the first OLD=NEW prefix mapping (NEW=OLD when reverse) to a path string."""
    text = str(path)
    for old, new in pairs:
        a, b = (new, old) if reverse else (old, new)
        a = a.rstrip("/")
        if a and (text == a or text.startswith(a + "/")):
            return b.rstrip("/") + text[len(a):]
    return text


def invocations(command):
    """Controller invocations in a shell command: [{op, options}] for ``<script>.py ... <op>``."""
    found = []
    for argv in shell.commands(command):
        for index, token in enumerate(argv):
            if not token.endswith(".py"):
                continue
            options, op, rest, i = {}, None, argv[index + 1:], 0
            while i < len(rest):
                tok = rest[i]
                name, sep, value = tok.partition("=")
                if name in VALUE_OPTIONS:
                    if not sep:
                        value = rest[i + 1] if i + 1 < len(rest) else None
                        i += 1
                    options.setdefault(name, []).append(value)
                elif tok in FLAG_OPTIONS:
                    options[tok] = [True]
                elif op is None and tok in OPS:
                    op = tok
                i += 1
            if op is not None:
                found.append({"op": "python" if op == "cell" else op, "options": options})
            break
    return found


class Session:
    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.commands = sorted(read_jsonl(path / "commands.jsonl"), key=lambda r: r.get("time") or 0)
        self.by_id = {c.get("id"): c for c in self.commands}
        self.responses = {}
        for command in self.commands:
            response = read_json(path / "responses" / f"{command.get('id')}.json")
            if isinstance(response, dict):
                self.responses[command.get("id")] = response
        self.obs_index = {cid: (r.get("observation") or {}).get("observation_index")
                          for cid, r in self.responses.items()}
        self.run_config = read_json(path / "run_config.json")
        self.result = read_json(path / "result.json") or read_json(path / "episode_summary.json")
        self.logs = {}
        for log in sorted(path.glob("*.jsonl")):
            if log.name in NOT_EVENT_LOGS or log.stat().st_size > MAX_LOG_BYTES:
                continue
            rows = [r for r in read_jsonl(log) if isinstance(r.get("time"), (int, float))]
            if rows:
                self.logs[log.stem] = rows
        self.primitives = self._assign_primitives(read_jsonl(path / "primitive_calls.jsonl"))

    def window(self, command):
        """[accept time of this command, accept time of the next one)."""
        start = command.get("time") or 0
        later = [c.get("time") for c in self.commands if (c.get("time") or 0) > start]
        return start, (min(later) if later else float("inf"))

    def native_range(self, command):
        index = self.commands.index(command)
        before = 0
        for previous in reversed(self.commands[:index]):
            obs = (self.responses.get(previous.get("id")) or {}).get("observation") or {}
            if "native_action_index" in obs:
                before = obs["native_action_index"]
                break
        obs = (self.responses.get(command.get("id")) or {}).get("observation") or {}
        after = obs.get("native_action_index", before)
        return before, after

    def observation_before(self, command):
        index = self.commands.index(command)
        for previous in reversed(self.commands[:index]):
            obs = (self.responses.get(previous.get("id")) or {}).get("observation")
            if isinstance(obs, dict):
                return obs
        first = read_json(self.path / "observations" / "0000" / "observation.json")
        return first if isinstance(first, dict) else None

    def _assign_primitives(self, rows):
        assigned = {}
        if not rows:
            return assigned
        if "started_at" in rows[0]:
            for row in rows:
                for command in self.commands:
                    start, end = self.window(command)
                    if start <= (row.get("started_at") or 0) < end:
                        assigned.setdefault(command.get("id"), []).append(row)
                        break
            return assigned
        pointer = 0
        for command in self.commands:
            if command.get("op") != "python":
                continue
            before, after = self.native_range(command)
            while pointer < len(rows) and (rows[pointer].get("native_actions") or 0) <= after:
                if (rows[pointer].get("native_actions") or 0) > before:
                    assigned.setdefault(command.get("id"), []).append(rows[pointer])
                pointer += 1
        return assigned

    def events_in(self, command):
        start, end = self.window(command)
        found = {}
        for name, rows in self.logs.items():
            hits = [compact(r) for r in rows if start <= r["time"] < end]
            if hits:
                found[name] = hits
        return found


class Sessions:
    def __init__(self, pairs=()):
        self.cache = {}
        self.pairs = tuple(pairs)

    def load(self, recorded):
        local = Path(remap(recorded, self.pairs))
        key = str(local)
        if key not in self.cache:
            self.cache[key] = Session(local, str(recorded)) if (local / "commands.jsonl").is_file() else None
        return self.cache[key]

    def candidates(self, options, workdir):
        dirs = []
        for value in options.get("--session", []):
            dirs.append(shell.resolve_path(value, workdir))
        for value in options.get("--output", []):
            recorded_root = shell.resolve_path(value, workdir)
            root = Path(remap(recorded_root, self.pairs))
            current = read_json(root / "current.json")
            if isinstance(current, dict) and current.get("session"):
                dirs.append(current["session"])
            if root.is_dir():
                for child in sorted(root.glob("*/commands.jsonl")) + sorted(root.glob("*/*/commands.jsonl")):
                    dirs.append(remap(child.parent, self.pairs, reverse=True))
        seen, sessions = set(), []
        for d in dirs:
            if d and d not in seen:
                seen.add(d)
                session = self.load(d)
                if session is not None:
                    sessions.append(session)
        return sessions


def _response_obs_index(output):
    if not output:
        return None
    text = output.strip()
    start = text.find("{")
    if start >= 0:
        try:
            parsed = json.loads(text[start:])
            obs = parsed.get("observation") if isinstance(parsed, dict) else None
            if isinstance(obs, dict) and isinstance(obs.get("observation_index"), int):
                return obs["observation_index"]
        except ValueError:
            pass
    match = BRIEF_OBS_RE.search(text)
    return int(match.group(1)) if match else None


def match(invocation, event, sessions):
    """(session, command, method) for one controller invocation, or (None, None, reason)."""
    if not sessions:
        return None, None, "session_dir_missing"
    ids = invocation["options"].get("--id", []) + ID_RE.findall(event.output or "")
    for session in sessions:
        for cid in ids:
            if cid in session.by_id:
                return session, session.by_id[cid], "command_id"
    op, note = invocation["op"], (invocation["options"].get("--note") or [""])[0]
    if op in (None, "poll", "where"):
        return None, None, "not_a_command"
    obs = _response_obs_index(event.output)
    if obs is not None:
        for session in sessions:
            for command in session.commands:
                if session.obs_index.get(command.get("id")) == obs and command.get("op") == op \
                        and str(command.get("note", "")).strip() == str(note).strip():
                    return session, command, "observation_index"
    when = epoch(event.ts)
    best = None
    for session in sessions:
        for command in session.commands:
            if command.get("op") != op or str(command.get("note", "")).strip() != str(note).strip():
                continue
            gap = abs((command.get("issued_at") or command.get("time") or 0) - when) if when else 0
            if best is None or gap < best[0]:
                best = (gap, session, command)
    if best and (when is None or best[0] <= MAX_TIME_GAP_S):
        return best[1], best[2], "note_and_time" if when else "note_only"
    return None, None, "no_matching_command"


def _response_summary(response):
    output = (response or {}).get("output")
    summary = {"command_error": (response or {}).get("error")}
    if isinstance(output, dict):
        for key in ("error", "native_actions", "episode_ended", "finishing"):
            if key in output:
                summary[key] = output[key]
    return {k: v for k, v in summary.items() if v is not None}


def enrich(steps, rollout, options):
    pairs = getattr(options, "remap_paths", ())
    sessions = Sessions(pairs)
    stats = {"invocations": 0, "matched": {}, "unmatched": {}}
    episodes = {}
    for step in steps:
        act = step.get("_act")
        events = [(e, "observe") for e in step.get("_observe", [])] + ([(act, "act")] if act else []) \
            + [(e, "result") for e in step.get("_results", [])]
        info = {"acks": [], "commands": []}
        for event, role in events:
            if not event.command:
                continue
            for invocation in invocations(event.command):
                stats["invocations"] += 1
                session, command, method = match(invocation, event, sessions.candidates(invocation["options"],
                                                                                         event.workdir))
                bucket = stats["matched"] if command else stats["unmatched"]
                bucket[method] = bucket.get(method, 0) + 1
                if command is None:
                    continue
                response = session.responses.get(command.get("id"))
                record = {"role": role, "op": command.get("op"), "command_id": command.get("id"), "match": method}
                info["commands"].append(record)
                step["_episode_hint"] = session.label
                episodes.setdefault(session.label, {"session": session.label, "run_config": session.run_config,
                                                    "result": compact(session.result)})
                if command.get("op") == "ack_visual":
                    ack = {k: command.get(k) for k in ("observation_index", "native_action_index", "image_sha256",
                                                       "evidence")}
                    ack["accepted"] = not (response or {}).get("error")
                    info["acks"].append(ack)
                if role != "act":
                    continue
                before, after = session.native_range(command)
                observation = session.observation_before(command)
                if observation is not None:
                    step["observation"]["state"] = {k: v for k, v in observation.items() if k not in ("images",)}
                    seen = {d for a in info["acks"] for d in (a.get("image_sha256") or {}).values()}
                    seen |= {i["id"] for i in step["observation"]["images"]}
                    hashes = observation.get("image_sha256") or {}
                    info["session_images"] = [
                        {"camera": cam, "path": path, "sha256": hashes.get(cam),
                         "seen_by_model": hashes.get(cam) in seen, "_local": remap(path, pairs)}
                        for cam, path in (observation.get("images") or {}).items()]
                if not step["action"]["code"] and command.get("code"):
                    step["action"]["code"], step["action"]["code_source"] = command["code"], "session:commands.jsonl"
                info.update(session=session.label, command_id=command.get("id"), op=command.get("op"),
                            note=command.get("note"), match=method,
                            native_actions={"before": before, "after": after, "count": after - before},
                            response=_response_summary(response),
                            primitives=[compact(r) for r in session.primitives.get(command.get("id"), [])],
                            events=session.events_in(command))
        if info["commands"]:
            step["enrichment"][NAME] = info
    return {"episodes": episodes, "stats": stats}
