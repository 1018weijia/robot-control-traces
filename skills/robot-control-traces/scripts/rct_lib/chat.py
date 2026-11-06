"""Rollouts → raw chat export (rct.py chat): manifest.json, sessions/<id>/{session.json, rounds.jsonl, images/}.

Every piece of the conversation is copied verbatim (after redaction) in the order of the rollout file and labelled
task / observation / analysis / action / feedback / other. A round runs from the end of the previous round to the last
feedback of its own actions, so one line of rounds.jsonl holds what an action was based on, the action and what it
returned. A tool call and its output are separate records in the file; ordering by file position keeps parallel calls
and late results where the agent actually received them.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import codex, export as export_mod, steps as steps_mod
from .redact import Redactor

SCHEMA = "robot-control-chat/v1"
MANIFEST_SCHEMA = "robot-control-chat-manifest/v1"
ROLE_CATEGORY = {"act": "action", "observe": "observation", "result": "feedback"}
MESSAGE_KIND = {"user_message": ("task", None), "image": ("observation", "user_image"),
                "reasoning": ("analysis", "reasoning"), "assistant_message": ("analysis", "message")}
COUNTED = ("user_message", "image", "reasoning", "assistant_message", "compaction", "turn_aborted")
NO_OUTPUT = "no output in the recording"


@dataclasses.dataclass
class Options:
    profiles: tuple = ()
    auto_profiles: bool = True
    images: str = "seen"            # seen | none
    redact: bool = True
    path_map: tuple = ()
    max_chars: int = 0              # 0 keeps every text whole
    min_control_steps: int = 1      # 0 also exports rollouts without robot-control evidence


@dataclasses.dataclass
class Piece:
    pos: tuple                      # (line in the rollout file, seq, call before output)
    event: object
    part: str                       # message | call (an action) | output (of an action) | tool (other call + output)
    category: str
    action: object = None           # the act event a feedback belongs to


def pieces(rollout):
    found, by_seq, last_act = [], {e.seq: e for e in rollout.events}, None
    for event in rollout.events:
        if event.kind != "tool_call":
            category, _ = MESSAGE_KIND.get(event.kind, ("other", None))
            found.append(Piece((event.raw, event.seq, 0), event, "message", category))
        elif event.role == "act":
            found.append(Piece((event.raw, event.seq, 0), event, "call", "action"))
            if event.output_raw is not None:
                found.append(Piece((event.output_raw, event.seq, 1), event, "output", "feedback", event))
            last_act = event
        else:
            category, action = ROLE_CATEGORY.get(event.role, "other"), None
            if event.role == "result":
                origin = by_seq.get(event.continuation_of)
                action = origin if origin is not None and origin.role == "act" else last_act
                category = "feedback" if action is not None else "other"
            line = event.output_raw if event.output_raw is not None else event.raw
            found.append(Piece((line, event.seq, 1), event, "tool", category, action))
    return sorted(found, key=lambda p: p.pos)


def rounds(ordered):
    """[[pieces of round 0], ...], [pieces after the last round]."""
    last = {}
    for index, piece in enumerate(ordered):
        if piece.category == "action":
            last.setdefault(piece.event.seq, index)
        elif piece.category == "feedback":
            last[piece.action.seq] = index
    groups, current, end = [], [], None
    for index, piece in enumerate(ordered):
        current.append(piece)
        if piece.category == "action":
            end = max(index if end is None else end, last[piece.event.seq])
        if end is not None and index >= end:
            groups.append(current)
            current, end = [], None
    return groups, current


def source_counts(rollout):
    """What the recording holds, counted straight from the parsed rollout; check() compares the export with it."""
    events, calls = rollout.events, [e for e in rollout.events if e.kind == "tool_call"]
    counts = {kind: sum(1 for e in events if e.kind == kind) for kind in COUNTED}
    counts.update(tool_call=len(calls), tool_output=sum(1 for e in calls if e.output_raw is not None),
                  images=sum(len(e.images) for e in events))
    return counts


def _is_rationale(record):
    return record["category"] == "analysis" and "call_id" in record


def _exported_counts(records):
    counts = dict.fromkeys(COUNTED + ("tool_call", "tool_output", "images"), 0)
    kinds = {"user_image": "image", "reasoning": "reasoning", "message": "assistant_message",
             "compaction": "compaction", "turn_aborted": "turn_aborted"}
    for record in records:
        if _is_rationale(record):
            continue
        if record["category"] == "task":
            counts["user_message"] += 1
        elif "call_id" not in record and record.get("kind") in kinds:
            counts[kinds[record["kind"]]] += 1
        if "call_id" in record:
            counts["tool_call"] += "command" in record or "args" in record
            counts["tool_output"] += "text" in record
        counts["images"] += len(record.get("images", []))
    return counts


class Session:
    def __init__(self, folder, rollout, profile, options):
        self.folder, self.rollout, self.profile, self.options = folder, rollout, profile, options

    def text(self, value):
        return steps_mod.clip(value, self.options.max_chars)

    def image(self, key):
        """images/<sha256[:16]>.<ext>, or None when the recording holds no bytes (or --images none)."""
        image = self.rollout.images.get(key)
        if self.options.images == "none" or image is None or image.data is None:
            return None
        name = key[:16] + export_mod.EXTENSIONS.get(image.mime or "", ".png")
        path = self.folder / "images" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.data)
        return f"images/{name}"

    def images(self, record, keys):
        if keys:
            record["images"] = [self.image(key) for key in keys]

    def records(self, piece, seq):
        event, record = piece.event, {"seq": seq, "category": piece.category}
        if piece.part == "message":
            kind = MESSAGE_KIND.get(event.kind, (None, event.kind))[1]
            if kind:
                record["kind"] = kind
            if event.text:
                record["text"] = self.text(event.text)
            if event.phase:
                record["phase"] = event.phase
            self.images(record, event.images)
            record["ts"] = event.ts
            return [record]
        record["call_id"] = event.call_id
        if piece.part != "output":
            record["tool"] = event.tool
            if event.command is not None:
                record["command"] = event.command
            else:
                record["args"] = event.args
            if piece.part == "call" and event.code and event.command is not None \
                    and event.code.strip() not in event.command:
                record["code"] = event.code
        if piece.action is not None:
            record["action"] = piece.action.call_id
        if piece.part != "call":
            output, exit_code = (event.own_output, event.own_exit_code) if event.continued \
                else (event.output, event.exit_code)
            if event.output_raw is not None:
                record["text"] = self.text(output or "")
            if exit_code is not None:
                record["exit_code"] = exit_code
            self.images(record, event.images)
            if event.output_parts and not event.continued:
                record["parts"] = [{"text": self.text(value)} if kind == "text" else {"image": self.image(value)}
                                   for kind, value in event.output_parts]
        record["ts"] = event.ts if piece.part == "call" else event.end_ts or event.ts
        if piece.part == "output" or event.role not in ("act", "observe"):
            return [record]
        notes = [{"seq": seq, "category": "analysis", "kind": r["key"], "call_id": event.call_id,
                  "text": self.text(r["text"])} for r in steps_mod.rationale([event], None, self.profile)]
        return notes + [record]

    def write(self, redactor):
        ordered = pieces(self.rollout)
        groups, tail = rounds(ordered)
        lines, seq, counts = [], 0, {}
        for number, group in enumerate(groups + ([tail] if tail else [])):
            events = []
            for piece in group:
                seq += 1
                events += self.records(piece, seq)
            for record in events:
                counts[record["category"]] = counts.get(record["category"], 0) + 1
            lines.append({"round": number if number < len(groups) else "final",
                          "actions": [p.event.call_id for p in group if p.category == "action"],
                          "context_reset": any(p.event.kind == "compaction" for p in group),
                          "events": events})
        answered = {p.action.seq for p in ordered if p.category == "feedback"}
        missing = [{"call_id": p.event.call_id, "reason": NO_OUTPUT} for p in ordered
                   if p.category == "action" and p.event.seq not in answered]
        with open(self.folder / "rounds.jsonl", "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(redactor.obj(line), ensure_ascii=False, default=str) + "\n")
        rollout = self.rollout
        session = {
            "schema": SCHEMA, "session": self.folder.name,
            "source": {"rollout_file": rollout.path.name, "rollout_sha256": rollout.sha256,
                       "thread_id": rollout.thread_id,
                       "parent_thread_id": rollout.meta.get("parent_thread_id") or rollout.meta.get("forked_from_id"),
                       "started_at": rollout.started,
                       "originator": rollout.meta.get("originator"), "cli_version": rollout.meta.get("cli_version")},
            "agent": {"models": rollout.models,
                      "efforts": sorted({c.get("effort") for c in rollout.turn_contexts if c.get("effort")})},
            "task": next((e.text for e in rollout.events if e.kind == "user_message" and e.text), None),
            "profiles": self.profile.names, "rounds": len(groups), "final_round": bool(tail),
            "counts": counts, "source_counts": source_counts(rollout), "actions_without_feedback": missing,
        }
        with open(self.folder / "session.json", "w", encoding="utf-8") as handle:
            json.dump(redactor.obj(session), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        return len(lines), len(groups)


def check_session(folder):
    folder = Path(folder)
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    lines = [json.loads(l) for l in (folder / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    errors, previous, records = [], 0, []
    unanswered = {a["call_id"] for a in session["actions_without_feedback"]}
    for line in lines:
        where, events = f"round {line['round']}", line["events"]
        actions = [e["call_id"] for e in events if e["category"] == "action"]
        if actions != line["actions"]:
            errors.append(f"{where}: 'actions' does not list the actions of the round")
        answered = {e.get("action") for e in events if e["category"] == "feedback"}
        for index, record in enumerate(events):
            records.append(record)
            if _is_rationale(record):
                after = next((e for e in events[index + 1:] if not _is_rationale(e)), None)
                if after is None or (after["seq"], after.get("call_id")) != (record["seq"], record["call_id"]):
                    errors.append(f"{where}: seq {record['seq']}: analysis taken from a call is not followed by it")
                continue
            if record["seq"] <= previous:
                errors.append(f"{where}: seq {record['seq']} does not increase")
            previous = record["seq"]
            if record["category"] == "feedback" and record.get("action") not in actions:
                errors.append(f"{where}: seq {record['seq']}: feedback for an action outside this round")
            for rel in record.get("images", []):
                path = folder / rel if rel else None
                if path is None:
                    continue
                if not path.is_file():
                    errors.append(f"{where}: image {rel} is missing")
                elif hashlib.sha256(path.read_bytes()).hexdigest()[:16] != Path(rel).stem:
                    errors.append(f"{where}: image {rel} does not match its content")
            if "parts" in record and [p["image"] for p in record["parts"] if "image" in p] != record.get("images"):
                errors.append(f"{where}: seq {record['seq']}: parts and images list different images")
        for call_id in actions:
            if call_id not in answered and call_id not in unanswered:
                errors.append(f"{where}: action {call_id} without feedback")
            if call_id in answered and call_id in unanswered:
                errors.append(f"{where}: action {call_id} has feedback but is listed without")
    exported = _exported_counts(records)
    for key, count in session["source_counts"].items():
        if exported.get(key) != count:
            errors.append(f"{key}: {count} in the recording, {exported.get(key)} exported")
    return errors


def check(out, manifest=None):
    """Read the export back and verify order, pairing, images and that every recorded piece appears exactly once."""
    out = Path(out)
    manifest = manifest or json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    for entry in manifest["sessions"]:
        if "id" in entry:
            errors += [f"{entry['id']}: {e}" for e in check_session(out / "sessions" / entry["id"])]
    return errors


def _raw_key(url):
    _, data = codex.decode_data_url(url)
    if data is not None:
        return hashlib.sha256(data).hexdigest()
    return "url:" + hashlib.sha256(url.encode("utf-8", "replace")).hexdigest()


def _raw_keys(value):
    """Image items anywhere in a recorded value, in document order."""
    if isinstance(value, list):
        return [k for v in value for k in _raw_keys(v)]
    if not isinstance(value, dict):
        return []
    if value.get("file_id"):
        return ["file:" + str(value["file_id"])]
    url = codex.image_url(value) if "image_url" in value or value.get("type") == "image" else None
    if isinstance(url, str) and url:
        return [_raw_key(url)]
    return [k for v in value.values() for k in _raw_keys(v)]


def raw_images(path):
    """Read straight from the rollout file, without the reader: the images of each tool output by call_id (an MCP
    result recorded as a JSON string, or only in its end event, included), the images of user-role messages in file
    order, and the images event_msg user messages list."""
    calls, user, typed, ends = {}, [], set(), {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        kind, item = (record.get("type"), record.get("payload")) if "payload" in record else ("response_item", record)
        if not isinstance(item, dict):
            continue
        if kind == "response_item" and item.get("type") in ("function_call_output", "custom_tool_call_output",
                                                             "tool_search_output"):
            output = item.get("output")
            if isinstance(output, str) and codex.mcp_result(codex.parse_json(output)):
                output = json.loads(output)
            calls.setdefault(item.get("call_id"), []).extend(_raw_keys(output))
        elif kind == "response_item" and item.get("type") == "message" and item.get("role") == "user":
            user += _raw_keys(item.get("content"))
        elif kind == "event_msg" and item.get("type") == "user_message":
            typed.update(_raw_key(u) for u in item.get("images") or [] if isinstance(u, str) and u)
        elif kind == "event_msg" and item.get("type") == "mcp_tool_call_end":
            ends[item.get("call_id")] = _raw_keys(item.get("result"))
    for call_id, keys in ends.items():
        calls.setdefault(call_id, keys)
    return calls, user, typed


def _same(ref, key, saved):
    if ref is None:
        return not saved or key.startswith(("url:", "file:"))
    return Path(ref).stem == key[:16]


def crosscheck(rollout_path, folder, saved=True):
    """Every image reference in rounds.jsonl against where the image is in the raw rollout. A tool output's images
    belong to that call, in that order. Images an older CLI injected after view_image, and images on user messages,
    come from user-role messages: in file order, the injected ones in the order of their calls."""
    calls, user, typed = raw_images(rollout_path)
    exported, free = {}, []
    for line in (Path(folder) / "rounds.jsonl").read_text(encoding="utf-8").splitlines():
        for record in json.loads(line)["events"] if line.strip() else []:
            if "images" in record and not _is_rationale(record):
                if "call_id" in record:
                    exported.setdefault(record["call_id"], []).extend(record["images"])
                else:
                    free.extend(record["images"])
    errors, injected = [], []
    for call_id in list(exported) + [c for c in calls if c not in exported]:
        raw, got = calls.get(call_id, []), exported.get(call_id, [])
        if len(got) < len(raw) or not all(_same(r, k, saved) for r, k in zip(got, raw)):
            errors.append(f"call {call_id}: images differ from its output in the recording")
        else:
            injected += [(call_id, r) for r in got[len(raw):]]
    taken, found = set(), []
    for call_id, r in injected:
        index = next((i for i, k in enumerate(user) if i not in taken and _same(r, k, saved)), None)
        if index is None:
            errors.append(f"call {call_id}: an attached image is not in the recording")
            continue
        taken.add(index)
        found.append((call_id, index))
    wrong = {c for n, (first, a) in enumerate(found) for second, b in found[n + 1:] if b < a for c in (first, second)}
    if wrong:
        errors.append(f"calls {', '.join(sorted(map(str, wrong)))}: attached images are not in the recorded order")
    pool, position = [k for i, k in enumerate(user) if i not in taken], 0
    for r in free:
        if position < len(pool) and _same(r, pool[position], saved):
            position += 1
        elif not any(_same(r, k, saved) for k in typed):
            errors.append(f"user image {r} is not where the recording has it")
    if position < len(pool):
        errors.append(f"{len(pool) - position} user-role image(s) of the recording are not in the export")
    return errors


def export(paths, out, options: Options, overwrite=False, log=print):
    out = Path(out)
    export_mod.prepare_out(out, overwrite, "sessions", MANIFEST_SCHEMA)
    redactor = Redactor(options.redact, options.path_map)
    settings = {k: v for k, v in dataclasses.asdict(options).items() if k != "path_map"}
    manifest = {"schema": MANIFEST_SCHEMA, "generator": {"name": "robot-control-traces", "version": export_mod.VERSION},
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "options": redactor.obj(settings), "sessions": []}
    gate = export_mod.Options(profiles=options.profiles, auto_profiles=options.auto_profiles,
                              images="none" if options.images == "none" else "seen", use_adapters=False)
    image_errors = []
    paths = list(dict.fromkeys(map(Path, paths)))
    copies = codex.copies(paths)
    for path in paths:
        if path in copies:
            manifest["sessions"].append({"rollout_file": path.name, "skipped": f"copy of {copies[path].name}"})
            log(f"skip {path.name}: copy of {copies[path].name}")
            continue
        try:
            rollout, profile, episodes, _, _ = export_mod.analyse(path, gate)
        except Exception as exc:  # one broken rollout must not stop a batch
            manifest["sessions"].append({"rollout_file": path.name, "skipped": f"parse error: {type(exc).__name__}"})
            log(f"skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        acts = sum(len(g) for _, g in episodes)
        control = sum(1 for _, g in episodes for s in g if s["flags"].get("control"))
        if control < options.min_control_steps:
            manifest["sessions"].append({"rollout_file": path.name, "rollout_sha256": rollout.sha256,
                                         "skipped": f"no robot-control evidence: {control} of {acts} act steps"})
            log(f"skip {path.name}: {acts} act steps but {control} with robot-control evidence")
            continue
        folder = out / "sessions" / export_mod.unique_name(out / "sessions", rollout)
        folder.mkdir(parents=True)
        lines, groups = Session(folder, rollout, profile, options).write(redactor)
        image_errors += [f"{folder.name}: {e}" for e in crosscheck(path, folder, options.images != "none")]
        manifest["sessions"].append({"id": folder.name, "rollout_file": path.name, "rollout_sha256": rollout.sha256,
                                     "rounds": groups, "final_round": lines > groups, "actions": acts})
        log(f"{path.name}: {groups} round(s) with actions, {lines - groups} final")
    manifest["redaction"] = {"enabled": options.redact, "counts": redactor.counts,
                             "custom_paths": len(options.path_map)}
    errors = check(out, manifest) + image_errors
    manifest["check"] = {"errors": len(errors), "first": errors[:20]}
    with open(out / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest
