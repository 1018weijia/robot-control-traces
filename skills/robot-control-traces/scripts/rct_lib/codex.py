"""Read Codex rollout files (``$CODEX_HOME/sessions/**/rollout-*.jsonl``) into a normalized timeline.

The reader is deliberately tolerant: it accepts the current ``{timestamp, type, payload}`` line format and the
legacy format (bare response items after a session-meta line), skips malformed or truncated lines with a warning,
never expands compaction ``replacement_history`` (it repeats earlier items), and prefers ``response_item`` records
over their ``event_msg`` mirrors so that nothing is counted twice.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

SHELL_TOOLS = {"shell", "shell_command", "exec_command", "container.exec", "local_shell", "unified_exec"}
CONTEXT_PREFIXES = ("<environment_context>", "<user_instructions>", "# AGENTS.md instructions", "<INSTRUCTIONS>",
                    "<permissions instructions>", "<collaboration_mode>", "<personality>", "<skills_instructions>",
                    "<turn_aborted>", "<subagent_notification>", "<model_switch>")
SESSION_ID_RE = re.compile(r"[Ss]ession ID[:\s]+(\d+)")
EXIT_RE = re.compile(r"(?:Process exited with code|Exit code:)\s*(-?\d+)")
WALL_RE = re.compile(r"Wall time:\s*([\d.]+)\s*s")


@dataclasses.dataclass
class Image:
    key: str
    mime: str | None
    data: bytes | None
    source: str
    path_hint: str | None = None
    call_id: str | None = None


@dataclasses.dataclass
class Event:
    seq: int
    ts: str | None
    kind: str
    turn: int = 0
    text: str = ""
    phase: str | None = None
    call_id: str | None = None
    tool: str | None = None
    args: object = None
    command: str | None = None
    workdir: str | None = None
    output: str | None = None
    exit_code: int | None = None
    duration_s: float | None = None
    end_ts: str | None = None
    images: list = dataclasses.field(default_factory=list)
    origin: str = "response_item"
    continuation_of: int | None = None
    process_session: str | None = None
    # filled by later stages
    role: str | None = None
    rule: str | None = None
    code: str | None = None
    code_source: str | None = None
    skeleton: str | None = None
    signature: str | None = None


@dataclasses.dataclass
class Rollout:
    path: Path
    sha256: str
    meta: dict
    turn_contexts: list
    events: list
    images: dict
    warnings: list
    stats: dict

    @property
    def thread_id(self):
        return str(self.meta.get("id") or self.meta.get("session_id") or self.path.stem)

    @property
    def started(self):
        return self.meta.get("timestamp") or (self.events[0].ts if self.events else None)

    @property
    def models(self):
        seen = []
        for ctx in self.turn_contexts:
            model = ctx.get("model")
            if model and model not in seen:
                seen.append(model)
        return seen

    @property
    def cwd(self):
        return self.meta.get("cwd") or next((c.get("cwd") for c in self.turn_contexts if c.get("cwd")), None)


def codex_home(explicit=None) -> Path:
    return Path(explicit or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


def find_rollouts(home: Path, since: str | None = None) -> list:
    """Every rollout under sessions/ and archived_sessions/, oldest first. ``since`` is YYYY-MM-DD."""
    found = []
    for sub in ("sessions", "archived_sessions"):
        root = home / sub
        if root.is_dir():
            found += [p for p in root.rglob("*.jsonl") if p.is_file()]
    if since:
        found = [p for p in found if _file_date(p) >= since]
    return sorted(set(found), key=lambda p: (_file_date(p), p.name))


def _file_date(path: Path) -> str:
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)
    if match:
        return "-".join(match.groups())
    import time
    return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))


def command_text(tool, args):
    """The shell script a shell-like tool call ran, or None for non-shell tools."""
    if not isinstance(args, dict) or not tool:
        return None
    base = tool.split(".")[-1]
    if tool not in SHELL_TOOLS and base not in SHELL_TOOLS and "shell" not in base and "exec" not in base:
        return None
    for key in ("cmd", "command", "script"):
        value = args.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            if len(value) >= 3 and os.path.basename(value[0]) in ("bash", "sh", "zsh") and value[1] in ("-lc", "-c"):
                return value[2]
            return shlex.join(value)
    return None


def decode_data_url(url: str):
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    header, _, body = url.partition(",")
    mime = header[5:].split(";")[0] or None
    try:
        data = base64.b64decode(body, validate=False) if ";base64" in header else body.encode()
    except (binascii.Error, ValueError):
        return mime, None
    return mime, data


def _parse_json(text):
    if isinstance(text, str) and text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            return None
    return None


class _Builder:
    def __init__(self, keep_images):
        self.keep_images = keep_images
        self.meta = {}
        self.extra_meta = []
        self.turn_contexts = []
        self.events = []
        self.images = {}
        self.warnings = []
        self.stats = {"lines": 0, "records": {}, "context_messages": 0, "encrypted_dropped": 0}
        self.calls = {}
        self.fallback = {"user_message": [], "agent_message": [], "agent_reasoning": []}
        self.exec_end = {}
        self.view_paths = {}
        self.mcp_end = {}
        self.cwd = None
        self.seq = 0

    # ---- helpers ----------------------------------------------------------------------------------------------
    def _event(self, kind, ts, **kw):
        self.seq += 1
        event = Event(seq=self.seq, ts=ts, kind=kind, **kw)
        self.events.append(event)
        return event

    def _image(self, url=None, file_id=None, source="tool_output", call_id=None):
        if file_id:
            key = "file:" + str(file_id)
            self.images.setdefault(key, Image(key, None, None, source, call_id=call_id))
            return key
        mime, data = decode_data_url(url)
        if data is None:
            return None
        key = hashlib.sha256(data).hexdigest()
        if key not in self.images:
            self.images[key] = Image(key, mime, data if self.keep_images else None, source, call_id=call_id)
        return key

    def _content(self, content, source, call_id=None):
        texts, images = [], []
        if isinstance(content, str):
            return [content], []
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("input_text", "output_text", "text", "summary_text", "reasoning_text") and item.get("text"):
                texts.append(str(item["text"]))
            elif kind in ("input_image", "image", "image_url") or "image_url" in item or "file_id" in item:
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                key = self._image(url=url, file_id=item.get("file_id"), source=source, call_id=call_id)
                if key:
                    images.append(key)
            if item.get("encrypted_content"):
                self.stats["encrypted_dropped"] += 1
        return texts, images

    def _output(self, output, call_id):
        """Tool output → (text, image keys, exit code, duration, running process session id)."""
        exit_code = duration = session = None
        images = []
        if isinstance(output, dict):
            if "content_items" in output:
                output = output["content_items"]
            else:
                if output.get("success") is False:
                    exit_code = 1
                output = output.get("content", json.dumps(output, ensure_ascii=False))
        if isinstance(output, list):
            texts, images = self._content(output, "tool_output", call_id)
            text = "\n".join(texts)
        else:
            text = "" if output is None else str(output)
        parsed = _parse_json(text)
        if isinstance(parsed, dict) and "output" in parsed and isinstance(parsed.get("metadata"), dict):
            meta = parsed["metadata"]
            text = str(parsed["output"])
            exit_code = meta.get("exit_code", exit_code)
            duration = meta.get("duration_seconds", duration)
        else:
            match = EXIT_RE.search(text[:600])
            if match:
                exit_code = int(match.group(1))
            match = WALL_RE.search(text[:600])
            if match:
                duration = float(match.group(1))
            match = SESSION_ID_RE.search(text[:600]) if exit_code is None else None
            if match:
                session = match.group(1)
            if "\nOutput:\n" in text[:2000]:
                text = text.split("\nOutput:\n", 1)[1]
        return text, images, exit_code, duration, session

    # ---- records ----------------------------------------------------------------------------------------------
    def feed(self, kind, payload, ts):
        self.stats["records"][kind] = self.stats["records"].get(kind, 0) + 1
        if not isinstance(payload, dict):
            return
        if kind == "session_meta":
            meta = payload.get("meta", payload)
            if not self.meta:
                self.meta = dict(meta, git=payload.get("git"))
                self.cwd = meta.get("cwd")
            else:
                self.extra_meta.append(meta)
        elif kind == "turn_context":
            self.cwd = payload.get("cwd") or self.cwd
            self.turn_contexts.append({k: payload.get(k) for k in ("turn_id", "cwd", "model", "effort", "summary",
                                                                     "approval_policy")} | {"ts": ts})
        elif kind == "response_item":
            self._response_item(payload, ts)
        elif kind == "event_msg":
            self._event_msg(payload, ts)
        elif kind == "compacted":
            self._event("compaction", ts, text=str(payload.get("message") or "")[:4000])

    def _response_item(self, item, ts):
        kind = item.get("type")
        if item.get("encrypted_content"):
            self.stats["encrypted_dropped"] += 1
        if kind == "message":
            role = item.get("role")
            if role in ("developer", "system"):
                self.stats["context_messages"] += 1
                return
            texts, images = self._content(item.get("content"), "user" if role == "user" else "assistant")
            text = "\n".join(texts).strip()
            if role == "assistant":
                if text:
                    self._event("assistant_message", ts, text=text, phase=item.get("phase"))
                return
            pending = self._pending_view_image()
            if images and pending is not None:
                for key in images:
                    self.images[key].source = "view_image"
                    self.images[key].call_id = pending.call_id
                pending.images += images
                return
            if text and text.startswith(CONTEXT_PREFIXES):
                self.stats["context_messages"] += 1
                text = ""
            if text:
                self._event("user_message", ts, text=text, images=images)
            elif images:
                self._event("image", ts, images=images, origin="user")
        elif kind == "reasoning":
            summary, _ = self._content(item.get("summary"), "reasoning")
            raw, _ = self._content(item.get("content"), "reasoning")
            text = "\n\n".join(t.strip() for t in summary + raw if t.strip())
            if text:
                self._event("reasoning", ts, text=text)
        elif kind in ("function_call", "custom_tool_call", "local_shell_call", "web_search_call", "tool_search_call"):
            self._call(kind, item, ts)
        elif kind in ("function_call_output", "custom_tool_call_output", "tool_search_output"):
            call = self.calls.get(item.get("call_id"))
            if call is None:
                call = self._event("tool_call", ts, call_id=item.get("call_id"), tool=item.get("name") or "unknown")
                self.warnings.append(f"output without a matching call: {item.get('call_id')}")
            text, images, exit_code, duration, session = self._output(item.get("output"), call.call_id)
            call.output = text if call.output is None else call.output + "\n" + text
            call.images += images
            call.exit_code = exit_code if exit_code is not None else call.exit_code
            call.duration_s = duration if duration is not None else call.duration_s
            call.process_session = session or call.process_session
            call.end_ts = ts

    def _call(self, kind, item, ts):
        name = item.get("name") or {"local_shell_call": "local_shell", "web_search_call": "web_search",
                                    "tool_search_call": "tool_search"}.get(kind, kind)
        if item.get("namespace"):
            name = f"{item['namespace']}.{name}"
        if kind == "function_call":
            raw = item.get("arguments")
            args = _parse_json(raw) if isinstance(raw, str) else raw
            args = raw if args is None else args
        elif kind == "custom_tool_call":
            args = item.get("input")
        elif kind == "local_shell_call":
            args = item.get("action") or {}
        else:
            args = item.get("action") or item.get("arguments")
        call_id = item.get("call_id") or item.get("id")
        workdir = args.get("workdir") or args.get("working_directory") or args.get("cwd") if isinstance(args, dict) \
            else None
        event = self._event("tool_call", ts, call_id=call_id, tool=name, args=args,
                            command=command_text(name, args), workdir=workdir or self.cwd)
        if call_id:
            self.calls[call_id] = event

    def _pending_view_image(self):
        for event in reversed(self.events[-6:]):
            if event.kind == "tool_call" and event.tool and event.tool.split(".")[-1] == "view_image":
                return None if event.images else event
            if event.kind in ("tool_call", "user_message", "assistant_message"):
                return None
        return None

    def _event_msg(self, msg, ts):
        kind = msg.get("type")
        if kind == "user_message":
            images = [k for k in (self._image(url=u, source="user") for u in msg.get("images") or []) if k]
            self.fallback["user_message"].append((ts, str(msg.get("message") or ""), images))
        elif kind == "agent_message":
            self.fallback["agent_message"].append((ts, str(msg.get("message") or ""), msg.get("phase")))
        elif kind == "agent_reasoning":
            self.fallback["agent_reasoning"].append((ts, str(msg.get("text") or "")))
        elif kind == "exec_command_end":
            self.exec_end[msg.get("call_id")] = msg
        elif kind == "view_image_tool_call":
            path = msg.get("path")
            self.view_paths[msg.get("call_id")] = path if isinstance(path, str) else json.dumps(path)
        elif kind == "mcp_tool_call_end":
            self.mcp_end[msg.get("call_id")] = msg
        elif kind == "turn_aborted":
            self._event("turn_aborted", ts, text=str(msg.get("reason") or ""))
        elif kind == "token_count" and isinstance(msg.get("info"), dict):
            self.stats["token_usage"] = msg["info"].get("total_token_usage")

    # ---- finish -----------------------------------------------------------------------------------------------
    def finish(self):
        self._merge_fallbacks()
        for call_id, end in self.exec_end.items():
            call = self.calls.get(call_id)
            if call is None:
                continue
            if end.get("exit_code") is not None:
                call.exit_code = end["exit_code"]
            if not call.output and end.get("aggregated_output"):
                call.output = str(end["aggregated_output"])
        for call_id, path in self.view_paths.items():
            call = self.calls.get(call_id)
            if call is not None:
                for key in call.images:
                    self.images[key].path_hint = path
        for call in self.calls.values():
            if call.tool and call.tool.split(".")[-1] == "view_image":
                hint = self.view_paths.get(call.call_id) or (call.args.get("path") if isinstance(call.args, dict)
                                                              else None)
                for key in call.images:
                    self.images[key].path_hint = self.images[key].path_hint or hint
        for call_id, end in self.mcp_end.items():
            call = self.calls.get(call_id)
            if call is not None and call.output is None:
                call.output = self._output(json.dumps(end.get("result"), ensure_ascii=False), call_id)[0]
        self._link_stdin()
        self.events.sort(key=lambda e: e.seq)
        turn = 0
        for event in self.events:
            if event.kind == "user_message":
                turn += 1
            event.turn = turn

    def _merge_fallbacks(self):
        has = {k: any(e.kind == k for e in self.events) for k in ("user_message", "assistant_message", "reasoning")}
        events = self.fallback["user_message"]
        if events:
            # event_msg user messages are exactly what the human typed; response-item copies may include context.
            kept = []
            for event in self.events:
                if event.kind == "user_message":
                    if not event.images:
                        continue
                    event.kind, event.text, event.origin = "image", "", "user"
                kept.append(event)
            self.events = kept
            known = {k for e in self.events for k in e.images}
            for ts, text, images in events:
                self._event("user_message", ts, text=text, images=[k for k in images if k not in known],
                            origin="event_msg")
        if not has["assistant_message"]:
            for ts, text, phase in self.fallback["agent_message"]:
                self._event("assistant_message", ts, text=text, phase=phase, origin="event_msg")
        if not has["reasoning"]:
            for ts, text in self.fallback["agent_reasoning"]:
                self._event("reasoning", ts, text=text, origin="event_msg")
        if events or not has["assistant_message"] or not has["reasoning"]:
            self._resequence()

    def _resequence(self):
        """Fallback events were appended at the end; put them back in time order when every event has a time."""
        if all(e.ts for e in self.events):
            self.events.sort(key=lambda e: (e.ts, e.seq))
        for index, event in enumerate(self.events, 1):
            event.seq = index

    def _link_stdin(self):
        sessions = {}
        for event in sorted(self.events, key=lambda e: e.seq):
            if event.kind != "tool_call":
                continue
            base = (event.tool or "").split(".")[-1]
            if base == "write_stdin" and isinstance(event.args, dict):
                origin = sessions.get(str(event.args.get("session_id")))
                if origin is not None:
                    event.continuation_of = origin.seq
                    if event.output:
                        origin.output = (origin.output or "") + "\n" + event.output
                    if event.exit_code is not None:
                        origin.exit_code = event.exit_code
            elif event.process_session:
                sessions[event.process_session] = event


def read_rollout(path, keep_images=True) -> Rollout:
    path = Path(path)
    raw = path.read_bytes()
    builder = _Builder(keep_images)
    for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        builder.stats["lines"] += 1
        try:
            record = json.loads(line)
        except ValueError:
            builder.warnings.append(f"line {number}: not valid JSON (truncated live session?)")
            continue
        if not isinstance(record, dict):
            builder.warnings.append(f"line {number}: not a JSON object")
            continue
        if "payload" in record and "type" in record:
            builder.feed(record["type"], record["payload"], record.get("timestamp"))
        elif "record_type" in record:
            continue
        elif "type" in record:
            builder.feed("response_item", record, record.get("timestamp"))
        elif "id" in record and "timestamp" in record:
            builder.feed("session_meta", record, record.get("timestamp"))
        else:
            builder.warnings.append(f"line {number}: unrecognised record")
    builder.finish()
    return Rollout(path=path, sha256=hashlib.sha256(raw).hexdigest(), meta=builder.meta,
                   turn_contexts=builder.turn_contexts, events=builder.events, images=builder.images,
                   warnings=builder.warnings, stats=builder.stats)
