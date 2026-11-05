"""Synthetic Codex rollouts and cell-queue session directories for the tests."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct
import sys
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "robot-control-traces" / "scripts"))

T0 = datetime(2026, 11, 4, 10, 0, 0, tzinfo=timezone.utc)


def png(rgb=(200, 30, 30), size=2):
    """A tiny valid PNG without third-party libraries."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


class RolloutBuilder:
    """Writes the current {timestamp, type, payload} rollout format."""

    def __init__(self, cwd="/home/alice/robot", model="gpt-5-codex", thread=None):
        self.lines = []
        self.t = T0
        self.thread = thread or str(uuid.UUID(int=1))
        self.n = 0
        self._line("session_meta", {"id": self.thread, "timestamp": self.ts(), "cwd": cwd, "originator": "codex_cli_rs",
                                    "cli_version": "0.99.0", "base_instructions": {"text": "SECRET SYSTEM PROMPT"},
                                    "git": {"repository_url": "git@github.com:alice/private.git"}})
        self._line("turn_context", {"cwd": cwd, "model": model, "effort": "high", "approval_policy": "never",
                                    "sandbox_policy": {"type": "danger-full-access"}, "summary": "auto"})

    def ts(self):
        self.t += timedelta(seconds=2)
        return self.t.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _line(self, kind, payload):
        self.lines.append(json.dumps({"timestamp": self.ts(), "type": kind, "payload": payload}))

    def call_id(self):
        self.n += 1
        return f"call_{self.n:03d}"

    def context(self):
        self._line("response_item", {"type": "message", "role": "developer",
                                     "content": [{"type": "input_text", "text": "<permissions instructions>x"}]})
        self._line("response_item", {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context>\n  <cwd>/home/alice</cwd>\n</environment_context>"}]})
        return self

    def user(self, text):
        self._line("response_item", {"type": "message", "role": "user",
                                     "content": [{"type": "input_text", "text": text}]})
        self._line("event_msg", {"type": "user_message", "message": text, "images": []})
        return self

    def reasoning(self, text):
        self._line("response_item", {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}],
                                     "content": None, "encrypted_content": "gAAAAAB-opaque"})
        self._line("event_msg", {"type": "agent_reasoning", "text": text})
        return self

    def assistant(self, text, phase="commentary"):
        self._line("response_item", {"type": "message", "role": "assistant", "phase": phase,
                                     "content": [{"type": "output_text", "text": text}]})
        self._line("event_msg", {"type": "agent_message", "message": text})
        return self

    def shell(self, cmd, output="", exit_code=0, tool="exec_command", workdir=None, style="unified"):
        cid = self.call_id()
        args = {"cmd": cmd} if tool == "exec_command" else {"command": ["bash", "-lc", cmd]}
        if workdir:
            args["workdir"] = workdir
        self._line("response_item", {"type": "function_call", "name": tool, "arguments": json.dumps(args),
                                     "call_id": cid})
        if style == "json":
            text = json.dumps({"output": output, "metadata": {"exit_code": exit_code, "duration_seconds": 0.4}})
        elif style == "running":
            text = f"Chunk ID: a1\nWall time: 1.0 seconds\nProcess running with session ID 7\nOutput:\n{output}"
        else:
            text = (f"Chunk ID: a1\nWall time: 0.4 seconds\nProcess exited with code {exit_code}\n"
                    f"Original token count: 10\nOutput:\n{output}")
        self._line("event_msg", {"type": "exec_command_end", "call_id": cid, "turn_id": "t",
                                 "command": ["bash", "-lc", cmd], "cwd": workdir or "/home/alice/robot",
                                 "parsed_cmd": [], "aggregated_output": output, "exit_code": exit_code,
                                 "duration": {"secs": 0, "nanos": 400}, "status": "completed"}
                   if style != "running" else {"type": "token_count", "info": None})
        self._line("response_item", {"type": "function_call_output", "call_id": cid, "output": text})
        return cid

    def stdin(self, session_id, output, exit_code=0):
        cid = self.call_id()
        self._line("response_item", {"type": "function_call", "name": "write_stdin", "call_id": cid,
                                     "arguments": json.dumps({"session_id": session_id, "chars": ""})})
        self._line("response_item", {"type": "function_call_output", "call_id": cid,
                                     "output": f"Wall time: 2.0 seconds\nProcess exited with code {exit_code}\n"
                                               f"Output:\n{output}"})
        return cid

    def view_image(self, path, data, inline=True):
        cid = self.call_id()
        self._line("response_item", {"type": "function_call", "name": "view_image", "call_id": cid,
                                     "arguments": json.dumps({"path": path})})
        self._line("event_msg", {"type": "view_image_tool_call", "call_id": cid, "path": path})
        if inline:
            self._line("response_item", {"type": "function_call_output", "call_id": cid, "output": [
                {"type": "input_image", "image_url": data_url(data)}]})
        else:  # older CLIs: text output, then the image injected as a user message
            self._line("response_item", {"type": "function_call_output", "call_id": cid, "output": "attached"})
            self._line("response_item", {"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": data_url(data)}]})
        return cid

    def mcp(self, name, args, output="ok", image=None, namespace=None):
        cid = self.call_id()
        item = {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": cid}
        if namespace:
            item["namespace"] = namespace
        self._line("response_item", item)
        items = [{"type": "input_text", "text": output}]
        if image is not None:
            items.append({"type": "input_image", "image_url": data_url(image)})
        self._line("response_item", {"type": "function_call_output", "call_id": cid, "output": items})
        return cid

    def apply_patch(self, patch):
        cid = self.call_id()
        self._line("response_item", {"type": "custom_tool_call", "name": "apply_patch", "call_id": cid,
                                     "input": patch})
        self._line("response_item", {"type": "custom_tool_call_output", "call_id": cid,
                                     "output": json.dumps({"output": "Success.", "metadata": {"exit_code": 0}})})
        return cid

    def compact(self, summary):
        self._line("compacted", {"message": summary, "replacement_history": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "DUPLICATE OLD MESSAGE"}]}]})
        return self

    def raw(self, text):
        self.lines.append(text)
        return self

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")
        return path


class SessionBuilder:
    """A cell-queue session directory (commands, responses, observations, primitive ledger, event log)."""

    def __init__(self, root, t0=None):
        self.root = Path(root)
        for name in ("commands", "responses", "observations", "workspace"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.t = (t0 or T0).timestamp()
        self.obs = 0
        self.native = 0
        self.commands = []
        self.primitives = []
        self.events = []
        self._observation()
        (self.root / "run_config.json").write_text(json.dumps({"task": "place_block", "planner": "agent"}))

    def _observation(self):
        folder = self.root / "observations" / f"{self.obs:04d}"
        folder.mkdir(parents=True)
        images, hashes = {}, {}
        import hashlib
        for cam, color in (("cam_head", (10, 10, 200)), ("cam_right_wrist", (10, 200, 10))):
            data = png(tuple((c + self.obs * 7) % 256 for c in color))
            (folder / f"{cam}.png").write_bytes(data)
            images[cam] = str(folder / f"{cam}.png")
            hashes[cam] = hashlib.sha256(data).hexdigest()
        value = {"observation_index": self.obs, "instruction": "Put the block into the bowl.",
                 "robot_state": {"right": {"tcp_pos": [0.2, -0.5, 1.0 - self.native / 1000]}},
                 "native_action_index": self.native, "native_budget": 1200, "images": images,
                 "image_sha256": hashes}
        (folder / "observation.json").write_text(json.dumps(value))
        (self.root / "latest.json").write_text(json.dumps(value))
        self.latest = value
        self.obs += 1
        return value

    def command(self, op, note, at, code=None, moved=0, primitives=(), evidence=None, error=None):
        cid = uuid.uuid4().hex
        command = {"time": at + 0.2, "id": cid, "op": op, "issued_at": at, "note": note}
        if code is not None:
            command["code"] = code
        if op == "ack_visual":
            command.update(observation_index=self.latest["observation_index"],
                           native_action_index=self.latest["native_action_index"],
                           image_sha256={"cam_head": self.latest["image_sha256"]["cam_head"]},
                           viewed_by="agent", evidence=evidence or "a bowl on the left")
        self.commands.append(command)
        self.events.append({"time": at + 0.5, "event": "planner_tick", "op": op})
        (self.root / "commands" / f"{cid}.accepted").write_text(json.dumps(command))
        response = {}
        if error:
            response["error"] = error
        elif op == "python":
            response["output"] = {"stdout": "ok\n", "stderr": "", "error": None, "native_actions": moved,
                                  "episode_ended": False}
            for tool, args, result in primitives:
                self.native += result.get("steps_used", 0)
                self.primitives.append({"tool": tool, "args": args, "result": result,
                                        "native_actions": self.native, "wall_s": 1.0})
            self.native = max(self.native, self.latest["native_action_index"] + moved)
        elif op == "ack_visual":
            response["output"] = {k: command[k] for k in ("observation_index", "native_action_index", "evidence")}
        else:
            response["output"] = None
        response["observation"] = self._observation()
        (self.root / "responses" / f"{cid}.json").write_text(json.dumps(response))
        return cid, response

    def finish(self, success=True):
        with open(self.root / "commands.jsonl", "w") as handle:
            for command in self.commands:
                handle.write(json.dumps(command) + "\n")
        with open(self.root / "primitive_calls.jsonl", "w") as handle:
            for row in self.primitives:
                handle.write(json.dumps(row) + "\n")
        with open(self.root / "planner_events.jsonl", "w") as handle:
            for row in self.events:
                handle.write(json.dumps(row) + "\n")
        (self.root / "result.json").write_text(json.dumps({"task_success": success, "stop_reason": "planner_finish"}))
        return self.root
