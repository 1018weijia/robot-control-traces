#!/usr/bin/env python3
"""Build the synthetic example inputs and export them: python3 examples/make_examples.py [--out DIR]

input/codex_home/   two Codex rollouts (fictional robot sessions, written in the current rollout format)
input/workspace/    the run directory one of them drove (cell-queue session: commands, responses, observations)
output/             what `rct.py export` produces from them
sft/                what `rct.py sft output` turns that export into (step-level training samples)

Everything is generated here; no real conversation, robot or person is involved.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
RCT = HERE.parent / "skills" / "robot-control-traces" / "scripts" / "rct.py"
RECORDED = "/workspace/demo"          # where the fictional run happened; remapped to input/workspace on export
SESSION = "runs/demo_000"
CTL = "python tools/robot_ctl.py --session " + SESSION
W, H = 128, 96


# ---- tiny raster images -------------------------------------------------------------------------------------------

class Image:
    def __init__(self, color):
        self.px = [[color] * W for _ in range(H)]

    def rect(self, cx, cy, w, h, color):
        for y in range(max(0, round(cy - h / 2)), min(H, round(cy + h / 2))):
            for x in range(max(0, round(cx - w / 2)), min(W, round(cx + w / 2))):
                self.px[y][x] = color

    def disc(self, cx, cy, r, color):
        for y in range(max(0, int(cy - r)), min(H, int(cy + r) + 1)):
            for x in range(max(0, int(cx - r)), min(W, int(cx + r) + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.px[y][x] = color

    def png(self):
        raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in self.px)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


TABLE, BOWL, RIM, BLOCK, FINGER = (205, 190, 160), (236, 236, 240), (150, 150, 160), (40, 90, 220), (50, 50, 55)


def top_view(scene):
    """Head camera looking down at the table: x 0.25..0.60 m (far is up), y -0.30..0.30 m (left to right)."""
    img = Image(TABLE)

    def px(x, y):
        return (y + 0.30) / 0.60 * W, (0.60 - x) / 0.35 * H
    bx, by = px(*scene["bowl"])
    img.disc(bx, by, 0.06 / 0.60 * W, RIM)
    img.disc(bx, by, 0.05 / 0.60 * W, BOWL)
    cx, cy = px(*scene["block"])
    img.rect(cx, cy, 0.03 / 0.60 * W, 0.03 / 0.35 * H, BLOCK)
    gx, gy = px(*scene["grip"][:2])
    gap = (0.045 if scene["open"] else 0.03) / 0.60 * W
    img.rect(gx - gap / 2 - 1, gy, 2, 6, FINGER)
    img.rect(gx + gap / 2 + 1, gy, 2, 6, FINGER)
    return img.png()


def wrist_view(scene):
    """Wrist camera centred on the gripper, 0.12 m across."""
    img = Image(TABLE)
    gx, gy = scene["grip"][:2]

    def px(x, y):
        return W / 2 + (y - gy) / 0.12 * W, H / 2 - (x - gx) / 0.12 * W
    bx, by = px(*scene["bowl"])
    img.disc(bx, by, 0.06 / 0.12 * W, RIM)
    img.disc(bx, by, 0.05 / 0.12 * W, BOWL)
    cx, cy = px(*scene["block"])
    img.rect(cx, cy, 0.03 / 0.12 * W, 0.03 / 0.12 * W, BLOCK)
    gap = (0.045 if scene["open"] else 0.03) / 0.12 * W
    img.rect(W / 2 - gap / 2 - 3, H / 2, 6, 30, FINGER)
    img.rect(W / 2 + gap / 2 + 3, H / 2, 6, 30, FINGER)
    return img.png()


def front_view(cube_y, tip):
    """Front camera of the MCP demo: y -0.30..0.30 m left to right, z up; table top at row 70."""
    img = Image((225, 225, 230))
    img.rect(W / 2, 83, W, 26, TABLE)
    col = (cube_y + 0.30) / 0.60 * W
    img.rect(col, 70 - 0.02 / 0.30 * 70, 0.04 / 0.60 * W, 0.04 / 0.30 * 70, (210, 40, 40))
    img.rect((tip[0] + 0.30) / 0.60 * W, 70 - tip[1] / 0.30 * 70, 3, 10, FINGER)
    return img.png()


# ---- rollout writer -----------------------------------------------------------------------------------------------

class Rollout:
    """Current Codex rollout format: one {timestamp, type, payload} object per line."""

    def __init__(self, thread, start, cwd):
        self.lines, self.t, self.n = [], start, 0
        self.line("session_meta", {"id": thread, "timestamp": self.ts(), "cwd": cwd, "originator": "codex_cli_rs",
                                   "cli_version": "0.99.0", "source": "cli"})
        self.line("turn_context", {"cwd": cwd, "model": "gpt-5-codex", "effort": "high", "summary": "auto"})

    def ts(self):
        return self.t.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def tick(self, seconds=2.0):
        self.t += timedelta(seconds=seconds)

    def line(self, kind, payload):
        self.lines.append(json.dumps({"timestamp": self.ts(), "type": kind, "payload": payload}, ensure_ascii=False))
        self.tick(0.5)

    def user(self, text):
        self.line("response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})
        self.line("event_msg", {"type": "user_message", "message": text, "images": []})

    def reasoning(self, text):
        self.line("response_item", {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}],
                                    "content": None, "encrypted_content": "gAAAAAB-opaque"})

    def say(self, text, phase="commentary"):
        self.line("response_item", {"type": "message", "role": "assistant", "phase": phase,
                                    "content": [{"type": "output_text", "text": text}]})

    def call(self, name, arguments, output):
        self.n += 1
        cid = f"call_{self.n:03d}"
        self.line("response_item", {"type": "function_call", "name": name, "call_id": cid,
                                    "arguments": json.dumps(arguments, ensure_ascii=False)})
        self.tick()
        self.line("response_item", {"type": "function_call_output", "call_id": cid, "output": output})

    def shell(self, cmd, output, exit_code=0, workdir=RECORDED):
        self.call("exec_command", {"cmd": cmd, "workdir": workdir},
                  f"Wall time: 1.2 seconds\nProcess exited with code {exit_code}\nOutput:\n{output}")

    def view(self, path, data):
        self.call("view_image", {"path": path},
                  [{"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(data).decode()}])

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")


# ---- demo 1: cell-queue session with a run directory --------------------------------------------------------------

class Session:
    """The run directory a cell-queue runner would write while the agent works."""

    def __init__(self, root, start):
        self.root, self.start = root, start
        self.commands, self.primitives, self.events = [], [], []
        self.obs = self.native = 0
        self.scene = {"grip": (0.35, 0.0, 0.30), "block": (0.42, -0.135), "bowl": (0.40, 0.18), "open": True}
        self.latest = self.observe()

    def observe(self):
        folder = self.root / "observations" / f"{self.obs:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        images, hashes = {}, {}
        for cam, render in (("cam_head", top_view), ("cam_wrist", wrist_view)):
            data = render(self.scene)
            (folder / f"{cam}.png").write_bytes(data)
            images[cam] = f"{RECORDED}/{SESSION}/observations/{self.obs:04d}/{cam}.png"
            hashes[cam] = hashlib.sha256(data).hexdigest()
        g = self.scene["grip"]
        value = {"observation_index": self.obs, "instruction": "Put the blue block into the bowl.",
                 "native_action_index": self.native, "native_budget": 600,
                 "robot_state": {"tcp": [round(v, 3) for v in g], "gripper_width": 0.08 if self.scene["open"] else 0.031},
                 "images": images, "image_sha256": hashes}
        (folder / "observation.json").write_text(json.dumps(value, indent=2))
        self.obs += 1
        return value

    def command(self, op, note, at, code=None, evidence=None, cameras=(), moves=(), error=None, scene=None, out=""):
        cid = hashlib.md5(f"demo-{len(self.commands)}".encode()).hexdigest()
        command = {"time": at + 0.2, "id": cid, "op": op, "issued_at": at, "note": note}
        if code is not None:
            command["code"] = code
        if op == "ack_visual":
            command.update(observation_index=self.latest["observation_index"],
                           native_action_index=self.latest["native_action_index"],
                           image_sha256={c: self.latest["image_sha256"][c] for c in cameras}, evidence=evidence)
        self.commands.append(command)
        response = {}
        if error:
            response["error"] = error
            self.events.append({"time": at + 0.6, "event": "ik_failure", "detail": error})
        elif op == "python":
            for tool, args, steps in moves:
                self.native += steps
                self.primitives.append({"tool": tool, "args": args, "result": {"reached": True, "steps_used": steps},
                                        "native_actions": self.native, "wall_s": round(steps / 40, 2)})
            response["output"] = {"stdout": out, "stderr": "", "error": None,
                                  "native_actions": sum(m[2] for m in moves), "episode_ended": False}
        elif op == "ack_visual":
            response["output"] = {"observation_index": command["observation_index"], "accepted": True}
        else:
            response["output"] = {"episode_ended": True}
        if scene:
            self.scene.update(scene)
        self.latest = response["observation"] = self.observe()
        (self.root / "responses").mkdir(exist_ok=True)
        (self.root / "responses" / f"{cid}.json").write_text(json.dumps(response, indent=2))
        return cid, response

    def close(self, success):
        for name, rows in (("commands", self.commands), ("primitive_calls", self.primitives),
                           ("safety_events", self.events)):
            (self.root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.root / "run_config.json").write_text(json.dumps(
            {"task": "place_block_in_bowl", "native_budget": 600, "cameras": ["cam_head", "cam_wrist"]}, indent=2))
        (self.root / "result.json").write_text(json.dumps(
            {"task_success": success, "stop_reason": "agent_finish", "native_actions": self.native}, indent=2))


def brief(response):
    obs = response["observation"]
    stdout = (response.get("output") or {}).get("stdout", "")
    return f"{stdout}[observation {obs['observation_index']}, native {obs['native_action_index']}/600, ack_required True]"


def demo_cell_queue(base):
    start = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    session = Session(base / "input" / "workspace" / SESSION, start.timestamp())
    r = Rollout("0199c0de-0000-7000-8000-00000000c001", start, RECORDED)
    r.user("Put the blue block into the bowl using tools/robot_ctl.py (session runs/demo_000).")
    steps = [
        {"look": ["cam_head"], "think": "Start from the head camera to locate the block and the bowl.",
         "evidence": "Head: blue block on the left of the table, white bowl on the right; gripper open above the centre.",
         "note": "hover above the block", "scene": {"grip": (0.42, -0.12, 0.13)},
         "code": "BLOCK = np.array([0.42, -0.12, 0.03])\n# hover 10 cm above the block, estimated from the head camera\n"
                 "print(robot.reachable(BLOCK + [0, 0, 0.10]))\nprint(robot.move_to(BLOCK + [0, 0, 0.10], speed=0.15))\n",
         "moves": [("move_to", {"target": [0.42, -0.12, 0.13], "speed": 0.15}, 48)],
         "out": "True\nMove(target=[0.42, -0.12, 0.13], reached=True, steps=48)\n"},
        {"look": ["cam_wrist"], "think": "The wrist view shows the block left of the finger centre line, so the head "
                                         "estimate was off in y by about 1.5 cm.",
         "evidence": "Wrist: block ~9 px (about 1.5 cm) left of the finger centre; fingers open.",
         "note": "correct y and descend", "scene": {"grip": (0.42, -0.135, 0.045)},
         "code": "BLOCK = np.array([0.42, -0.135, 0.03])\n# wrist view: block 1.5 cm further left than the head estimate\n"
                 "robot.move_to(BLOCK + [0, 0, 0.10], speed=0.15)\nrobot.move_to(BLOCK + [0, 0, 0.015], speed=0.05)\n",
         "moves": [("move_to", {"target": [0.42, -0.135, 0.13]}, 12), ("move_to", {"target": [0.42, -0.135, 0.045]}, 30)],
         "out": "", "brief": True},
        {"look": ["cam_wrist"], "think": "Fingers straddle the block; close and lift.",
         "evidence": "Wrist: fingers straddle the block, about 1 cm above the table.",
         "note": "grasp and lift", "scene": {"grip": (0.42, -0.135, 0.165), "open": False},
         "code": "robot.set_gripper(0.0)\nrobot.move_by([0, 0, 0.12], speed=0.1)\nprint(robot.state().gripper_width)\n",
         "moves": [("set_gripper", {"width": 0.0}, 10), ("move_by", {"delta": [0, 0, 0.12]}, 30)],
         "out": "0.031\n"},
        {"look": ["cam_head"], "think": "Holding the block. Carry it high over the bowl to stay clear of the rim.",
         "evidence": "Head: block lifted in the gripper; bowl unchanged on the right.",
         "note": "carry over the bowl", "error": "IK failed: target z=0.35 m is above the workspace (max 0.32 m)",
         "code": "BOWL = np.array([0.40, 0.18, 0.05])\nrobot.move_to(BOWL + [0, 0, 0.30], speed=0.15)\n"},
        {"user": "Go slower near the bowl, please.",
         "think": "The carry height was out of reach and the user asked for less speed: 15 cm above the bowl at 0.06 m/s.",
         "note": "carry over the bowl, lower and slower", "scene": {"grip": (0.40, 0.18, 0.20)},
         "code": "BOWL = np.array([0.40, 0.18, 0.05])\nrobot.move_to(BOWL + [0, 0, 0.15], speed=0.06)\n",
         "moves": [("move_to", {"target": [0.40, 0.18, 0.20], "speed": 0.06}, 75)], "out": "", "async": True},
        {"look": ["cam_head", "cam_wrist"], "think": "Block is centred above the bowl; lower it inside and release.",
         "evidence": "Wrist: block centred over the bowl, about 10 cm above the rim.",
         "note": "lower and release", "scene": {"grip": (0.40, 0.18, 0.21), "block": (0.40, 0.18), "open": True},
         "code": "robot.move_to(BOWL + [0, 0, 0.06], speed=0.04)\nrobot.set_gripper(0.08)\n"
                 "robot.move_by([0, 0, 0.10], speed=0.1)\n",
         "moves": [("move_to", {"target": [0.40, 0.18, 0.11], "speed": 0.04}, 45), ("set_gripper", {"width": 0.08}, 10),
                   ("move_by", {"delta": [0, 0, 0.10]}, 20)], "out": ""},
        {"look": ["cam_head"], "think": "The block is in the bowl and the gripper is clear; finish the episode.",
         "evidence": "Head: block inside the bowl; gripper open and clear.", "note": "block placed in the bowl",
         "finish": True},
    ]
    for k, step in enumerate(steps):
        r.tick(10)
        if step.get("user"):
            r.user(step["user"])
        for cam in step.get("look", []):
            r.view(session.latest["images"][cam], (session.root / "observations"
                                                   / f"{session.latest['observation_index']:04d}" / f"{cam}.png").read_bytes())
        r.reasoning(step["think"])
        if step.get("evidence"):
            cid, resp = session.command("ack_visual", f"ack step {k}", r.t.timestamp(), evidence=step["evidence"],
                                        cameras=step["look"])
            cams = " ".join(f"--camera {c}" for c in step["look"])
            r.shell(f"{CTL} --note 'ack step {k}' ack_visual {cams} --evidence '{step['evidence']}'",
                    json.dumps(resp, indent=2))
        if step.get("finish"):
            cid, resp = session.command("finish", step["note"], r.t.timestamp())
            r.shell(f"{CTL} --note '{step['note']}' finish", json.dumps(resp, indent=2))
            continue
        path = f"/tmp/cells/step_{k}.py"
        cid, resp = session.command("python", step["note"], r.t.timestamp(), code=step["code"],
                                    moves=step.get("moves", ()), error=step.get("error"), scene=step.get("scene"),
                                    out=step.get("out", ""))
        write = f"cat > {path} <<'EOF'\n{step['code']}EOF\n"
        if step.get("async"):
            r.shell(f"{write}{CTL} --async --note '{step['note']}' cell --file {path}",
                    json.dumps({"id": cid, "pending": True}))
            r.say("Submitted asynchronously; polling for the result.")
            r.shell(f"{CTL} poll --id {cid}", json.dumps(resp, indent=2))
        elif step.get("error"):
            r.shell(f"{write}{CTL} --note '{step['note']}' cell --file {path}", json.dumps(resp, indent=2), exit_code=1)
        else:
            r.shell(f"{write}{CTL} {'--brief ' if step.get('brief') else ''}--note '{step['note']}' cell --file {path}",
                    brief(resp) if step.get("brief") else json.dumps(resp, indent=2))
    r.tick(5)
    r.say("The blue block is in the bowl and the episode is finished.", phase="final_answer")
    session.close(success=True)
    r.write(base / "input" / "codex_home" / "sessions" / "2026" / "01" / "15"
            / "rollout-2026-01-15T09-00-00-0199c0de-0000-7000-8000-00000000c001.jsonl")


# ---- demo 2: MCP robot server plus an SDK script, transcript only -------------------------------------------------

def demo_mcp(base):
    start = datetime(2026, 1, 16, 14, 0, 0, tzinfo=timezone.utc)
    r = Rollout("0199c0de-0000-7000-8000-00000000c002", start, "/workspace/pusher")
    r.user("Slide the red cube 5 cm to the right without tipping it, then send the arm home.")

    def camera(cube_y, tip):
        r.call("robot__get_camera", {"camera": "front"}, [{"type": "input_text", "text": "front camera, 128x96"},
                                                          {"type": "input_image", "image_url": "data:image/png;base64,"
                                                           + base64.b64encode(front_view(cube_y, tip)).decode()}])
    camera(0.0, (-0.10, 0.12))
    r.reasoning("Cube at y = 0.00 m. Put the fingertip against its left face first.")
    r.call("robot__move_arm", {"target": {"x": 0.45, "y": -0.035, "z": 0.02}, "speed": 0.1,
                               "reason": "fingertip against the left face of the cube"}, "reached target")
    camera(0.0, (-0.035, 0.02))
    r.reasoning("Contact made. Push 5 cm slowly so the cube slides instead of tipping.")
    r.call("robot__move_arm", {"target": {"x": 0.45, "y": 0.015, "z": 0.02}, "speed": 0.03,
                               "reason": "push 5 cm to the right, slowly"}, "reached target")
    camera(0.04, (0.015, 0.02))
    r.call("robot__get_state", {}, json.dumps({"cube_y": 0.04, "tcp": [0.45, 0.015, 0.02]}))
    r.say("The cube moved 4 cm, not 5; pushing 1 cm more.")
    r.call("robot__move_arm", {"target": {"x": 0.45, "y": 0.025, "z": 0.02}, "speed": 0.02,
                               "reason": "close the remaining 1 cm"}, "reached target")
    camera(0.05, (0.025, 0.02))
    r.reasoning("Cube at y = 0.05 m and upright. Retract and send the arm home with the lab SDK.")
    r.shell("python - <<'PY'\nfrom lab_sdk import Arm\n\narm = Arm()\n# lift off the cube before homing\n"
            "arm.move_by([0, 0, 0.10], speed=0.1)\narm.home(speed=0.2)\nPY", "homed", workdir="/workspace/pusher")
    r.say("The cube is 5 cm to the right and upright; the arm is home.", phase="final_answer")
    r.write(base / "input" / "codex_home" / "sessions" / "2026" / "01" / "16"
            / "rollout-2026-01-16T14-00-00-0199c0de-0000-7000-8000-00000000c002.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE, help="directory for input/ and output/ (default: examples/)")
    args = parser.parse_args()
    base = args.out.resolve()
    for name in ("input", "output", "sft"):
        shutil.rmtree(base / name, ignore_errors=True)
    demo_cell_queue(base)
    demo_mcp(base)
    commands = [[sys.executable, str(RCT), "--codex-home", "input/codex_home", "export", "--auto", "--out", "output",
                 "--images", "all", "--remap-path", f"{RECORDED}=input/workspace"],
                [sys.executable, str(RCT), "sft", "output", "--out", "sft"]]
    for command in commands:
        print("$ (cd", base, "&&", " ".join(command[1:]).replace(str(RCT), "rct.py") + ")", flush=True)
        code = subprocess.call(command, cwd=base)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
