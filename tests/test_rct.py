"""Offline tests: python3 -m unittest discover -s tests"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

from fixtures import ROOT, RolloutBuilder, SessionBuilder, T0, png

from rct_lib import audit, calls, codex, shell
from rct_lib import chat as chat_mod
from rct_lib import export as export_mod
from rct_lib import sft as sft_mod
from rct_lib.redact import Redactor
from rct_lib.steps import FileState, apply_update, clip
from rct_lib.validate import validate

CELL = """GRIP = np.array([0.206, -0.44, 0.838])
# approach from behind the cup
print(robo.reachable(GRIP, arm="right"))
robo.move_to(GRIP + [0, 0, 0.06], arm="right", pitch=20, speed=0.1)
robo.set_gripper("right", OPEN)
"""
CELL2 = CELL.replace("-0.44", "-0.455").replace("# approach", "# cup axis 1.5 cm nearer; approach")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(data):
    return f"images/{sha(data)[:16]}.png"


def export(paths, out, **kw):
    options = export_mod.Options(**kw)
    export_mod.export(paths, out, options, log=lambda *_: None)
    errors, totals, _ = validate(out)
    return errors, totals


def steps_of(out):
    found = []
    for folder in sorted((Path(out) / "episodes").iterdir()):
        found.append([json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines()])
    return found


class ReaderTests(unittest.TestCase):
    def test_current_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("put the block in the bowl")
            b.reasoning("the cup is on the right")
            b.assistant("Looking at the head camera first.")
            b.shell("ls", "a\nb", style="json")
            b.view_image("/home/alice/run/observations/0000/cam_head.png", png())
            b.compact("summary of earlier work")
            b.raw('{"timestamp": "2026-11-04T10:59:00Z", "type": "response_item", "payload": {"type": "mess')
            rollout = codex.read_rollout(b.write(Path(tmp) / "rollout-2026-11-04T10-00-00-x.jsonl"))
        kinds = [e.kind for e in rollout.events]
        self.assertEqual(kinds.count("user_message"), 1)
        self.assertEqual(kinds.count("reasoning"), 1)
        self.assertEqual(kinds.count("assistant_message"), 1)
        self.assertNotIn("DUPLICATE OLD MESSAGE", json.dumps([e.text for e in rollout.events]))
        self.assertEqual(rollout.stats["context_messages"], 2)
        self.assertTrue(rollout.warnings and "truncated" in rollout.warnings[0])
        ls, view = [e for e in rollout.events if e.kind == "tool_call"]
        self.assertEqual((ls.command, ls.output, ls.exit_code), ("ls", "a\nb", 0))
        image = rollout.images[view.images[0]]
        self.assertEqual((image.source, image.path_hint[-12:]), ("tool_output", "cam_head.png"))
        self.assertEqual(rollout.models, ["gpt-5-codex"])

    def test_legacy_format_and_injected_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-2025-05-01-legacy.jsonl"
            lines = [{"id": "legacy-1", "timestamp": "2025-05-01T00:00:00Z", "instructions": None},
                     {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "move arm"}]},
                     {"record_type": "state"},
                     {"type": "function_call", "name": "shell", "call_id": "c1",
                      "arguments": json.dumps({"command": ["bash", "-lc", "python move.py --x 0.1"]})},
                     {"type": "function_call_output", "call_id": "c1",
                      "output": json.dumps({"output": "moved", "metadata": {"exit_code": 0}})},
                     {"type": "function_call", "name": "view_image", "call_id": "c2",
                      "arguments": json.dumps({"path": "/tmp/cam.png"})},
                     {"type": "function_call_output", "call_id": "c2", "output": "attached"},
                     {"type": "message", "role": "user", "content": [
                         {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}]}]
            path.write_text("\n".join(json.dumps(l) for l in lines))
            rollout = codex.read_rollout(path)
        self.assertEqual(rollout.thread_id, "legacy-1")
        calls_ = [e for e in rollout.events if e.kind == "tool_call"]
        self.assertEqual(calls_[0].command, "python move.py --x 0.1")
        self.assertEqual(calls_[0].exit_code, 0)
        self.assertEqual(len(calls_[1].images), 1, "image injected after view_image belongs to that call")
        self.assertEqual([e.text for e in rollout.events if e.kind == "user_message"], ["move arm"])

    def test_interactive_session_output_is_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("python long_run.py", "starting", style="running")
            b.stdin(7, "done: reached", exit_code=0)
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        first, second = [e for e in rollout.events if e.kind == "tool_call"]
        self.assertIn("done: reached", first.output)
        self.assertEqual((first.exit_code, second.continuation_of), (0, first.seq))
        self.assertEqual((first.own_output, first.own_exit_code), ("starting", None))

    def test_injected_images_follow_their_calls(self):
        head, wrist, late = png((10, 0, 0)), png((0, 10, 0)), png((0, 0, 10))
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("look")
            gone = b.view_call("/home/alice/run/missing.png")
            first = b.view_call("/home/alice/run/cam_head.png")
            second = b.view_call("/home/alice/run/cam_wrist.png")
            b.raw_output(gone, "unable to locate image at `/home/alice/run/missing.png`: No such file or directory")
            b.raw_output(first, "attached local image path").raw_output(second, "attached local image path")
            b.image_message(head).image_message(wrist)
            b.reasoning("both views checked")
            side = b.view_call("/home/alice/run/cam_side.png")
            b.raw_output(side, "attached local image path")
            b.reasoning("the side view did not arrive")
            b.image_message(late)
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        calls = {e.call_id: e for e in rollout.events if e.kind == "tool_call"}
        self.assertEqual([calls[c].images for c in (gone, first, second, side)],
                         [[], [sha(head)], [sha(wrist)], []])
        self.assertEqual([e.images for e in rollout.events if e.kind == "image"], [[sha(late)]])

    def test_user_images_stay_with_their_message(self):
        photo = png((1, 2, 3))
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("put it here", image=photo)
            b.user("and this one", url="https://example.com/cup.png")
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        first, second = [e for e in rollout.events if e.kind == "user_message"]
        self.assertEqual((first.text, first.images), ("put it here", [sha(photo)]))
        self.assertEqual(len(second.images), 1, "an image without recorded bytes is kept as missing, not dropped")
        self.assertIsNone(rollout.images[second.images[0]].data)
        self.assertFalse([e for e in rollout.events if e.kind == "image"])

    def test_interleaved_output_keeps_order(self):
        head, wrist, frame = png((10, 0, 0)), png((0, 10, 0)), png((0, 0, 10))
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("look")
            b.mcp("get_cameras", {}, parts=["head:", head, "wrist:", wrist], namespace="mcp__robot")
            b.mcp("observe", {}, "frame", image=frame, namespace="mcp__robot")
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        cams, observe = [e for e in rollout.events if e.kind == "tool_call"]
        self.assertEqual(cams.output_parts, [("text", "head:"), ("image", sha(head)), ("text", "wrist:"),
                                             ("image", sha(wrist))])
        self.assertEqual((cams.output, cams.images), ("head:\nwrist:", [sha(head), sha(wrist)]))
        self.assertIsNone(observe.output_parts, "text then images is already unambiguous")

    def test_record_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("go")
            a = b.call("python robot_ctl.py move --x 1")
            c = b.call("python robot_ctl.py move --x 2")
            b.output(a, "moved 1")
            b.reasoning("first one is done")
            b.output(c, "moved 2")
            b.shell("ls", "a")
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        user, first, second, reasoning, ls = rollout.events
        self.assertEqual((user.kind, reasoning.kind), ("user_message", "reasoning"))
        self.assertLess(user.raw, first.raw)
        self.assertLess(second.raw, first.output_raw)
        self.assertLess(first.output_raw, reasoning.raw)
        self.assertLess(reasoning.raw, second.output_raw)
        self.assertLess(ls.raw, ls.output_raw)
        self.assertFalse(first.continued)

    def test_mirrored_context_is_not_typed_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context(mirrored=True).user("go")
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        self.assertEqual([e.text for e in rollout.events if e.kind == "user_message"], ["go"])

    def test_mcp_images_in_native_format(self):
        frames = [png((i, 0, 0)) for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("look")
            for frame, native in zip(frames, ("text", "object", "event")):
                b.mcp("get_camera", {}, "head", image=frame, namespace="mcp__robot", native=native)
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        cams = [e for e in rollout.events if e.kind == "tool_call"]
        self.assertEqual([(c.output, c.images) for c in cams], [("head", [sha(f)]) for f in frames])
        self.assertEqual([rollout.images[sha(f)].data for f in frames], frames)

    def test_long_compaction_summary_is_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go").compact("s" * 6000)
            rollout = codex.read_rollout(b.write(Path(tmp) / "r.jsonl"))
        self.assertEqual([len(e.text) for e in rollout.events if e.kind == "compaction"], [6000])


class ShellTests(unittest.TestCase):
    def test_heredoc_write_and_skeleton(self):
        script = "cat > /tmp/cell.py <<'EOF'\nprint('cell')\nEOF\npython ctl.py --note 'look at cell' cell --file /tmp/cell.py"
        self.assertEqual(shell.file_writes(script), [("/tmp/cell.py", "print('cell')\n", "heredoc")])
        self.assertNotIn("look at", shell.skeleton(script))
        self.assertEqual(shell.option_values(script, ["--note"]), {"--note": ["look at cell"]})
        self.assertEqual(shell.signature(script), "python ctl.py cell")
        self.assertEqual(shell.signature("ros2 topic pub --once /cmd_vel geometry_msgs/Twist '{x: 1}'"),
                         "ros2 topic pub")
        self.assertEqual(shell.signature("python - <<'PY'\narm.home()\nPY"), "python -")
        self.assertEqual(shell.referenced_files(script, ["--file"])[0], "/tmp/cell.py")

    def test_inline_code(self):
        self.assertEqual(shell.inline_code("cd x && python3 - <<'PY'\nrobo.home()\nPY")[0], "robo.home()")
        self.assertEqual(shell.inline_code("python -c 'robo.wait(3)'")[0], "robo.wait(3)")
        self.assertEqual(shell.inline_code("python script.py <<EOF\ndata\nEOF")[0], None)

    def test_apply_patch(self):
        add = "*** Begin Patch\n*** Add File: cell.py\n+a = 1\n+robo.wait(a)\n*** End Patch"
        self.assertEqual(shell.patch_writes(add), [("cell.py", "a = 1\nrobo.wait(a)\n", "apply_patch_add")])
        update = shell.patch_writes("*** Begin Patch\n*** Update File: cell.py\n@@\n-a = 1\n+a = 2\n robo.wait(a)\n*** End Patch")
        self.assertEqual(apply_update("a = 1\nrobo.wait(a)\n", update[0][1]), "a = 2\nrobo.wait(a)\n")
        files = FileState()
        files.write("/w/cell.py", update[0][1], "apply_patch_update", 1)
        self.assertIsNone(files.lookup("cell.py", "/w"), "a diff against unknown content is not executable code")


class CallTests(unittest.TestCase):
    def test_extract_and_adjust(self):
        pattern = __import__("re").compile(r"^robo\.(move_to|set_gripper)$")
        first, _ = calls.extract(CELL, pattern, {"OPEN": 1.0})
        second, _ = calls.extract(CELL2, pattern, {"OPEN": 1.0})
        self.assertEqual(first[0]["args"][0], [0.206, -0.44, 0.898])
        self.assertEqual(first[1]["args"], ["right", 1.0])
        tracker = calls.AdjustmentTracker()
        self.assertEqual(tracker.step(0, first), [])
        (adjustment,) = tracker.step(1, second)
        self.assertEqual(adjustment["changes"]["args[0]"]["delta"], [0.0, -0.015, 0.0])
        self.assertEqual(calls.comments(CELL2), ["cup axis 1.5 cm nearer; approach from behind the cup"])

    def test_unknown_values_stay_symbolic(self):
        found, _ = calls.extract("robo.move_to(target(), arm=side)", __import__("re").compile("move_to"), {})
        self.assertEqual(found[0]["args"], [{"expr": "target()"}])
        self.assertEqual(found[0]["kwargs"], {"arm": {"expr": "side"}})


class CleanupTests(unittest.TestCase):
    def test_redaction(self):
        r = Redactor(True)
        text = r.text('{"stdout": "ok\\n/mnt/home/alice/run/a.png /srv/home/x OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv '
                      'mail bob@example.com C:\\\\Users\\\\bob\\\\x"}')
        self.assertIn("ok\\n~/run/a.png", text)
        self.assertIn("/srv/home/x", text, "only real home roots are rewritten")
        self.assertNotIn("sk-abc", text)
        self.assertNotIn("bob", text)

    def test_paths_next_to_cjk_volumes_and_private_ips(self):
        text = Redactor(True).text("代码放在/home/carol/data，清单：/Volumes/backup1t/notes/Q.md，"
                                   "打开http://172.20.31.7/或 10.0.0.5:8080，版本 10.0.0")
        for leaked in ("carol", "backup1t", "172.20.31.7", "10.0.0.5"):
            self.assertNotIn(leaked, text)
        self.assertIn("代码放在~/data", text)
        self.assertIn("版本 10.0.0", text, "three-part version numbers are not addresses")

    def test_settings_and_placeholders_are_not_secrets(self):
        r = Redactor(True)
        kept = ("judge 默认 `max_tokens=4096`，", "num_tokens: 128000", "export ZAI_API_KEY=...", "API_KEY=$ZAI_KEY",
                "client(api_key=args.api_key)", "token = os.environ.get(", "PASSWORD=<your password>")
        for text in kept:
            self.assertEqual(r.text(text), text)
        self.assertEqual(r.text("`ZAI_API_KEY=3f2a9c1d.Xy7Qk2`"), "`ZAI_API_KEY=<redacted>`")
        self.assertEqual(r.obj({"api_key": "3f2a9c1d.Xy7Qk2", "max_tokens": "4096", "token": "$T", "Secrets": "use env"}),
                         {"api_key": "<redacted>", "max_tokens": "4096", "token": "$T", "Secrets": "use env"})
        self.assertEqual(r.counts, {"assignment": 1, "secret_field": 1})

    def test_progress_output(self):
        self.assertEqual(clip("step 1/9\rstep 2/9\rstep 9/9\ndone", 0), "step 9/9\ndone")
        raw = json.dumps({"stdout": "\x1b[92mstep 1\x1b[0m\rstep 2\nok"})
        self.assertEqual(json.loads(clip(raw, 0)), {"stdout": "step 2\nok"})


class GenericExportTests(unittest.TestCase):
    """No framework adapter: everything comes from the transcript."""

    def build(self, tmp):
        b = RolloutBuilder().context().user("Pick up the cup. My key is sk-abcdefghijklmnopqrstuvwx")
        b.view_image("/home/alice/run/cam_head.png", png((1, 2, 3)))
        b.reasoning("cup at right front; approach from behind")
        b.shell(f"cat > /tmp/cell.py <<'EOF'\n{CELL}EOF\npython /tmp/cell.py --note 'pre-grasp'", "moved")
        b.view_image("/home/alice/run/cam_wrist.png", png((4, 5, 6)))
        b.reasoning("cup axis is 1.5 cm nearer than planned")
        b.apply_patch("*** Begin Patch\n*** Update File: /tmp/cell.py\n@@\n-GRIP = np.array([0.206, -0.44, 0.838])\n"
                      "+GRIP = np.array([0.206, -0.455, 0.838])\n*** End Patch")
        b.shell("python /tmp/cell.py", "Traceback: blocked", exit_code=1)
        b.mcp("robot__move_arm", {"x": 0.1, "y": -0.2, "arm": "left", "reason": "clear the mug"})
        b.mcp("robot__get_camera", {"name": "head"}, "frame", image=png((9, 9, 9)))
        b.mcp("robot__move_arm", {"x": 0.12, "y": -0.2, "arm": "left", "reason": "slightly further"})
        b.user("stop now")
        b.assistant("Done; the cup is held.", phase="final_answer")
        return b.write(Path(tmp) / "sessions" / "2026" / "11" / "04" / "rollout-2026-11-04T10-00-00-a.jsonl")

    def test_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build(tmp)
            errors, totals = export([path], Path(tmp) / "out")
            self.assertEqual(errors, [])
            (steps,) = steps_of(Path(tmp) / "out")
            episode = json.loads(next((Path(tmp) / "out" / "episodes").iterdir()).joinpath("episode.json").read_text())
            blob = (Path(tmp) / "out").joinpath("episodes").glob("*/*.json*")
            text = "".join(p.read_text() for p in blob)
        self.assertEqual(len(steps), 4)
        first, second, third, fourth = steps
        self.assertEqual(first["observation"]["images"][0]["file"][:7], "images/")
        self.assertEqual(first["analysis"]["reasoning"], ["cup at right front; approach from behind"])
        self.assertEqual(first["analysis"]["rationale"][0]["text"], "pre-grasp")
        self.assertEqual([c["func"] for c in first["action"]["calls"]], ["robo.move_to", "robo.set_gripper"])
        self.assertEqual(second["action"]["code_source"], "apply_patch:/tmp/cell.py")
        self.assertEqual((first["observation"]["prior_feedback"], second["observation"]["prior_feedback"]["step"]),
                         (None, 0))
        self.assertEqual(second["action"]["adjustments"][0]["changes"]["args[0]"]["delta"], [0.0, -0.015, 0.0])
        self.assertTrue(second["flags"]["action_failed"])
        self.assertEqual(third["action"]["calls"][0]["kwargs"]["x"], 0.1)
        self.assertEqual(fourth["observation"]["reads"][0]["signature"], "robot__get_camera")
        self.assertEqual(len(fourth["observation"]["images"]), 1)
        self.assertEqual(fourth["action"]["adjustments"][0]["changes"]["x"]["delta"], 0.02)
        self.assertEqual(episode["final"]["messages"], ["Done; the cup is held."])
        self.assertEqual(episode["final"]["user_messages"], ["stop now"])
        for secret in ("sk-abcdefghijklmnop", "/home/alice", "SECRET SYSTEM PROMPT", "alice/private.git"):
            self.assertNotIn(secret, text)
        self.assertEqual(totals["with_images"], 3)

    def test_profile_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("mybot send --joint 3 --angle 10", "ok")
            b.shell("mybot send --joint 3 --angle 12", "ok")
            path = b.write(Path(tmp) / "r.jsonl")
            out = Path(tmp) / "none"
            export_mod.export([path], out, export_mod.Options(), log=lambda *_: None)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["episodes"], [])
            spec = Path(tmp) / "mybot.json"
            spec.write_text(json.dumps({"name": "mybot", "rules": [
                {"role": "act", "kind": "shell", "command": r"\bmybot send\b"}]}))
            errors, _ = export([path], Path(tmp) / "out", profiles=(str(spec),))
            (steps,) = steps_of(Path(tmp) / "out")
        self.assertEqual(errors, [])
        self.assertEqual(steps[1]["action"]["adjustments"][0]["changes"]["--angle"]["delta"], 2.0)

    def test_sessions_with_the_same_id_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for n in (1, 2):  # uuid7 ids of sessions started within a minute share their first 8 characters
                b = RolloutBuilder(thread=str(uuid.UUID(int=n))).user("go")
                b.shell(f"robot_cli move_to --x 0.{n}", "ok")
                paths.append(b.write(Path(tmp) / f"r{n}.jsonl"))
            errors, _ = export(paths, Path(tmp) / "out")
            folders = sorted(p.name for p in (Path(tmp) / "out" / "episodes").iterdir())
        self.assertEqual(errors, [])
        self.assertEqual(folders, ["2026-11-04_00000000_00", "2026-11-04_00000000_2_00"])


class ControlEvidenceTests(unittest.TestCase):
    """Failure modes seen in a published dataset built with this exporter."""

    def test_agent_infrastructure_is_not_robot_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("我们先复现 cycle vla？")
            b.reasoning("先查本地是否已有 CycleVLA 相关代码")
            b.mcp("send_message", {"target": "worker", "message": "list the repo"}, "sent", namespace="collaboration")
            b.mcp("close_agent", {"id": "w1"}, '{"previous_status": {"completed": "结论：..."}}',
                  namespace="multi_agent_v1")
            b.mcp("open_in_codex", {"path": "/Volumes/backup1t/x.pptx"}, "opened", namespace="mcp__codex_app")
            b.mcp("create_pull_request", {"title": "fix"}, "ok", namespace="mcp__github")
            b.mcp("move_file", {"src": "a", "dst": "b"}, "ok", namespace="mcp__filesystem")
            b.shell("python eval_policy.py --config cfg.yaml", "success rate 0.4")
            path = b.write(Path(tmp) / "r.jsonl")
            manifest = export_mod.export([path], Path(tmp) / "out", export_mod.Options(), log=lambda *_: None)
        self.assertEqual(manifest["episodes"], [])
        self.assertIn("act steps", manifest["sources"][0]["skipped"])

    def test_robot_mcp_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("把两个碗叠起来")
            b.mcp("task_demos", {"query": "stack bowls"}, "没有匹配的任务。", namespace="mcp__robot")
            b.mcp("observe", {}, "fingertips L (210, 300)", image=png((5, 5, 5)), namespace="mcp__robot")
            b.mcp("send_message", {"target": "critic", "message": "check"}, "sent", namespace="collaboration")
            b.reasoning("the small bowl rim is at (320, 240)")
            b.mcp("execute_skill", {"left": {"skill": "grasp", "object": "bowl", "target_point": [320, 240]},
                                    "seconds": 8, "reason": "grasp the rim"}, "termination: done",
                  namespace="mcp__robot")
            b.mcp("zoom_head", {"u": 320, "v": 240}, "zoomed", image=png((6, 6, 6)), namespace="mcp__robot")
            b.mcp("execute_skill", {"left": {"skill": "grasp", "object": "bowl", "target_point": [326, 240]},
                                    "seconds": 8, "reason": "rim is 6 px further right"}, "termination: done",
                  namespace="mcp__robot")
            path = b.write(Path(tmp) / "r.jsonl")
            errors, _ = export([path], Path(tmp) / "out")
            (steps,) = steps_of(Path(tmp) / "out")
        self.assertEqual(errors, [])
        self.assertEqual([s["action"]["signature"] for s in steps], ["mcp__robot.execute_skill"] * 2)
        first, second = steps
        self.assertEqual([r["signature"] for r in first["observation"]["reads"]],
                         ["mcp__robot.task_demos", "mcp__robot.observe"])
        self.assertEqual(first["action"]["calls"][0]["kwargs"]["left.target_point"], [320, 240])
        self.assertEqual(second["observation"]["reads"][0]["signature"], "mcp__robot.zoom_head")
        self.assertEqual(second["action"]["adjustments"][0]["changes"]["left.target_point"]["delta"], [6, 0])
        self.assertTrue(first["flags"]["control"])


class InteractiveInputTests(unittest.TestCase):
    """write_stdin with nothing typed polls a running command; typed text is a call of its own."""

    def test_command_typed_into_a_robot_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("move from the robot shell")
            shell_id = b.shell("robot_cli repl", "robot> ", style="running")
            typed = b.stdin(7, "robot> moving", chars="move_to 0.3 0.1 0.2\n", running=True)
            poll = b.stdin(7, "robot> reached", running=True)
            path = b.write(Path(tmp) / "r.jsonl")
            out = Path(tmp) / "chat"
            manifest = chat_mod.export([path], out, chat_mod.Options(), log=lambda *_: None)
            lines = [json.loads(l) for l in next((out / "sessions").iterdir()).joinpath("rounds.jsonl").read_text()
                     .splitlines()]
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual([l["actions"] for l in lines], [[shell_id], [typed]])
        records = {(e["call_id"], e["category"]): e for l in lines for e in l["events"] if "call_id" in e}
        self.assertEqual(records[typed, "action"]["args"]["chars"], "move_to 0.3 0.1 0.2\n")
        self.assertEqual(records[typed, "feedback"]["text"], "robot> moving")
        self.assertEqual((records[poll, "feedback"]["action"], records[poll, "feedback"]["text"]),
                         (typed, "robot> reached"))
        self.assertEqual(records[shell_id, "feedback"]["text"], "robot> ")

    def test_robot_calls_typed_into_a_python_repl(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("move from python")
            b.shell("python3 -i", ">>> ", style="running", session=9)
            b.stdin(9, "moving", chars="robo.move_to([0.2, 0, 0.3])\n", running=True)
            b.stdin(9, "reached", running=True)
            path = b.write(Path(tmp) / "r.jsonl")
            errors, _ = export([path], Path(tmp) / "out")
            (steps,) = steps_of(Path(tmp) / "out")
        self.assertEqual(errors, [])
        (step,) = steps
        self.assertEqual((step["action"]["tool"], step["action"]["code"]), ("write_stdin", "robo.move_to([0.2, 0, 0.3])\n"))
        self.assertEqual([c["func"] for c in step["action"]["calls"]], ["robo.move_to"])
        self.assertTrue(step["flags"]["control"])
        self.assertEqual(step["outcome"]["output"], "moving\nreached")

    def test_polled_output_is_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("python -c 'import robo; robo.move_to([0.2, 0, 0.3])'", "starting", style="running")
            b.stdin(7, "done: reached")
            errors, _ = export([b.write(Path(tmp) / "r.jsonl")], Path(tmp) / "out")
            text = next((Path(tmp) / "out" / "episodes").iterdir()).joinpath("steps.jsonl").read_text()
        self.assertEqual(errors, [])
        self.assertEqual(text.count("done: reached"), 1)


class ChatTests(unittest.TestCase):
    def build(self, tmp):
        b = RolloutBuilder().context().user("把碗放到盘子里", image=png((1, 1, 1)))
        b.view_image("/home/alice/run/cam_head.png", png((2, 2, 2)))
        b.reasoning("bowl at left; approach from above")
        b.shell("robot_cli move_to --x 0.1 --note 'reach over the bowl'", "reached")
        b.reasoning("close both grippers")
        left = b.call("robot_cli set_gripper --arm left --width 0.02")
        right = b.call("robot_cli set_gripper --arm right --width 0.02")
        b.output(left, "left closed")
        b.reasoning("left done, waiting for right")
        b.output(right, "right closed", image=png((3, 3, 3)))
        b.shell("ls /home/alice/run", "cam_head.png")
        b.compact("summary of earlier work")
        b.shell("python -c 'import robo; robo.move_to([0.2, 0, 0.3])'", "starting", style="running")
        b.reasoning("still moving")
        b.stdin(7, "done: reached")
        b.user("stop now")
        b.assistant("Done.", phase="final_answer")
        return b.write(Path(tmp) / "rollout-2026-11-04T10-00-00-a.jsonl")

    def export(self, tmp, paths, **kw):
        out = Path(tmp) / "chat"
        manifest = chat_mod.export(paths, out, chat_mod.Options(**kw), log=lambda *_: None)
        return out, manifest

    def rounds(self, out):
        (folder,) = (out / "sessions").iterdir()
        return folder, [json.loads(l) for l in (folder / "rounds.jsonl").read_text().splitlines()]

    def test_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, manifest = self.export(tmp, [self.build(tmp)])
            folder, rounds = self.rounds(out)
            session = json.loads((folder / "session.json").read_text())
            errors = chat_mod.check(out)
            text = "".join(p.read_text() for p in out.rglob("*.json*"))
            images = sorted(p.name for p in (folder / "images").iterdir())
        self.assertEqual(errors, [])
        self.assertEqual(session["task"], "把碗放到盘子里")
        self.assertEqual([r["round"] for r in rounds], [0, 1, 2, "final"])
        cats = [[e["category"] for e in r["events"]] for r in rounds]
        self.assertEqual(cats[0], ["task", "observation", "analysis", "analysis", "action", "feedback"])
        self.assertEqual(cats[1], ["analysis", "action", "action", "feedback", "analysis", "feedback"])
        self.assertEqual(cats[2], ["other", "other", "action", "feedback", "analysis", "feedback"])
        self.assertEqual(cats[3], ["task", "analysis"])
        first, second, third, final = [r["events"] for r in rounds]
        self.assertEqual((first[0]["text"], first[0]["images"]), ("把碗放到盘子里", [ref(png((1, 1, 1)))]))
        self.assertEqual((first[1]["tool"], first[1]["images"]), ("view_image", [ref(png((2, 2, 2)))]))
        self.assertEqual(first[2]["text"], "bowl at left; approach from above")
        note, act, feedback = first[3:]
        self.assertEqual((note["kind"], note["text"], note["call_id"], note["seq"]),
                         ("--note", "reach over the bowl", act["call_id"], act["seq"]))
        self.assertEqual(act["command"], "robot_cli move_to --x 0.1 --note 'reach over the bowl'")
        self.assertEqual((feedback["call_id"], feedback["action"], feedback["text"], feedback["exit_code"]),
                         (act["call_id"], act["call_id"], "reached", 0))
        self.assertEqual(rounds[1]["actions"], [second[1]["call_id"], second[2]["call_id"]])
        self.assertEqual([e.get("text") for e in second[3:]], ["left closed", "left done, waiting for right",
                                                               "right closed"])
        self.assertEqual(second[5]["action"], second[2]["call_id"])
        self.assertEqual(second[5]["images"], [ref(png((3, 3, 3)))])
        self.assertTrue(rounds[2]["context_reset"])
        self.assertFalse(rounds[1]["context_reset"])
        self.assertEqual(third[1]["kind"], "compaction")
        self.assertNotIn("code", third[2], "inline code is already in the command")
        self.assertEqual((third[3]["text"], third[5]["text"], third[5]["tool"]), ("starting", "done: reached",
                                                                                 "write_stdin"))
        self.assertEqual(third[5]["action"], third[2]["call_id"])
        self.assertEqual((final[0]["text"], final[1]["text"], final[1]["phase"]), ("stop now", "Done.",
                                                                                   "final_answer"))
        seqs = [e["seq"] for r in rounds for e in r["events"] if not (e["category"] == "analysis" and "call_id" in e)]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))
        self.assertEqual(len(images), 3)
        for secret in ("/home/alice", "SECRET SYSTEM PROMPT", "environment_context", "permissions instructions"):
            self.assertNotIn(secret, text)
        self.assertEqual(manifest["sessions"][0]["rounds"], 4)

    def test_images_stay_with_their_content(self):
        photo, head, wrist, head2, wrist2 = (png((1, 2, 3)), png((10, 0, 0)), png((0, 10, 0)), png((20, 0, 0)),
                                             png((0, 20, 0)))
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().context().user("put it here", image=photo)
            b.user("and this one", url="https://example.com/cup.png")
            first = b.view_call("/home/alice/run/cam_head.png")
            second = b.view_call("/home/alice/run/cam_wrist.png")
            b.raw_output(first, "attached local image path").raw_output(second, "attached local image path")
            b.image_message(head).image_message(wrist)
            b.mcp("get_cameras", {}, parts=["head:", head2, "wrist:", wrist2], namespace="mcp__robot")
            b.shell("robot_cli move_to --x 0.1", "ok")
            path = b.write(Path(tmp) / "r.jsonl")
            out, manifest = self.export(tmp, [path])
            folder, rounds = self.rounds(out)
            events = rounds[0]["events"]
            tasks = [e for e in events if e["category"] == "task"]
            views = [e for e in events if e.get("tool") == "view_image"]
            cams = next(e for e in events if e.get("tool") == "mcp__robot.get_cameras")
            views[0]["images"], views[1]["images"] = views[1]["images"], views[0]["images"]
            (folder / "rounds.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rounds))
            swapped = chat_mod.check(out), chat_mod.crosscheck(path, folder)
        self.assertEqual(manifest["check"]["first"], [])
        self.assertEqual([(t["text"], t["images"]) for t in tasks], [("put it here", [ref(photo)]),
                                                                     ("and this one", [None])])
        self.assertEqual([v["call_id"] for v in views], [first, second])
        self.assertEqual([v["images"] for v in views], [[ref(wrist)], [ref(head)]], "after the swap")
        self.assertEqual(cams["parts"], [{"text": "head:"}, {"image": ref(head2)}, {"text": "wrist:"},
                                         {"image": ref(wrist2)}])
        self.assertEqual((cams["text"], cams["images"]), ("head:\nwrist:", [ref(head2), ref(wrist2)]))
        self.assertEqual(swapped[0], [], "file names still match their content")
        self.assertTrue(any(first in e for e in swapped[1]), swapped[1])

    def test_gate_and_missing_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("list files")
            b.shell("ls", "a")
            plain = b.write(Path(tmp) / "plain.jsonl")
            b = RolloutBuilder(thread=str(uuid.UUID(int=2))).user("move")
            b.call("robot_cli move_to --x 0.3")
            cut = b.write(Path(tmp) / "cut.jsonl")
            out, manifest = self.export(tmp, [plain, cut])
            folder, rounds = self.rounds(out)
            session = json.loads((folder / "session.json").read_text())
            errors = chat_mod.check(out)
        self.assertIn("robot-control evidence", manifest["sessions"][0]["skipped"])
        self.assertEqual(errors, [])
        self.assertEqual([r["round"] for r in rounds], [0])
        self.assertEqual(session["actions_without_feedback"], [
            {"call_id": rounds[0]["actions"][0], "reason": "no output in the recording"}])

    def test_check_finds_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self.export(tmp, [self.build(tmp)])
            folder, rounds = self.rounds(out)
            next((folder / "images").iterdir()).write_bytes(png((7, 7, 7)))
            rounds[1]["events"] = [e for e in rounds[1]["events"] if e.get("text") != "left closed"]
            (folder / "rounds.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rounds))
            errors = chat_mod.check(out)
        self.assertTrue(any("does not match its content" in e for e in errors), errors)
        self.assertTrue(any("without feedback" in e for e in errors), errors)
        self.assertTrue(any("tool_output" in e for e in errors), errors)

    def events(self, out):
        return [e for r in self.rounds(out)[1] for e in r["events"]]

    def test_code_from_an_earlier_file_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("cat > /tmp/cell.py <<'EOF'\nimport robo\nrobo.move_to([0.2, 0, 0.3])\nEOF", "")
            b.shell("python /tmp/cell.py", "moved")
            out, _ = self.export(tmp, [b.write(Path(tmp) / "r.jsonl")])
            (act,) = [e for e in self.events(out) if e["category"] == "action"]
        self.assertEqual((act["command"], act["code"]), ("python /tmp/cell.py",
                                                         "import robo\nrobo.move_to([0.2, 0, 0.3])\n"))

    def test_terminal_control_characters_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("robot_cli move_to --x 0.1", "\x1b[32mok\x1b[0m\n" + "".join(f"\r{i}%" for i in (0, 50, 100)))
            out, _ = self.export(tmp, [b.write(Path(tmp) / "r.jsonl")])
            (feedback,) = [e for e in self.events(out) if e["category"] == "feedback"]
        self.assertEqual(feedback["text"], "ok\n100%")

    def test_native_mcp_images_are_checked(self):
        frame = png((4, 4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("look")
            cam = b.mcp("get_camera", {}, "head", image=frame, namespace="mcp__robot", native="text")
            b.shell("robot_cli move_to --x 0.1", "ok")
            path = b.write(Path(tmp) / "r.jsonl")
            out, manifest = self.export(tmp, [path])
            folder, rounds = self.rounds(out)
            record = next(e for e in rounds[0]["events"] if e.get("call_id") == cam)
            seen = dict(record)
            record["images"] = []
            (folder / "rounds.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rounds))
            dropped = chat_mod.crosscheck(path, folder)
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual((seen["text"], seen["images"]), ("head", [ref(frame)]))
        self.assertTrue(any(cam in e for e in dropped), dropped)

    def test_copies_are_exported_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build(tmp)
            lines = path.read_text().splitlines(keepends=True)
            early = Path(tmp) / "early.jsonl"
            early.write_text("".join(lines[:12]) + lines[12][:30])  # copied while the session was still running
            same = Path(tmp) / "same.jsonl"
            same.write_bytes(path.read_bytes())
            out, manifest = self.export(tmp, [early, path, same])
            steps = export_mod.export([early, path, same], Path(tmp) / "steps", export_mod.Options(),
                                      log=lambda *_: None)
        self.assertEqual([s.get("id") is not None for s in manifest["sessions"]], [False, True, False])
        self.assertEqual({s.get("skipped") for s in manifest["sessions"] if "skipped" in s}, {f"copy of {path.name}"})
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual(len({e["rollout"] for e in steps["episodes"]}), 1)

    def test_missing_rollout_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            gone = Path(tmp) / "gone.jsonl"
            out, manifest = self.export(tmp, [gone, self.build(tmp)])
            steps = export_mod.export([gone], Path(tmp) / "steps", export_mod.Options(), log=lambda *_: None)
        self.assertIn("parse error", manifest["sessions"][0]["skipped"])
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertIn("parse error", steps["sources"][0]["skipped"])

    def test_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, manifest = self.export(tmp, [self.build(tmp)], images="none")
            refs = [r for e in self.events(out) for r in e.get("images", [])]
            saved = list(out.rglob("images"))
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual((len(refs), set(refs), saved), (3, {None}, []))

    def test_max_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("robot_cli status", "x" * 500)
            out, manifest = self.export(tmp, [b.write(Path(tmp) / "r.jsonl")], max_chars=100)
            (feedback,) = [e for e in self.events(out) if e["category"] == "feedback"]
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertIn("400 chars cut", feedback["text"])
        self.assertLess(len(feedback["text"]), 130)

    def test_poll_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("pour")
            b.shell("python robot_ctl.py ack_visual --camera cam_head --evidence 'cup on the right'", "acked")
            act = b.shell("python robot_ctl.py --async --note 'tilt' cell --file /tmp/c.py", '{"pending": true}')
            poll = b.shell("python robot_ctl.py poll --id 1", '{"done": true}')
            out, manifest = self.export(tmp, [b.write(Path(tmp) / "r.jsonl")])
            events = self.events(out)
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual([(e["category"], e.get("kind")) for e in events],
                         [("task", None), ("analysis", "--evidence"), ("observation", None), ("analysis", "--note"),
                          ("action", None), ("feedback", None), ("feedback", None)])
        self.assertEqual((events[1]["text"], events[1]["call_id"]), ("cup on the right", events[2]["call_id"]))
        self.assertEqual((events[6]["call_id"], events[6]["action"], events[6]["text"]), (poll, act, '{"done": true}'))

    def test_feedback_after_the_next_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            slow = b.shell("python -c 'import robo; robo.move_to([0.2, 0, 0.3])'", "starting", style="running")
            fast = b.shell("robot_cli set_gripper --width 0.02", "closed")
            b.stdin(7, "done: reached")
            out, manifest = self.export(tmp, [b.write(Path(tmp) / "r.jsonl")])
            folder, rounds = self.rounds(out)
        self.assertEqual(manifest["check"]["errors"], 0)
        self.assertEqual([r["actions"] for r in rounds], [[slow, fast]])
        self.assertEqual([(e.get("text"), e.get("action")) for e in rounds[0]["events"] if e["category"] == "feedback"],
                         [("starting", slow), ("closed", fast), ("done: reached", slow)])

    def test_output_directory_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.build(tmp)
            out, _ = self.export(tmp, [path])
            with self.assertRaises(SystemExit):
                chat_mod.export([path], out, chat_mod.Options(), log=lambda *_: None)
            chat_mod.export([path], out, chat_mod.Options(), overwrite=True, log=lambda *_: None)
            with self.assertRaises(SystemExit):
                export_mod.export([path], out, export_mod.Options(), overwrite=True, log=lambda *_: None)
            steps = Path(tmp) / "steps"
            export_mod.export([path], steps, export_mod.Options(), log=lambda *_: None)
            with self.assertRaises(SystemExit):
                chat_mod.export([path], steps, chat_mod.Options(), overwrite=True, log=lambda *_: None)
            self.assertEqual(chat_mod.check(out), [])

    def test_command_line(self):
        rct = ROOT / "skills" / "robot-control-traces" / "scripts" / "rct.py"
        with tempfile.TemporaryDirectory() as tmp:
            path, out = self.build(tmp), Path(tmp) / "chat"
            run = [sys.executable, str(rct), "chat", "--rollout", str(path), "--out", str(out)]
            first = subprocess.run(run, capture_output=True, text=True)
            again = subprocess.run(run, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("check: 0 error(s)", first.stdout)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("not empty", again.stderr)


class SftTests(unittest.TestCase):
    def build(self, tmp, **kw):
        path = GenericExportTests.build(None, tmp)
        export([path], Path(tmp) / "out")
        manifest = sft_mod.build(Path(tmp) / "out", Path(tmp) / "sft", sft_mod.SftOptions(**kw))
        rows = [json.loads(l) for l in (Path(tmp) / "sft" / "train.jsonl").read_text().splitlines()]
        return manifest, rows, sft_mod.check(Path(tmp) / "sft")

    def test_step_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, rows, errors = self.build(tmp)
            info = json.loads((Path(tmp) / "sft" / "dataset_info.json").read_text())
        self.assertEqual(errors, [])
        self.assertEqual(manifest["counts"]["samples"], 4)
        self.assertEqual(info["robot_control_sft"]["columns"], {"messages": "messages", "images": "images"})
        first, second, third, fourth = rows
        prompt, target = first["messages"][1]["content"], first["messages"][2]["content"]
        self.assertEqual((prompt.count("<image>"), len(first["images"])), (1, 1))
        self.assertTrue(target.startswith("cup at right front; approach from behind"))
        self.assertIn("robo.move_to(GRIP + [0, 0, 0.06]", target)
        self.assertNotIn("sk-abcdefghijklmnop", prompt)
        prompt, target = second["messages"][1]["content"], second["messages"][2]["content"]
        self.assertIn("File `/tmp/cell.py`", target, "code written by an earlier patch is part of the target")
        self.assertIn("-0.455", target)
        self.assertEqual(prompt.count("moved"), 2, "previous result once in the summary, once in full")
        self.assertTrue(second["meta"]["action_failed"])
        self.assertIn('"reason": "clear the mug"', third["messages"][2]["content"])
        self.assertIn("Traceback: blocked", third["messages"][1]["content"])
        self.assertEqual(len(fourth["images"]), 1)
        self.assertTrue(all(r["meta"]["control"] for r in rows))

    def test_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, rows, errors = self.build(tmp, drop_failed=True)
        self.assertEqual(errors, [])
        self.assertEqual([r["meta"]["step_index"] for r in rows], [0, 2, 3])
        self.assertIn("→ failed", rows[1]["messages"][1]["content"], "failed steps stay in the history")
        self.assertEqual(manifest["skipped_steps"], {"action_failed": 1})
        with tempfile.TemporaryDirectory() as tmp:
            manifest, rows, _ = self.build(tmp, success_only=True)
        self.assertEqual((rows, manifest["skipped_steps"]), ([], {"episode_not_successful": 4}))

    def test_audit_third_party_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            path.write_text(json.dumps({"text": "代码放在/home/carol/data, /Volumes/backup1t/a.md, http://172.20.31.7/",
                                        "ok": "~/data <redacted:openai_key> /Volumes/<volume>/a TOKEN=<redacted>",
                                        "doc": "```bash\nexport ZAI_API_KEY=...\n```\n", "max_tokens": "16384",
                                        "api_key": "3f2a9c1d.Xy7Qk2"}, ensure_ascii=False) + "\n")
            found = audit.scan([tmp])
        self.assertEqual(sorted(found), ["home_path", "private_ip", "secret_field", "volume_path"])
        self.assertEqual(found["volume_path"]["count"], 1, "already redacted text is not reported again")
        self.assertNotIn("carol", json.dumps(found), "findings are masked")


class CellQueueAdapterTests(unittest.TestCase):
    def test_matching_and_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            s = SessionBuilder(output / "layout_000")
            b = RolloutBuilder(cwd=tmp).user("run the pour task")
            ctl = "python /opt/lab/robot_ctl.py --output " + str(output)
            (output / "current.json").write_text(json.dumps({"session": str(s.root)}))

            b.view_image(s.latest["images"]["cam_head"], Path(s.latest["images"]["cam_head"]).read_bytes())
            at = T0.timestamp() + 8
            cid, resp = s.command("ack_visual", "ack head", at, evidence="mug left, cup right")
            b.shell(f"{ctl} --note 'ack head' ack_visual --camera cam_head --evidence 'mug left, cup right'",
                    json.dumps(resp, indent=2))
            b.reasoning("go behind the cup")
            cid, resp = s.command("python", "pre-grasp", at + 6, code=CELL, moved=56,
                                  primitives=[("move_to", {"arm": "right"}, {"reached": True, "steps_used": 56})])
            b.shell(f"cat > /tmp/c.py <<'EOF'\n{CELL}EOF\n{ctl} --note 'pre-grasp' cell --file /tmp/c.py",
                    json.dumps(resp, indent=2))
            cid, resp = s.command("ack_visual", "ack wrist", at + 10)
            b.shell(f"{ctl} --brief --note 'ack wrist' ack_visual --camera cam_head --evidence 'x'",
                    f"[observation {resp['observation']['observation_index']}, native 56/1200, ack_required False]")
            cid, resp = s.command("python", "slide in", at + 14, code=CELL2, moved=20,
                                  primitives=[("move_to", {"arm": "right"}, {"reached": True, "steps_used": 20})])
            b.shell(f"{ctl} --async --note 'slide in' cell --file /tmp/missing.py",
                    json.dumps({"id": cid, "response": "r.json", "pending": True}))
            b.shell(f"{ctl} poll --id {cid}", json.dumps(resp))
            cid, resp = s.command("finish", "done", at + 20)
            b.shell(f"{ctl} --note done finish", json.dumps(resp))
            s.finish()
            path = b.write(Path(tmp) / "rollout.jsonl")
            errors, totals = export([path], Path(tmp) / "out", images="all")
            (steps,) = steps_of(Path(tmp) / "out")
            episode = json.loads(next((Path(tmp) / "out" / "episodes").iterdir()).joinpath("episode.json").read_text())
        self.assertEqual(errors, [])
        self.assertEqual([s_["action"]["signature"] for s_ in steps], ["python robot_ctl.py cell"] * 2
                         + ["python robot_ctl.py finish"])
        pre, slide, finish = steps
        info = pre["enrichment"]["cell_queue"]
        self.assertEqual([c["match"] for c in info["commands"]], ["observation_index", "observation_index"])
        self.assertEqual(info["native_actions"], {"before": 0, "after": 56, "count": 56})
        self.assertEqual(info["primitives"][0]["result"]["reached"], True)
        self.assertEqual(info["acks"][0]["evidence"], "mug left, cup right")
        self.assertEqual(pre["observation"]["state"]["observation_index"], 1)
        self.assertTrue(any(r["seen_by_model"] is False for r in info["session_images"]))
        self.assertTrue(all(r.get("file") for r in info["session_images"]))
        slide_info = slide["enrichment"]["cell_queue"]
        self.assertEqual([c["match"] for c in slide_info["commands"]], ["observation_index", "command_id",
                                                                         "command_id"])
        self.assertEqual(slide["action"]["code_source"], "session:commands.jsonl")
        self.assertEqual(slide["action"]["adjustments"][0]["changes"]["args[0]"]["delta"], [0.0, -0.015, 0.0])
        self.assertEqual(slide_info["native_actions"]["count"], 20)
        self.assertEqual(info["events"]["planner_events"][0]["op"], "python")
        self.assertEqual(finish["enrichment"]["cell_queue"]["op"], "finish")
        self.assertEqual(episode["enrichment"]["cell_queue"]["result"]["task_success"], True)
        self.assertEqual(episode["task"]["instruction"], "Put the block into the bowl.")
        self.assertEqual(totals["enriched"], 3)

    def test_moved_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "copied_runs"
            recorded = "/lab/runs"
            s = SessionBuilder(local / "s1")
            cid, resp = s.command("python", "reach", T0.timestamp() + 4, code=CELL, moved=10)
            s.finish()
            for file in (local / "s1").rglob("*.json"):  # the run recorded /lab/runs paths before it was copied
                file.write_text(file.read_text().replace(str(local), recorded))
            b = RolloutBuilder().user("go")
            b.shell("python robot_ctl.py --session runs/s1 --note reach cell --file c.py",
                    json.dumps(resp).replace(str(local), recorded), workdir="/lab")
            path = b.write(Path(tmp) / "r.jsonl")
            errors, totals = export([path], Path(tmp) / "out", images="all", remap_paths=((recorded, str(local)),))
            (steps,) = steps_of(Path(tmp) / "out")
            folder = next((Path(tmp) / "out" / "episodes").iterdir())
            text = "".join(p.read_text() for p in folder.glob("*.*"))
            markdown = (folder / "episode.md").read_text()
        self.assertEqual(errors, [])
        info = steps[0]["enrichment"]["cell_queue"]
        self.assertEqual(info["session"], "/lab/runs/s1")
        self.assertTrue(all(r["file"].startswith("images/session_") for r in info["session_images"]))
        self.assertNotIn(tmp, text)
        self.assertNotIn("_local", text)
        self.assertIn("## Step 0 · `python robot_ctl.py cell`", markdown)
        self.assertIn("`robo.move_to(", markdown)
        self.assertIn("native actions 0 → 10 (+10)", markdown)

    def test_missing_session_dir_stays_transcript_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = RolloutBuilder().user("go")
            b.shell("python robot_ctl.py --session /nowhere/s1 --note 'move' cell --file c.py", "{}")
            path = b.write(Path(tmp) / "r.jsonl")
            errors, _ = export([path], Path(tmp) / "out")
            (steps,) = steps_of(Path(tmp) / "out")
            episode = json.loads(next((Path(tmp) / "out" / "episodes").iterdir()).joinpath("episode.json").read_text())
        self.assertEqual(errors, [])
        self.assertEqual(steps[0]["enrichment"], {})
        self.assertEqual(episode["adapter_stats"]["cell_queue"]["unmatched"], {"session_dir_missing": 1})


class ExampleTests(unittest.TestCase):
    """examples/output must be what examples/make_examples.py produces with the current code."""

    @staticmethod
    def summary(out):
        rows = []
        for folder in sorted((Path(out) / "episodes").iterdir()):
            for line in (folder / "steps.jsonl").read_text().splitlines():
                step = json.loads(line)
                rows.append((folder.name, step["step_index"], step["action"]["signature"],
                             [c["func"] for c in step["action"]["calls"]],
                             [a["changes"] for a in step["action"]["adjustments"]],
                             [r["text"] for r in step["analysis"]["rationale"]], step["analysis"]["reasoning"],
                             len(step["observation"]["images"]), sorted(step["enrichment"]), step["flags"]))
        return rows

    def test_examples_are_current(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run([sys.executable, str(examples / "make_examples.py"), "--out", tmp],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            errors, totals, _ = validate(Path(tmp) / "output")
            fresh = self.summary(Path(tmp) / "output")
            markdown = [p.name for p in (Path(tmp) / "output" / "episodes").glob("*/episode.md")]
            sft_errors = sft_mod.check(Path(tmp) / "sft")
            fresh_sft = (Path(tmp) / "sft" / "train.jsonl").read_text()
            chat_errors = chat_mod.check(Path(tmp) / "chat")
            fresh_chat = {p.parent.name: p.read_text() for p in (Path(tmp) / "chat").glob("sessions/*/rounds.jsonl")}
        stored_chat = {p.parent.name: p.read_text() for p in (examples / "chat").glob("sessions/*/rounds.jsonl")}
        self.assertEqual((chat_errors, len(fresh_chat)), ([], 2))
        self.assertEqual(fresh_chat, stored_chat, "run python3 examples/make_examples.py")
        self.assertEqual(errors, [])
        self.assertEqual(totals["privacy"], {})
        self.assertEqual(len(markdown), 2)
        self.assertEqual(fresh, self.summary(examples / "output"), "run python3 examples/make_examples.py")
        self.assertEqual(sft_errors, [])
        self.assertEqual(fresh_sft, (examples / "sft" / "train.jsonl").read_text(), "run python3 examples/make_examples.py")


if __name__ == "__main__":
    unittest.main()
