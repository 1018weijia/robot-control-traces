"""Offline tests: python3 -m unittest discover -s tests"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixtures import RolloutBuilder, SessionBuilder, T0, png

from rct_lib import calls, codex, shell
from rct_lib import export as export_mod
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
        self.assertEqual(errors, [])
        self.assertEqual(len(markdown), 2)
        self.assertEqual(fresh, self.summary(examples / "output"), "run python3 examples/make_examples.py")


if __name__ == "__main__":
    unittest.main()
