"""Rollouts → dataset directory (manifest.json, episodes/<id>/{episode.json, episode.md, steps.jsonl, timeline.jsonl,
images/})."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from . import adapters, profile as profile_mod, report as report_mod, steps as steps_mod
from .codex import read_rollout
from .redact import Redactor

VERSION = "0.3.0"
MANIFEST_SCHEMA = "robot-control-trace-manifest/v1"
PRIVATE_KEYS = ("_act", "_observe", "_results", "_episode_hint")
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


@dataclasses.dataclass
class Options:
    profiles: tuple = ()
    auto_profiles: bool = True
    images: str = "seen"            # seen | all | none
    redact: bool = True
    path_map: tuple = ()
    max_chars: int = 4000
    use_adapters: bool = True
    read_local_files: bool = False
    min_act_steps: int = 1
    min_control_steps: int = 1      # act steps with flags.control; 0 keeps rollouts without robot evidence
    remap_paths: tuple = ()         # (recorded, local) prefixes for run directories that moved since recording


def analyse(path, options: Options):
    """Parse and segment one rollout; returns (rollout, profile, episodes, adapter_report)."""
    rollout = read_rollout(path, keep_images=options.images != "none")
    profile = profile_mod.build(rollout, options.profiles, options.auto_profiles)
    steps_mod.prepare(rollout, profile, options.read_local_files)
    steps, final = steps_mod.segment(rollout, profile, options.max_chars)
    report = adapters.enrich(profile.adapters, steps, rollout, options) if options.use_adapters else {}
    episodes = split(steps, profile)
    for _, group in episodes:
        for index, step in enumerate(group):
            step["step_index"] = index
            feedback = step["observation"]["prior_feedback"]
            if feedback is not None:
                if index:
                    feedback["step"] = index - 1
                else:
                    step["observation"]["prior_feedback"] = None  # it came from the previous episode
        steps_mod.annotate_actions(group, profile, profile.persistent_namespace)
    return rollout, profile, episodes, final, report


def split(steps, profile):
    """[(hint, steps)] grouped by adapter hints (run directory) or the profile's episode_key, in first-seen order."""
    adapter_hints = [s.pop("_episode_hint", None) for s in steps]
    if any(adapter_hints):
        hints = adapter_hints
    elif profile.episode_key is not None:
        hints = []
        for step in steps:
            act = step["_act"]
            found = profile.episode_key.search(act.command or json.dumps(act.args, default=str))
            hints.append((found.group(1) if found.groups() else found.group(0)) if found else None)
    else:
        hints = [None] * len(steps)
    last = None
    for i, hint in enumerate(hints):
        hints[i] = last = hint or last
    first = next((h for h in hints if h), None)
    groups = {}
    for step, hint in zip(steps, hints):
        groups.setdefault(hint or first, []).append(step)
    return list(groups.items())


def event_record(event, limit):
    record = {"seq": event.seq, "ts": event.ts, "turn": event.turn, "kind": event.kind, "role": event.role,
              "rule": event.rule, "tool": event.tool, "signature": event.signature,
              "text": steps_mod.clip(event.text, limit) if event.text else None, "phase": event.phase,
              "command": steps_mod.clip(event.command, limit), "code_source": event.code_source,
              "output": steps_mod.clip(event.output, limit), "exit_code": event.exit_code,
              "images": event.images or None, "continuation_of": event.continuation_of}
    if event.command is None and event.args is not None:
        record["args"] = steps_mod._clip_json(event.args, limit)
    return {k: v for k, v in record.items() if v is not None}


class Writer:
    def __init__(self, out: Path, options: Options):
        self.out = out
        self.options = options
        self.redactor = Redactor(options.redact, options.path_map)
        settings = {k: v for k, v in dataclasses.asdict(options).items() if k not in ("path_map", "remap_paths")}
        settings["remapped_paths"] = len(options.remap_paths)
        self.manifest = {"schema": MANIFEST_SCHEMA, "generator": {"name": "robot-control-traces", "version": VERSION},
                         "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "options": self.redactor.obj(settings), "sources": [], "episodes": []}

    def _save_image(self, folder, rollout, ref):
        image = rollout.images.get(ref["id"])
        if self.options.images == "none" or image is None or image.data is None:
            return
        name = ref["id"][:16] + EXTENSIONS.get(image.mime or "", ".png")
        path = folder / "images" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.data)
        ref["file"] = f"images/{name}"

    def _session_images(self, folder, step):
        """Copy run-directory frames an adapter listed (with --images all); local paths never reach the export."""
        for info in step["enrichment"].values():
            for ref in info.get("session_images", []) if isinstance(info, dict) else []:
                source = ref.pop("_local", None) or ref.get("path")
                if self.options.images != "all" or not source or not os.path.isfile(source):
                    continue
                digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                name = f"session_{digest[:16]}{Path(source).suffix or '.png'}"
                target = folder / "images" / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                ref["file"], ref["file_sha256"] = f"images/{name}", digest

    def write_rollout(self, path, rollout, profile, episodes, final, report):
        source = {"rollout_file": self.redactor.text(str(path)), "rollout_sha256": rollout.sha256,
                  "lines": rollout.stats.get("lines"), "warnings": len(rollout.warnings),
                  "profiles": profile.names, "episodes": []}
        date = (rollout.started or "")[:10] or "undated"
        thread = rollout.thread_id.replace("-", "")[:8]
        for number, (hint, group) in enumerate(episodes):
            episode_id = f"{date}_{thread}_{number:02d}"
            folder = self.out / "episodes" / episode_id
            folder.mkdir(parents=True)
            is_last = number == len(episodes) - 1
            seqs = [s["source"]["seq_start"] for s in group] + [s["source"]["seq_action"] for s in group]
            low, high = min(seqs), (float("inf") if is_last else max(seqs))
            with open(folder / "timeline.jsonl", "w", encoding="utf-8") as handle:
                for event in rollout.events:
                    if low <= event.seq <= high:
                        record = self.redactor.obj(event_record(event, self.options.max_chars))
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats = {"steps": len(group), "images_seen": 0, "images_saved": 0}
            records = []
            with open(folder / "steps.jsonl", "w", encoding="utf-8") as handle:
                for step in group:
                    for key in PRIVATE_KEYS:
                        step.pop(key, None)
                    step["episode_id"] = episode_id
                    refs = step["observation"]["images"] + step["outcome"]["images"]
                    for ref in refs:
                        self._save_image(folder, rollout, ref)
                    self._session_images(folder, step)
                    stats["images_seen"] += len(refs)
                    stats["images_saved"] += sum(1 for r in refs if r["file"])
                    records.append(json.loads(json.dumps(self.redactor.obj(step), ensure_ascii=False, default=str)))
                    handle.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
            if is_last:
                for ref in final["observation"]["images"]:
                    self._save_image(folder, rollout, ref)
            stats.update(coverage(group))
            prompt = next((e.text for e in rollout.events if e.kind == "user_message" and e.text), None)
            episode = {
                "schema": steps_mod.SCHEMA, "episode_id": episode_id,
                "source": {"rollout_file": path.name, "rollout_sha256": rollout.sha256,
                           "thread_id": rollout.thread_id,
                           "parent_thread_id": rollout.meta.get("parent_thread_id") or rollout.meta.get(
                               "forked_from_id"),
                           "originator": rollout.meta.get("originator"), "cli_version": rollout.meta.get("cli_version"),
                           "session_source": rollout.meta.get("source"), "started_at": rollout.started,
                           "cwd": rollout.cwd, "episode_key": hint},
                "agent": {"models": rollout.models,
                          "efforts": sorted({c.get("effort") for c in rollout.turn_contexts if c.get("effort")})},
                "task": {"prompt": steps_mod.clip(prompt, 8000),
                         "instruction": next((s["observation"]["state"].get("instruction") for s in group
                                              if isinstance(s["observation"]["state"], dict)
                                              and s["observation"]["state"].get("instruction")), None)},
                "profile": {"names": profile.names, "adapters": profile.adapters},
                "stats": stats,
                "final": final if is_last else None,
                "enrichment": {name: (r.get("episodes") or {}).get(hint) for name, r in report.items()
                               if (r.get("episodes") or {}).get(hint)},
                "adapter_stats": {name: r.get("stats") for name, r in report.items()},
                "warnings": rollout.warnings[:50],
            }
            episode = json.loads(json.dumps(self.redactor.obj(episode), ensure_ascii=False, default=str))
            with open(folder / "episode.json", "w", encoding="utf-8") as handle:
                json.dump(episode, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            (folder / "episode.md").write_text(report_mod.render(episode, records), encoding="utf-8")
            source["episodes"].append(episode_id)
            self.manifest["episodes"].append({"id": episode_id, "steps": len(group), "rollout": path.name})
        self.manifest["sources"].append(source)

    def skip(self, path, reason, rollout=None):
        self.manifest["sources"].append({"rollout_file": self.redactor.text(str(path)), "skipped": reason,
                                         "rollout_sha256": rollout.sha256 if rollout else None})

    def close(self):
        self.manifest["redaction"] = {"enabled": self.options.redact, "counts": self.redactor.counts,
                                      "custom_paths": len(self.options.path_map)}
        with open(self.out / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return self.manifest


def coverage(group):
    def share(predicate):
        return round(sum(1 for s in group if predicate(s)) / len(group), 3) if group else 0.0
    return {
        "with_images": share(lambda s: s["observation"]["images"]),
        "with_observation": share(lambda s: s["observation"]["images"] or s["observation"]["reads"]
                                  or s["observation"]["state"]),
        "with_reasoning": share(lambda s: s["analysis"]["reasoning"]),
        "with_analysis_text": share(lambda s: s["analysis"]["reasoning"] or s["analysis"]["messages"]
                                    or s["analysis"]["rationale"]),
        "with_code": share(lambda s: s["action"]["code"]),
        "with_calls": share(lambda s: s["action"]["calls"]),
        "with_adjustments": share(lambda s: s["action"]["adjustments"]),
        "enriched": share(lambda s: s["enrichment"]),
        "control_steps": sum(1 for s in group if s["flags"].get("control")),
        "human_input_steps": sum(1 for s in group if s["flags"]["human_input"]),
        "failed_actions": sum(1 for s in group if s["flags"]["action_failed"]),
    }


def prepare_out(out: Path, overwrite: bool):
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise SystemExit(f"{out} is not empty; choose a new directory or pass --overwrite")
        if not (out / "manifest.json").exists():
            raise SystemExit(f"refusing to overwrite {out}: it does not look like a previous export")
        shutil.rmtree(out / "episodes", ignore_errors=True)
        (out / "manifest.json").unlink()
    out.mkdir(parents=True, exist_ok=True)


def export(paths, out, options: Options, overwrite=False, log=print):
    out = Path(out)
    prepare_out(out, overwrite)
    writer = Writer(out, options)
    for path in paths:
        path = Path(path)
        try:
            rollout, profile, episodes, final, report = analyse(path, options)
        except Exception as exc:  # one broken rollout must not stop a batch
            writer.skip(path, f"parse error: {type(exc).__name__}: {exc}")
            log(f"skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        acts = sum(len(g) for _, g in episodes)
        control = sum(1 for _, g in episodes for s in g if s["flags"].get("control"))
        if acts < options.min_act_steps:
            writer.skip(path, f"{acts} act steps (< {options.min_act_steps})", rollout)
            continue
        if control < options.min_control_steps:
            writer.skip(path, f"no robot-control evidence: {control} of {acts} act steps "
                              f"(< {options.min_control_steps}); see scan --detail", rollout)
            log(f"skip {path.name}: {acts} act steps but {control} with robot-control evidence")
            continue
        writer.write_rollout(path, rollout, profile, episodes, final, report)
        log(f"{path.name}: {len(episodes)} episode(s), {acts} steps, profiles {', '.join(profile.names)}")
    return writer.close()
