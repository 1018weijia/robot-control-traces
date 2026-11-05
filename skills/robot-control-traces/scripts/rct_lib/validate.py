"""Check an export against the step schema and summarise how complete its observation/analysis/action data is."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .steps import SCHEMA

STEP_FIELDS = {"schema": str, "episode_id": str, "step_index": int, "time": dict, "turn": int,
               "observation": dict, "analysis": dict, "action": dict, "outcome": dict, "context": dict,
               "flags": dict, "source": dict, "enrichment": dict}
SECTION_FIELDS = {
    "observation": {"images": list, "reads": list},
    "analysis": {"reasoning": list, "messages": list, "rationale": list, "code_comments": list, "user_messages": list},
    "action": {"calls": list, "probes": list, "adjustments": list},
    "outcome": {"images": list, "results": list},
}
EPISODE_FIELDS = {"schema": str, "episode_id": str, "source": dict, "agent": dict, "task": dict, "stats": dict}


def _check(obj, fields, where, errors):
    for name, kind in fields.items():
        if name not in obj:
            errors.append(f"{where}: missing '{name}'")
        elif not isinstance(obj[name], kind) or (kind is int and isinstance(obj[name], bool)):
            errors.append(f"{where}: '{name}' should be {kind.__name__}")


def validate(out):
    out = Path(out)
    errors, totals, samples = [], {}, []
    try:
        manifest = json.loads((out / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"manifest.json: {exc}"], {}, []
    for entry in manifest.get("episodes", []):
        folder = out / "episodes" / entry["id"]
        try:
            episode = json.loads((folder / "episode.json").read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"{entry['id']}/episode.json: {exc}")
            continue
        _check(episode, EPISODE_FIELDS, f"{entry['id']}/episode.json", errors)
        try:
            lines = (folder / "steps.jsonl").read_text().splitlines()
        except OSError as exc:
            errors.append(f"{entry['id']}/steps.jsonl: {exc}")
            continue
        if len(lines) != entry.get("steps"):
            errors.append(f"{entry['id']}: manifest says {entry.get('steps')} steps, file has {len(lines)}")
        for number, line in enumerate(lines):
            where = f"{entry['id']}/steps.jsonl:{number + 1}"
            try:
                step = json.loads(line)
            except ValueError as exc:
                errors.append(f"{where}: {exc}")
                continue
            _check(step, STEP_FIELDS, where, errors)
            for section, fields in SECTION_FIELDS.items():
                if isinstance(step.get(section), dict):
                    _check(step[section], fields, f"{where} {section}", errors)
            if step.get("schema") != SCHEMA:
                errors.append(f"{where}: schema {step.get('schema')!r} != {SCHEMA!r}")
            if step.get("step_index") != number:
                errors.append(f"{where}: step_index {step.get('step_index')} out of order")
            if step.get("episode_id") != entry["id"]:
                errors.append(f"{where}: episode_id mismatch")
            for adjustment in (step.get("action") or {}).get("adjustments", []):
                if not isinstance(adjustment.get("previous_step"), int) or adjustment["previous_step"] >= number:
                    errors.append(f"{where}: adjustment refers to step {adjustment.get('previous_step')}")
            for ref in (step.get("observation") or {}).get("images", []) + (step.get("outcome") or {}).get("images", []):
                if ref.get("file"):
                    path = folder / ref["file"]
                    if not path.is_file():
                        errors.append(f"{where}: missing image {ref['file']}")
                    elif not ref["id"].startswith("file:") and hashlib.sha256(path.read_bytes()).hexdigest() != ref["id"]:
                        errors.append(f"{where}: image {ref['file']} does not match its sha256")
            _tally(totals, step)
            if len(samples) < 3 and (step.get("analysis") or {}).get("rationale") or len(samples) < 1:
                samples.append(_sample(step))
    totals["episodes"] = len(manifest.get("episodes", []))
    totals["skipped_sources"] = sum(1 for s in manifest.get("sources", []) if s.get("skipped"))
    return errors, totals, samples


def _tally(totals, step):
    def bump(key, cond=True):
        totals[key] = totals.get(key, 0) + (1 if cond else 0)
    obs, ana, act = step.get("observation") or {}, step.get("analysis") or {}, step.get("action") or {}
    bump("steps")
    bump("with_images", obs.get("images"))
    bump("with_any_observation", obs.get("images") or obs.get("reads") or obs.get("state"))
    bump("with_reasoning", ana.get("reasoning"))
    bump("with_rationale", ana.get("rationale"))
    bump("with_any_analysis", ana.get("reasoning") or ana.get("messages") or ana.get("rationale"))
    bump("with_code", act.get("code"))
    bump("with_calls", act.get("calls"))
    bump("with_adjustments", act.get("adjustments"))
    bump("enriched", step.get("enrichment"))
    bump("human_input", (step.get("flags") or {}).get("human_input"))


def _sample(step):
    ana, act = step["analysis"], step["action"]
    texts = ana["reasoning"] + ana["messages"] + [r["text"] for r in ana["rationale"] if r["source"] == "act"] \
        + [r["text"] for r in ana["rationale"]]
    text = texts[0] if texts else ""
    return {"episode_id": step["episode_id"], "step_index": step["step_index"], "analysis": text[:300],
            "action": act.get("signature"), "calls": [c["func"] for c in act.get("calls", [])][:6]}


def format_report(errors, totals, samples):
    lines = []
    steps = totals.get("steps", 0)
    lines.append(f"episodes {totals.get('episodes', 0)}, steps {steps}, skipped sources "
                 f"{totals.get('skipped_sources', 0)}")
    for key in ("with_images", "with_any_observation", "with_reasoning", "with_rationale", "with_any_analysis",
                "with_code", "with_calls", "with_adjustments", "enriched", "human_input"):
        value = totals.get(key, 0)
        lines.append(f"  {key:<22} {value:>6}  ({value / steps:.0%})" if steps else f"  {key:<22} {value:>6}")
    if samples:
        lines.append("samples (check that redaction and segmentation look right):")
        for sample in samples:
            lines.append(f"  [{sample['episode_id']} #{sample['step_index']}] {sample['action']} "
                         f"calls={sample['calls']}\n      analysis: {sample['analysis']!r}")
    if errors:
        lines.append(f"{len(errors)} schema error(s):")
        lines += [f"  {e}" for e in errors[:50]]
    else:
        lines.append("schema OK")
    return "\n".join(lines)
