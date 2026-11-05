"""episode.md: a readable view of one exported episode, rendered from the already-redacted records."""
from __future__ import annotations

import json


FIRST_KEYS = ("stdout", "stderr", "error", "message", "result", "output")


def _readable(text):
    """JSON tool output as `key: value` lines (stdout/error first) so it reads like a terminal; other text as is."""
    try:
        value = json.loads(text) if (text or "").lstrip()[:1] in "{[" else None
    except ValueError:
        value = None
    if not isinstance(value, (dict, list)):
        return text or ""
    leaves = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list) and not all(isinstance(v, (int, float)) for v in node):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")
        elif node not in (None, "", [], {}):
            leaves.append((path, node))
    walk(value, "")
    leaves.sort(key=lambda kv: next((i for i, k in enumerate(FIRST_KEYS) if kv[0].split(".")[-1] == k),
                                    len(FIRST_KEYS)))
    lines = []
    for path, leaf in leaves:
        if isinstance(leaf, str) and "\n" in leaf.strip():
            lines += [f"{path}:", leaf.rstrip()]
        else:
            lines.append(f"{path}: {leaf if isinstance(leaf, str) else json.dumps(leaf, ensure_ascii=False)}")
    return "\n".join(lines)


def _first_line(text, limit=160):
    line = next((l.strip() for l in _readable(text).splitlines() if l.strip()), "")
    return line if len(line) <= limit else line[:limit] + " …"


def _read(record):
    detail = _first_line(record.get("output"))
    if not detail and isinstance(record.get("args"), dict) and record["args"].get("path"):
        detail = str(record["args"]["path"]).rsplit("/", 1)[-1]
    return f"- read `{record['signature']}`" + (f": {detail}" if detail else "")


def _block(text, lang="", limit=40):
    lines = (text or "").rstrip().splitlines()
    body = "\n".join(lines[:limit])
    if len(lines) > limit:
        body += f"\n… ({len(lines) - limit} more lines)"
    fence = "````" if "```" in body else "```"
    return [f"{fence}{lang}", body, fence]


def _quote(text, limit=800):
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + " …"
    return "> " + text.replace("\n", "\n> ")


def _value(value):
    if isinstance(value, dict) and set(value) == {"expr"}:
        return value["expr"]
    return json.dumps(value, ensure_ascii=False)


def _call(call):
    args = [_value(a) for a in call.get("args", [])] + [f"{k}={_value(v)}" for k, v in call.get("kwargs", {}).items()]
    return f"`{call['func']}({', '.join(args)})`" + (" (in a loop)" if call.get("in_loop") else "")


def _change(name, change):
    text = f"{name} {_value(change.get('from'))} → {_value(change.get('to'))}"
    if "delta" in change:
        text += f" (Δ {_value(change['delta'])}" + (f", |Δ| {change['norm']}" if "norm" in change else "") + ")"
    return text


def _images(refs):
    stored = [r for r in refs if r.get("file")]
    lines = [" ".join(f"![{r.get('camera') or r.get('via') or 'image'}]({r['file']})" for r in stored)] if stored else []
    if len(refs) > len(stored):
        lines.append(f"({len(refs) - len(stored)} image(s) not stored)")
    return lines


def _step(step):
    act, obs, analysis, outcome = step["action"], step["observation"], step["analysis"], step["outcome"]
    when = (step["time"].get("action") or "")[11:19]
    out = [f"## Step {step['step_index']} · `{act['signature']}`" + (f" · {when}" if when else ""), ""]
    flags = [name for name, on in step["flags"].items() if on and name != "control"]
    if step["flags"].get("control") is False:
        flags.append("no control evidence")
    if flags:
        out += ["Flags: " + ", ".join(flags), ""]

    out += ["### Observation", ""]
    images = _images(obs["images"])
    details = [_read(r) for r in obs["reads"]]
    if isinstance(obs.get("state"), dict):
        scalars = [f"{k}={v}" for k, v in obs["state"].items()
                   if isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 60)]
        if scalars:
            details.append("- state: " + ", ".join(scalars[:8]))
    if images:
        out += images + [""]
    out += details
    if not images and not details:
        out.append("(nothing new since the previous action)")

    out += ["", "### Analysis", ""]
    for text in analysis["user_messages"]:
        out += [_quote("User: " + text), ""]
    for text in analysis["reasoning"] + analysis["messages"]:
        out += [_quote(text), ""]
    out += [f"- {r['key'].lstrip('-')}: {r['text']}" for r in analysis["rationale"]]
    out += [f"- code comment: {c}" for c in analysis["code_comments"]]

    out += ["", "### Action", ""]
    if act["code"]:
        source = act.get("code_source") or ""
        out += _block(act["code"], "bash" if source.endswith(".sh") else "python")
    elif act["command"]:
        out += _block(act["command"], "bash")
    elif act["args"] is not None:
        out += _block(json.dumps(act["args"], ensure_ascii=False, indent=2), "json")
    if act["calls"]:
        out += ["", "Control calls:"] + [f"- {_call(c)}" for c in act["calls"]]
    if act["adjustments"]:
        out += ["", "Adjustments against earlier steps:"]
        out += [f"- `{a['func']}` vs step {a['previous_step']}: "
                + "; ".join(_change(k, v) for k, v in a["changes"].items()) for a in act["adjustments"]]

    out += ["", "### Outcome", ""]
    bits = []
    if outcome["exit_code"] is not None:
        bits.append(f"exit {outcome['exit_code']}")
    if outcome["duration_s"]:
        bits.append(f"{outcome['duration_s']:.1f} s")
    for info in step["enrichment"].values():
        native = info.get("native_actions") if isinstance(info, dict) else None
        if native:
            bits.append(f"native actions {native['before']} → {native['after']} (+{native['count']})")
        if isinstance(info, dict) and info.get("primitives"):
            bits.append(f"{len(info['primitives'])} primitive call(s)")
    if bits:
        out += [" · ".join(bits), ""]
    if outcome["output"]:
        out += _block(_readable(outcome["output"]), "", limit=15)
    for result in outcome["results"]:
        out += ["", f"Result delivered by `{result['signature']}`:"] + _block(_readable(result.get("output")), "", 12)
    out += _images(outcome["images"])
    return out + [""]


def render(episode, steps):
    out = [f"# {episode['episode_id']}", ""]
    task = episode.get("task") or {}
    if task.get("prompt"):
        out += ["**Task prompt**", "", _quote(task["prompt"], 2000), ""]
    if task.get("instruction") and task.get("instruction") != task.get("prompt"):
        out += [f"**Instruction:** {task['instruction']}", ""]
    stats = episode.get("stats") or {}
    models = ", ".join((episode.get("agent") or {}).get("models") or []) or "unknown model"
    out += [f"{models} · profiles {', '.join(episode['profile']['names'])} · {stats.get('steps', len(steps))} steps"
            f" · started {episode['source'].get('started_at')}", ""]
    for name, info in (episode.get("enrichment") or {}).items():
        if isinstance(info, dict) and info.get("result"):
            out += [f"Result ({name}): `{json.dumps(info['result'], ensure_ascii=False)[:400]}`", ""]
    for step in steps:
        out += _step(step)
    final = episode.get("final") or {}
    closing = final.get("user_messages", []) + final.get("messages", [])
    if closing:
        out += ["## After the last action", ""]
        for text in final.get("user_messages", []):
            out += [_quote("User: " + text), ""]
        for text in final.get("messages", []):
            out += [_quote(text), ""]
    return "\n".join(out).rstrip() + "\n"
