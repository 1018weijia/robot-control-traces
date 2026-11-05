"""Profiles: JSON rules that say which tool calls observe, act, deliver results, or are auxiliary.

A profile is selected explicitly (``--profile name-or-path``) or automatically when one of its ``detect`` regexes
matches a command or tool name in the rollout. The built-in ``default`` profile is always appended last, so its
keyword heuristics only decide what the more specific profiles leave unmatched.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re

PROFILE_DIR = Path(__file__).resolve().parents[2] / "profiles"
ROLES = ("act", "observe", "result", "aux", "ignore")
LIST_FIELDS = ("rationale_options", "rationale_keys", "code_file_options", "adapters")
PATTERN_FIELDS = ("action_calls", "probe_calls")


@dataclasses.dataclass
class Rule:
    role: str
    origin: str
    tool: re.Pattern | None = None
    command: re.Pattern | None = None
    code: re.Pattern | None = None
    args: re.Pattern | None = None
    kind: str | None = None
    raw: bool = False

    def matches(self, event, kind):
        if self.kind and self.kind != kind:
            return False
        checks = [(self.tool, event.tool or ""),
                  (self.command, (event.command if self.raw else event.skeleton) if event.command else None),
                  (self.code, event.code),
                  (self.args, json.dumps(event.args, ensure_ascii=False) if event.args is not None else None)]
        used = False
        for pattern, text in checks:
            if pattern is None:
                continue
            used = True
            if text is None or not pattern.search(text):
                return False
        return used


@dataclasses.dataclass
class Profile:
    names: list
    rules: list
    rationale_options: list
    rationale_keys: list
    code_file_options: list
    action_calls: re.Pattern
    probe_calls: re.Pattern
    episode_key: re.Pattern | None
    adapters: list
    persistent_namespace: bool = False

    def classify(self, event, kind):
        for rule in self.rules:
            if rule.matches(event, kind):
                return rule.role, rule.origin
        return "aux", "unmatched"


def tool_kind(event):
    if event.command is not None:
        return "shell"
    name = event.tool or ""
    return "mcp" if "__" in name or "." in name else "builtin"


def builtin_names():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def load_spec(name_or_path):
    path = Path(name_or_path)
    if not path.suffix and not path.exists():
        path = PROFILE_DIR / f"{name_or_path}.json"
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {name_or_path} (built-in: {', '.join(builtin_names())})")
    spec = json.loads(path.read_text())
    validate_spec(spec, str(path))
    spec["_path"] = str(path)
    return spec


def validate_spec(spec, where="profile"):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
        raise ValueError(f"{where}: a profile is a JSON object with a string 'name'")
    for i, rule in enumerate(spec.get("rules", [])):
        if rule.get("role") not in ROLES:
            raise ValueError(f"{where}: rules[{i}].role must be one of {ROLES}")
        if not any(rule.get(k) for k in ("tool", "command", "code", "args")):
            raise ValueError(f"{where}: rules[{i}] needs at least one of tool/command/code/args")
        for key in ("tool", "command", "code", "args"):
            if rule.get(key):
                re.compile(rule[key])
    for key in PATTERN_FIELDS + ("episode_key",):
        if spec.get(key):
            re.compile(spec[key])


def detected(spec, rollout):
    patterns = [re.compile(p) for p in spec.get("detect", [])]
    if not patterns:
        return False
    for event in rollout.events:
        if event.kind == "tool_call":
            text = (event.command or "") + "\n" + (event.tool or "")
            if any(p.search(text) for p in patterns):
                return True
    return False


def build(rollout=None, explicit=(), auto=True):
    """Combined profile: explicit profiles, then auto-detected built-ins, then the default profile."""
    specs = [load_spec(p) for p in explicit]
    names = {s["name"] for s in specs}
    if auto and rollout is not None:
        for name in builtin_names():
            if name == "default" or name in names:
                continue
            spec = load_spec(name)
            if detected(spec, rollout):
                specs.append(spec)
                names.add(spec["name"])
    if "default" not in names:
        specs.append(load_spec("default"))
    rules = []
    for spec in specs:
        for index, rule in enumerate(spec.get("rules", [])):
            rules.append(Rule(role=rule["role"], origin=f"{spec['name']}:{rule.get('label', index)}",
                              kind=rule.get("kind"), raw=bool(rule.get("raw")),
                              **{k: re.compile(rule[k]) if rule.get(k) else None
                                 for k in ("tool", "command", "code", "args")}))
    merged = {k: [] for k in LIST_FIELDS}
    for spec in specs:
        for key in LIST_FIELDS:
            for value in spec.get(key, []):
                if value not in merged[key]:
                    merged[key].append(value)
    patterns = {}
    for key in PATTERN_FIELDS:
        parts = [f"(?:{s[key]})" for s in specs if s.get(key)]
        patterns[key] = re.compile("|".join(parts) if parts else r"(?!x)x")
    key = next((s["episode_key"] for s in specs if s.get("episode_key")), None)
    return Profile(names=[s["name"] for s in specs], rules=rules, episode_key=re.compile(key) if key else None,
                   persistent_namespace=any(s.get("persistent_namespace") for s in specs), **merged, **patterns)
