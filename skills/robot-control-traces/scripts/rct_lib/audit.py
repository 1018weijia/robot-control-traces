"""Privacy scan for exported or third-party datasets: what the redaction rules would still change.

Any text file works (JSON, JSONL, Markdown, TSV), so a dataset converted by another tool can be checked before
publication. JSON is decoded first, so escapes and field boundaries match what redaction sees on export.
Findings are shown masked; images are not inspected.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from .redact import (EMAIL_RE, HOME_RE, PRIVATE_IP_RE, SECRET_PATTERNS, VOLUME_RE, is_secret_assignment,
                     is_secret_field)

TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".tsv", ".csv", ".yaml", ".yml"}
PATTERNS = [(kind, pattern) for kind, pattern in SECRET_PATTERNS] + [
    ("email", EMAIL_RE), ("home_path", HOME_RE), ("volume_path", VOLUME_RE), ("private_ip", PRIVATE_IP_RE)]
MARKER = re.compile(r"<(?:redacted|volume|private-ip|email)\b")


def _mask(text):
    return text if len(text) <= 8 else text[:6] + "…" + f"({len(text)} chars)"


def files(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        elif path.is_file():
            yield path


def _walk(value, where, key=None):
    if isinstance(value, str):
        yield where, key, value
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, where)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                yield where, None, k
            yield from _walk(v, where, k)


def _texts(path):
    """(location, json key or None, text) for every string in the file."""
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8", errors="replace") as handle:
        if suffix == ".json":
            try:
                yield from _walk(json.load(handle), path.name)
                return
            except ValueError:
                handle.seek(0)
        for number, line in enumerate(handle, 1):
            where = f"{path.name}:{number}"
            if suffix == ".jsonl":
                try:
                    yield from _walk(json.loads(line), where)
                    continue
                except ValueError:
                    pass
            yield where, None, line


def _findings(key, text):
    if key is not None and is_secret_field(key, text):
        yield "secret_field", text
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            if MARKER.search(match.group(0)) or (kind == "assignment" and not is_secret_assignment(match)):
                continue
            yield kind, match.group(0)


def scan(paths, examples=3):
    """{kind: {"count": n, "files": n, "examples": [masked match (file:line)]}}"""
    found = {}
    for path in files(paths):
        seen = collections.Counter()
        for where, key, text in _texts(path):
            for kind, hit in _findings(key, text):
                entry = found.setdefault(kind, {"count": 0, "files": 0, "examples": []})
                entry["count"] += 1
                seen[kind] += 1
                if len(entry["examples"]) < examples:
                    entry["examples"].append(f"{_mask(hit)} ({where})")
        for kind in seen:
            found[kind]["files"] += 1
    return found


def format_findings(found):
    if not found:
        return "privacy scan: nothing the redaction rules would change"
    lines = [f"privacy scan: {sum(e['count'] for e in found.values())} finding(s)"]
    for kind, entry in sorted(found.items(), key=lambda kv: -kv[1]["count"]):
        lines.append(f"  {kind:<18} {entry['count']:>6} in {entry['files']} file(s): " + "; ".join(entry["examples"]))
    return "\n".join(lines)
