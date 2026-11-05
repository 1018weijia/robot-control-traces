"""Redaction applied to every exported string: home directories, credentials and e-mail addresses.

It is a safety net, not a guarantee: images are never altered, and free text can still identify people or
machines. Exports meant for publication should be spot-checked (``rct.py validate`` prints a sample).
"""
from __future__ import annotations

import re

SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]{16,}=*")),
    ("assignment", re.compile(r"(?i)\b([A-Z0-9_]*(?:API_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|ACCESS_KEY)[A-Z0-9_]*)"
                              r"(['\"]?\s*[:=]\s*)(['\"]?)([^\s'\",;]{6,})\3")),
    ("url_credentials", re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@")),
]
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A path may follow an escaped newline/tab inside JSON text that a tool printed ("...\n/home/alice/x").
_PATH_START = r"(?:(?<![\w.-])|(?<=\\n)|(?<=\\t)|(?<=\\r))"
HOME_RE = re.compile(_PATH_START + r"(?:/mnt/home|/home|/Users|/data/home)/[A-Za-z0-9._-]+"
                     r"|" + _PATH_START + r"/root(?=/|\b)"
                     r"|\b[A-Za-z]:\\{1,2}Users\\{1,2}[^\\\s\"']+")


class Redactor:
    def __init__(self, enabled=True, path_map=()):
        self.enabled = enabled
        self.path_map = sorted(((a.rstrip("/"), b) for a, b in path_map if a), key=lambda p: -len(p[0]))
        self.counts = {}

    def _count(self, kind, n=1):
        if n:
            self.counts[kind] = self.counts.get(kind, 0) + n

    def text(self, value):
        if not self.enabled or not isinstance(value, str) or not value:
            return value
        for old, new in self.path_map:
            if old in value:
                self._count("custom_path", value.count(old))
                value = value.replace(old, new)
        for kind, pattern in SECRET_PATTERNS:
            if kind == "assignment":
                value, n = pattern.subn(lambda m: m.group(1) + m.group(2) + m.group(3) + "<redacted>" + m.group(3),
                                        value)
            elif kind == "bearer":
                value, n = pattern.subn(lambda m: m.group(1) + " <redacted>", value)
            elif kind == "url_credentials":
                value, n = pattern.subn(lambda m: m.group(1) + "<redacted>@", value)
            else:
                value, n = pattern.subn(f"<redacted:{kind}>", value)
            self._count(kind, n)
        value, n = EMAIL_RE.subn("<email>", value)
        self._count("email", n)
        value, n = HOME_RE.subn("~", value)
        self._count("home_path", n)
        return value

    def obj(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.obj(v) for v in value]
        if isinstance(value, dict):
            return {self.text(k) if isinstance(k, str) else k: self.obj(v) for k, v in value.items()}
        return value
