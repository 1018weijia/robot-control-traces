"""Best-effort shell parsing: heredocs, quoted-string skeletons, option values, code and file writes.

Nothing here executes or reads the commands' files; it only looks at the text the agent sent.
"""
from __future__ import annotations

import os
import re
import shlex

HEREDOC_RE = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
QUOTED_RE = re.compile(r"\$?'[^']*'|\"(?:[^\"\\]|\\.)*\"")
REDIRECT_RE = re.compile(r"(?:^|[\s;&|])(?:>|>>|1>)\s*(['\"]?)([^\s'\";&|<>]+)\1")
PYTHON_RE = re.compile(r"^(?:.*/)?(?:python(?:\d+(?:\.\d+)?)?|pypy3?)$")
WRAPPERS = {"env", "sudo", "nohup", "time", "exec", "command", "stdbuf", "nice", "xvfb-run"}
OPERATORS = {"&&", "||", ";", "|", "&", "(", ")", ";;", "|&", "\n"}
BOOLEAN_FLAGS = {"--async", "--brief", "--verbose", "-v", "--quiet", "-q", "--json", "--dry-run", "--force", "-f",
                 "--all", "--debug", "--help", "-h", "-u", "--no-color", "--yes", "-y"}


def heredocs(script: str):
    """[(header_line, body, delimiter)] and the script with heredoc bodies removed."""
    lines = script.split("\n")
    found, kept, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        matches = list(HEREDOC_RE.finditer(line))
        i += 1
        for match in matches:
            strip, delimiter = match.group(1) == "-", match.group(3)
            body = []
            while i < len(lines):
                candidate = lines[i].lstrip("\t") if strip else lines[i]
                i += 1
                if candidate.strip() == delimiter:
                    break
                body.append(lines[i - 1])
            found.append((line, "\n".join(body), delimiter))
    return found, "\n".join(kept).replace("\\\n", " ")


def skeleton(script: str) -> str:
    """Script without heredoc bodies and with every quoted string replaced by '…' (for rule matching)."""
    _, stripped = heredocs(script)
    return QUOTED_RE.sub("'…'", stripped)


def tokens(script: str):
    """Shell tokens of the script without heredoc bodies; unquoted newlines become ``\\n`` separator tokens.

    None when the quoting cannot be parsed.
    """
    _, stripped = heredocs(script)
    lexer = shlex.shlex(stripped, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return [t for t in lexer if t]
    except ValueError:
        return None


def commands(script: str):
    """Split tokens into simple commands at shell operators, dropping env assignments and wrappers."""
    toks = tokens(script)
    if toks is None:
        return []
    out, current = [], []
    for tok in toks + [";"]:
        if tok in OPERATORS or set(tok) <= set(";&|()\n"):
            if current:
                out.append(current)
            current = []
            continue
        current.append(tok)
    cleaned = []
    for argv in out:
        while argv and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]) or argv[0] in WRAPPERS):
            argv = argv[1:]
        if argv and argv[0] == "timeout" and len(argv) > 1:
            argv = argv[2:] if not argv[1].startswith("-") else argv[3:]
        if argv:
            cleaned.append(argv)
    return cleaned


def option_values(script: str, names) -> dict:
    """{option: [values]} for options like --note given as ``--note X`` or ``--note=X``."""
    names = list(names)
    found = {}
    toks = tokens(script)
    if toks is not None:
        for i, tok in enumerate(toks):
            for name in names:
                if tok == name and i + 1 < len(toks):
                    found.setdefault(name, []).append(toks[i + 1])
                elif tok.startswith(name + "="):
                    found.setdefault(name, []).append(tok[len(name) + 1:])
        return found
    for name in names:  # unparseable quoting: fall back to a regex
        pattern = re.escape(name) + r"""(?:=|\s+)(?:'([^']*)'|"((?:[^"\\]|\\.)*)"|(\S+))"""
        for match in re.finditer(pattern, script):
            found.setdefault(name, []).append(next(g for g in match.groups() if g is not None))
    return found


def numeric_options(argv):
    """{--flag: value} for every option whose value parses as a number (used for CLI action deltas)."""
    params = {}
    for i, tok in enumerate(argv):
        if not tok.startswith("-") or len(tok) < 2 or _number(tok) is not None:
            continue
        name, sep, value = tok.partition("=")
        if not sep:
            value = argv[i + 1] if i + 1 < len(argv) else None
        number = _number(value) if value is not None else None
        if number is not None:
            params[name] = number
    return params


def _number(text):
    try:
        return float(text) if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", str(text)) else None
    except ValueError:
        return None


def signature(script: str) -> str:
    """Short, stable label for a command, e.g. ``python robot_ctl.py cell`` or ``ros2 topic pub``."""
    labels = []
    for argv in commands(skeleton(script))[:4]:
        if argv[0] in ("cd", "export", "set", "source", ".", "echo", "printf", "sleep", "true") \
                or argv[0] in ("cat", "tee") and ("<<" in argv or ">" in argv):
            continue
        redirect = next((i for i, tok in enumerate(argv) if i and set(tok) <= set("<>&")), len(argv))
        argv = argv[:redirect]
        prog = os.path.basename(argv[0])
        label = [prog]
        rest = argv[1:]
        if PYTHON_RE.match(prog) or prog in ("bash", "sh", "node", "uv", "conda", "poetry"):
            while rest and rest[0].startswith("-") and rest[0] not in ("-c", "-m", "-"):
                rest = rest[1:]
            if rest and rest[0] == "-":
                label.append("-")
                rest = rest[1:]
            elif rest and rest[0] in ("-c", "-m"):
                label.append(rest[0] + (" " + rest[1] if rest[0] == "-m" and len(rest) > 1 else ""))
                rest = rest[2:]
            elif rest:
                label.append(os.path.basename(rest[0]))
                rest = rest[1:]
        skip = False
        for i, tok in enumerate(rest):
            if skip:
                skip = False
            elif tok.startswith("-"):
                skip = "=" not in tok and tok not in BOOLEAN_FLAGS
            elif re.fullmatch(r"[A-Za-z][\w-]*", tok):
                label.append(tok)
                # Binaries often nest subcommands (ros2 topic pub, docker compose up); scripts rarely do.
                if len(label) == 2 and i + 1 < len(rest) and re.fullmatch(r"[A-Za-z][\w-]*", rest[i + 1]):
                    label.append(rest[i + 1])
                break
        labels.append(" ".join(label))
    return " && ".join(labels) if labels else (script.strip().split("\n")[0][:60] or "<empty>")


def inline_code(script: str):
    """(code, source) for ``python -c CODE`` or ``python - <<EOF``/``python <<EOF`` heredocs."""
    for header, body, _ in heredocs(script)[0]:
        cmds = commands(header.split("<<")[0])
        argv = cmds[-1] if cmds else []
        if argv and PYTHON_RE.match(os.path.basename(argv[0])) and all(t.startswith("-") for t in argv[1:]):
            return body, "heredoc_stdin"
    for argv in commands(script):
        if PYTHON_RE.match(os.path.basename(argv[0])) and "-c" in argv:
            index = argv.index("-c")
            if index + 1 < len(argv):
                return argv[index + 1], "python_-c"
    return None, None


def referenced_files(script: str, file_options=()):
    """Files the command runs or passes as code: interpreter scripts and values of ``file_options``."""
    paths = []
    for argv in commands(script):
        prog = os.path.basename(argv[0])
        if PYTHON_RE.match(prog) or prog in ("bash", "sh"):
            rest = [t for t in argv[1:] if not t.startswith("-")]
            if rest and re.search(r"\.(py|sh)$", rest[0]):
                paths.append(rest[0])
        elif re.search(r"\.(py|sh)$", argv[0]):
            paths.append(argv[0])
        for i, tok in enumerate(argv):
            for name in file_options:
                if tok == name and i + 1 < len(argv):
                    paths.insert(0, argv[i + 1])
                elif tok.startswith(name + "="):
                    paths.insert(0, tok[len(name) + 1:])
    return paths


def file_writes(script: str):
    """[(path, content, source)] for ``cat > f <<EOF`` style writes and ``apply_patch <<EOF`` blocks."""
    writes = []
    for header, body, _ in heredocs(script)[0]:
        if re.search(r"(^|[\s;&|])apply_patch\b", header):
            writes += patch_writes(body)
            continue
        cmds = commands(header.split("<<")[0])
        prog = os.path.basename(cmds[-1][0]) if cmds else ""
        tee = re.search(r"\btee\s+(?:-a\s+)?(['\"]?)([^\s'\";&|<>]+)\1", header)
        match = REDIRECT_RE.search(header) if prog == "cat" else None
        target = tee.group(2) if tee else match.group(2) if match else None
        if target and target not in ("/dev/null", "&1", "&2"):
            writes.append((target, body + "\n", "heredoc"))
    return writes


def patch_writes(patch: str):
    """[(path, content, source)] from an apply_patch envelope. Updates keep only the patch text."""
    writes, path, mode, lines = [], None, None, []

    def flush():
        if path is None:
            return
        if mode == "add":
            writes.append((path, "\n".join(l[1:] if l.startswith("+") else l for l in lines) + "\n", "apply_patch_add"))
        elif mode == "update":
            writes.append((path, "\n".join(lines) + "\n", "apply_patch_update"))

    for line in (patch or "").split("\n"):
        match = re.match(r"^\*\*\* (Add|Update|Delete) File: (.+)$", line)
        if match:
            flush()
            mode, path, lines = match.group(1).lower(), match.group(2).strip(), []
        elif line.startswith("*** End Patch") or line.startswith("*** Begin Patch"):
            continue
        elif line.startswith("*** Move to: ") and mode == "update":
            path = line[len("*** Move to: "):].strip()
        elif path is not None:
            lines.append(line)
    flush()
    return writes


def resolve_path(path, workdir):
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path) and workdir:
        path = os.path.join(str(workdir), path)
    return os.path.normpath(path)
