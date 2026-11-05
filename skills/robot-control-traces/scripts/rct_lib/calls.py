"""Action calls inside agent-written Python, their literal arguments, and step-to-step parameter changes.

Values are resolved statically: literals, simple arithmetic, ``np.array([...])`` and names assigned to such values
earlier in the same code or in earlier steps of the episode (the controlling namespace usually persists). Anything
else is kept as its source expression, so a call record is always faithful to what the agent wrote.
"""
from __future__ import annotations

import ast
import io
import math
import re
import tokenize

ARRAY_FUNCS = {"np.array", "np.asarray", "numpy.array", "numpy.asarray", "list", "tuple", "np.float64", "float"}
MAX_CALLS = 200


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(dotted(node.func) + "()")
    else:
        return "<expr>"
    return ".".join(reversed(parts))


class _Unknown(Exception):
    pass


def _value(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_value(e, env) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {str(_value(k, env)): _value(v, env) for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _Unknown
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _value(node.operand, env)
        return _arith(0, value, ast.Sub()) if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        return _arith(_value(node.left, env), _value(node.right, env), node.op)
    if isinstance(node, ast.Call) and dotted(node.func) in ARRAY_FUNCS and len(node.args) == 1 and not node.keywords:
        return _value(node.args[0], env)
    if isinstance(node, ast.Call) and dotted(node.func) in ("np.radians", "math.radians") and len(node.args) == 1:
        return _map(_value(node.args[0], env), math.radians)
    raise _Unknown


def _map(value, fn):
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return fn(value)
    raise _Unknown


def _arith(a, b, op):
    ops = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
           ast.Div: lambda x, y: x / y}
    fn = ops.get(type(op))
    if fn is None:
        raise _Unknown
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise _Unknown
        return [_arith(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_arith(x, b, op) for x in a]
    if isinstance(b, list):
        return [_arith(a, y, op) for y in b]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
        try:
            return round(fn(a, b), 10)
        except ZeroDivisionError:
            raise _Unknown
    raise _Unknown


def _arg(node, env, source):
    try:
        return _value(node, env)
    except (_Unknown, RecursionError):
        return {"expr": ast.get_source_segment(source, node) or ast.unparse(node)}


def _assign(stmt, env):
    targets = []
    if isinstance(stmt, ast.Assign):
        targets, value = stmt.targets, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets, value = [stmt.target], stmt.value
    else:
        return
    for target in targets:
        if isinstance(target, ast.Name):
            try:
                env[target.id] = _value(value, env)
            except (_Unknown, RecursionError):
                env.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            for t, v in zip(target.elts, value.elts):
                if isinstance(t, ast.Name):
                    try:
                        env[t.id] = _value(v, env)
                    except (_Unknown, RecursionError):
                        env.pop(t.id, None)


def extract(code: str, pattern, env: dict):
    """Action calls matching ``pattern`` (regex on the dotted name), in source order. Updates ``env`` in place."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return [], "syntax_error"
    calls = []
    for stmt in tree.body:
        local = dict(env)
        nodes = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                nodes.append(node)
        loops = {id(n) for loop in ast.walk(stmt) if isinstance(loop, (ast.For, ast.While))
                 for n in ast.walk(loop) if isinstance(n, ast.Call)}
        defs = {id(n) for fn in ast.walk(stmt) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                for n in ast.walk(fn) if isinstance(n, ast.Call)}
        for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset)):
            name = dotted(node.func)
            if not pattern.search(name) or len(calls) >= MAX_CALLS:
                continue
            record = {"func": name, "line": node.lineno,
                      "args": [_arg(a, local, code) for a in node.args if not isinstance(a, ast.Starred)],
                      "kwargs": {k.arg: _arg(k.value, local, code) for k in node.keywords if k.arg}}
            if id(node) in loops:
                record["in_loop"] = True
            if id(node) in defs:
                record["in_def"] = True
            calls.append(record)
        _assign(stmt, env)
    return calls, None


def comments(code: str, limit=40):
    """Comment texts in the code (the agent's inline rationale)."""
    found = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                text = tok.string.lstrip("#").strip()
                if text and not text.startswith("!") and not re.fullmatch(r"[-=*#\s]+", text):
                    found.append(text)
                    if len(found) >= limit:
                        break
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


# ---- adjustments ----------------------------------------------------------------------------------------------

def _selector(name, value):
    """Short identifier-like strings (arm="right", camera="head") select what a call acts on; ids and prose do not."""
    return isinstance(value, str) and len(value) <= 32 and not re.search(r"\s", value) \
        and not _id_name(name)


def _id_name(name):
    name = (name or "").lower()
    return name == "id" or name.endswith("_id")


def _free_text(value):
    return isinstance(value, str) and (len(value) > 32 or bool(re.search(r"\s", value)))


def call_key(call):
    """Calls are comparable when they have the same function and the same selector arguments (e.g. arm)."""
    strings = sorted((k, v) for k, v in call.get("kwargs", {}).items() if _selector(k, v))
    positional = [v for v in call.get("args", []) if _selector(None, v)]
    return call["func"] + "|" + repr(positional) + repr(strings)


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vector(v):
    return isinstance(v, list) and v and all(_numeric(x) for x in v)


def diff_value(old, new):
    if old == new:
        return None
    if _numeric(old) and _numeric(new):
        return {"from": old, "to": new, "delta": round(new - old, 6)}
    if _vector(old) and _vector(new) and len(old) == len(new):
        delta = [round(b - a, 6) for a, b in zip(old, new)]
        return {"from": old, "to": new, "delta": delta, "norm": round(math.sqrt(sum(d * d for d in delta)), 6)}
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        parts = [diff_value(a, b) for a, b in zip(old, new)]
        if any(parts):
            return {"items": parts}
        return None
    if _symbolic(old) or _symbolic(new):
        return {"from": _shown(old), "to": _shown(new), "symbolic": True}
    return {"from": old, "to": new}


def _symbolic(v):
    return isinstance(v, dict) and set(v) == {"expr"}


def _shown(v):
    return v["expr"] if _symbolic(v) else v


def diff_call(old, new):
    changes = {}
    for i in range(max(len(old.get("args", [])), len(new.get("args", [])))):
        a = old["args"][i] if i < len(old.get("args", [])) else None
        b = new["args"][i] if i < len(new.get("args", [])) else None
        change = diff_value(a, b)
        if change:
            changes[f"args[{i}]"] = change
    for name in sorted(set(old.get("kwargs", {})) | set(new.get("kwargs", {}))):
        a, b = old.get("kwargs", {}).get(name), new.get("kwargs", {}).get(name)
        if _free_text(a) and _free_text(b) or _id_name(name):
            continue
        change = diff_value(a, b)
        if change:
            changes[name] = change
    return changes


class AdjustmentTracker:
    """Remembers the last call per key across steps and reports what changed when it is called again."""

    def __init__(self):
        self.last = {}

    def step(self, step_index, calls):
        adjustments, latest = [], {}
        for call in calls:
            key = call_key(call)
            previous = self.last.get(key)
            if previous is not None and key not in latest:
                changes = diff_call(previous[1], call)
                if changes:
                    adjustments.append({"func": call["func"], "key": key.split("|", 1)[1],
                                        "previous_step": previous[0], "line": call.get("line"), "changes": changes})
            latest[key] = (step_index, call)
        self.last.update(latest)
        return adjustments
