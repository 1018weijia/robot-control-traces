#!/usr/bin/env python3
"""Turn Codex conversation logs of AI-driven robot control into a step-level dataset.

  rct.py scan [--since 2026-01-01]               rollouts that controlled something, with act/observe counts
  rct.py scan --detail ROLLOUT                    every command/tool signature in one rollout and its role
  rct.py export --auto --out DIR                  export every rollout found by scan
  rct.py export --rollout R1 R2 --out DIR         export specific rollouts
  rct.py validate DIR                             schema check, coverage and a redacted sample
  rct.py profiles                                 list built-in profiles

Python 3.9+, standard library only. Reads $CODEX_HOME (default ~/.codex); never reads auth.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rct_lib import codex, export as export_mod, profile as profile_mod, steps as steps_mod  # noqa: E402
from rct_lib.redact import Redactor  # noqa: E402
from rct_lib.validate import format_report, validate  # noqa: E402


def summarise(path, profiles, auto):
    rollout = codex.read_rollout(path, keep_images=False)
    profile = profile_mod.build(rollout, profiles, auto)
    steps_mod.prepare(rollout, profile)
    calls = [e for e in rollout.events if e.kind == "tool_call"]
    roles = {}
    for event in calls:
        roles[event.role] = roles.get(event.role, 0) + 1
    top = {}
    for event in calls:
        if event.role == "act":
            top[event.signature] = top.get(event.signature, 0) + 1
    return rollout, profile, {
        "rollout": str(path), "started": rollout.started, "models": rollout.models,
        "user_messages": sum(1 for e in rollout.events if e.kind == "user_message"),
        "tool_calls": len(calls), "act": roles.get("act", 0), "observe": roles.get("observe", 0),
        "result": roles.get("result", 0), "images": len(rollout.images), "profiles": profile.names,
        "top_act": max(top, key=top.get) if top else None, "warnings": len(rollout.warnings)}


def cmd_scan(args):
    redactor = Redactor(not args.no_redact)
    if args.detail:
        rollout, profile, summary = summarise(Path(args.detail), args.profile, not args.no_auto_profile)
        table = {}
        for event in rollout.events:
            if event.kind != "tool_call":
                continue
            key = (event.signature or event.tool or "?", event.role, event.rule)
            row = table.setdefault(key, {"count": 0, "example": event.command or json.dumps(event.args, default=str)})
            row["count"] += 1
        rows = sorted(table.items(), key=lambda kv: -kv[1]["count"])
        if args.json:
            print(json.dumps({"summary": summary, "signatures": [
                {"signature": k[0], "role": k[1], "rule": k[2], "count": v["count"],
                 "example": redactor.text(v["example"][:300])} for k, v in rows]}, ensure_ascii=False, indent=2))
            return 0
        print(f"{summary['rollout']}\n  models {summary['models']}  profiles {summary['profiles']}  "
              f"tool calls {summary['tool_calls']}  act {summary['act']}  observe {summary['observe']}")
        print(f"  {'count':>5}  {'role':<8} {'rule':<30} signature / example")
        for (signature, role, rule), row in rows:
            example = redactor.text(" ".join(row["example"].split())[:110])
            print(f"  {row['count']:>5}  {role:<8} {rule:<30} {signature}\n{'':>48}{example}")
        return 0
    home = codex.codex_home(args.codex_home)
    paths = codex.find_rollouts(home, args.since)
    results = []
    for path in paths:
        try:
            _, _, summary = summarise(path, args.profile, not args.no_auto_profile)
        except Exception as exc:
            summary = {"rollout": str(path), "error": f"{type(exc).__name__}: {exc}", "act": 0}
        if summary["act"] or args.all:
            results.append(summary)
    if args.json:
        print(json.dumps({"codex_home": str(home), "rollouts_scanned": len(paths), "rollouts": results},
                         ensure_ascii=False, indent=2))
        return 0
    print(f"{home}: {len(paths)} rollouts scanned, {len(results)} listed")
    for s in results:
        if "error" in s:
            print(f"  ERROR {s['rollout']}: {s['error']}")
            continue
        print(f"  {(s['started'] or '')[:19]:<19}  act {s['act']:>4}  observe {s['observe']:>4}  "
              f"images {s['images']:>4}  {','.join(s['models'])[:24]:<24}  {Path(s['rollout']).name}\n"
              f"      profiles {s['profiles']}  top act: {s['top_act']}")
    return 0


def cmd_export(args):
    if args.auto:
        paths = codex.find_rollouts(codex.codex_home(args.codex_home), args.since)
    else:
        paths = [Path(p) for p in args.rollout or []]
    if not paths:
        print("no rollouts selected (use --rollout FILE... or --auto)", file=sys.stderr)
        return 2
    pairs = {}
    for flag, items in (("--redact-path", args.redact_path), ("--remap-path", args.remap_path)):
        pairs[flag] = []
        for item in items:
            old, sep, new = item.partition("=")
            if not sep:
                print(f"{flag} expects OLD=NEW, got {item!r}", file=sys.stderr)
                return 2
            pairs[flag].append((old, new))
    options = export_mod.Options(profiles=tuple(args.profile), auto_profiles=not args.no_auto_profile,
                                 images=args.images, redact=not args.no_redact, path_map=tuple(pairs["--redact-path"]),
                                 max_chars=args.max_chars, use_adapters=not args.no_adapters,
                                 read_local_files=args.read_local_files, min_act_steps=args.min_act_steps,
                                 remap_paths=tuple(pairs["--remap-path"]))
    manifest = export_mod.export(paths, args.out, options, overwrite=args.overwrite)
    print(f"wrote {len(manifest['episodes'])} episode(s) from {len(paths)} rollout(s) to {args.out}")
    errors, totals, samples = validate(args.out)
    print(format_report(errors, totals, samples))
    return 1 if errors else 0


def cmd_validate(args):
    errors, totals, samples = validate(args.out)
    print(format_report(errors, totals, samples))
    return 1 if errors else 0


def cmd_profiles(_args):
    for name in profile_mod.builtin_names():
        spec = profile_mod.load_spec(name)
        print(f"{name:<16} {spec.get('description', '')}")
    print(f"(directory: {profile_mod.PROFILE_DIR})")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def common(p, top=False):
        # Accepted before or after the subcommand; SUPPRESS keeps a subparser from resetting top-level values.
        default = (lambda value: value) if top else (lambda value: argparse.SUPPRESS)
        p.add_argument("--codex-home", default=default(None), help="default: $CODEX_HOME or ~/.codex")
        p.add_argument("--since", default=default(None), help="only rollouts dated YYYY-MM-DD or later")
        p.add_argument("--profile", action="append", default=default([]),
                       help="built-in profile name or JSON path (repeat)")
        p.add_argument("--no-auto-profile", action="store_true", default=default(False),
                       help="do not auto-detect built-in profiles")
        p.add_argument("--no-redact", action="store_true", default=default(False),
                       help="keep paths/secrets verbatim (private use only)")

    common(parser, top=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="find rollouts that controlled something")
    common(scan)
    scan.add_argument("--detail", metavar="ROLLOUT", help="signature/role table for one rollout")
    scan.add_argument("--all", action="store_true", help="also list rollouts without act calls")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=cmd_scan)

    exp = sub.add_parser("export", help="write the dataset")
    common(exp)
    exp.add_argument("--rollout", nargs="+", help="rollout JSONL files")
    exp.add_argument("--auto", action="store_true", help="every rollout under $CODEX_HOME (filtered by --since)")
    exp.add_argument("--out", required=True, type=Path)
    exp.add_argument("--overwrite", action="store_true", help="replace a previous export in --out")
    exp.add_argument("--images", choices=("seen", "all", "none"), default="seen",
                     help="seen: images the model received (default); all: also run-directory frames from adapters")
    exp.add_argument("--redact-path", action="append", default=[], metavar="OLD=NEW",
                     help="replace a path or name prefix in all exported text")
    exp.add_argument("--remap-path", action="append", default=[], metavar="RECORDED=LOCAL",
                     help="adapters look for run directories recorded under RECORDED in LOCAL instead")
    exp.add_argument("--max-chars", type=int, default=4000, help="clip tool outputs to this many characters")
    exp.add_argument("--no-adapters", action="store_true", help="transcript only, ignore run directories")
    exp.add_argument("--read-local-files", action="store_true",
                     help="read code files the transcript does not contain from disk (current contents)")
    exp.add_argument("--min-act-steps", type=int, default=1, help="skip rollouts with fewer act steps")
    exp.set_defaults(func=cmd_export)

    val = sub.add_parser("validate", help="check an export")
    val.add_argument("out", type=Path)
    val.set_defaults(func=cmd_validate)

    prof = sub.add_parser("profiles", help="list built-in profiles")
    prof.set_defaults(func=cmd_profiles)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
