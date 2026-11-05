"""Optional framework adapters that enrich transcript-derived steps with ground truth from run directories.

An adapter module exposes ``NAME`` and ``enrich(steps, rollout, options) -> dict``. ``enrich`` may fill
``step["enrichment"][NAME]``, ``step["observation"]["state"]``, a missing ``step["action"]["code"]`` and
``step["_episode_hint"]``; it returns ``{"episodes": {hint: info}, "stats": {...}}``. Adapters must never fail an
export: missing or unreadable run directories are reported in the stats and the steps stay transcript-only.
"""
from __future__ import annotations

import traceback

from . import cell_queue

REGISTRY = {cell_queue.NAME: cell_queue}


def enrich(names, steps, rollout, options):
    report = {}
    for name in names:
        adapter = REGISTRY.get(name)
        if adapter is None:
            report[name] = {"stats": {"error": "unknown adapter"}, "episodes": {}}
            continue
        try:
            report[name] = adapter.enrich(steps, rollout, options)
        except Exception:
            report[name] = {"stats": {"error": traceback.format_exc(limit=3)}, "episodes": {}}
    return report
