#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "experiments/EXPERIMENT_MATRIX.csv"
PRIORITY = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
DONE = {"completed"}
BLOCKED_PREFIX = "blocked_"


def main() -> int:
    with MATRIX.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["experiment_id"]: row for row in rows}
    candidates=[]
    for row in rows:
        if row["status"] not in {"ready", "planned"}:
            continue
        deps=[x.strip() for x in row["depends_on"].split(',') if x.strip()]
        unknown=[d for d in deps if d not in by_id]
        unmet=[d for d in deps if d in by_id and by_id[d]["status"] not in DONE]
        if unknown or unmet:
            continue
        candidates.append(row)
    candidates.sort(key=lambda r:(PRIORITY.get(r['priority'],99), r['experiment_id']))
    if not candidates:
        print("No dependency-free ready/planned experiment. Consult TASKS.md and BLOCKERS.md.")
        return 0
    print("Eligible next experiments:")
    for row in candidates[:10]:
        print(f"{row['priority']} {row['experiment_id']} [{row['stage']}] {row['description']} | config={row['config'] or 'none'}")
    return 0

if __name__ == '__main__': raise SystemExit(main())
