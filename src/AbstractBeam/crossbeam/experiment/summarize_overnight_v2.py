#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Data/janis.aiad/abstractrl/src/AbstractBeam/crossbeam/experiment/runs/gcp_trm_scaleup_v2")
COMPARE = ROOT / "overnight_compare"
OUT = ROOT / "overnight_summary.json"


def last_valid_loss(train_log: Path) -> float | None:
    if not train_log.exists():
        return None
    lines = [ln for ln in train_log.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    best = None
    for ln in lines:
        row = json.loads(ln)
        v = row.get("valid_loss")
        if isinstance(v, (int, float)):
            best = float(v) if best is None else min(best, float(v))
    return best


def hard_score(hard_json: Path) -> float | None:
    if not hard_json.exists():
        return None
    obj = json.loads(hard_json.read_text())
    blocks = obj.get("blocks", [])
    rates = [float(b.get("solved_rate", 0.0)) for b in blocks if isinstance(b, dict)]
    if not rates:
        return None
    return sum(rates) / len(rates)


def main() -> None:
    rows = []
    if COMPARE.exists():
        for d in sorted(COMPARE.glob("model*_bg_*")):
            train_log = d / "train_log.jsonl"
            hard_json = Path(str(d) + ".hard.json")
            rows.append(
                {
                    "run_dir": str(d),
                    "best_valid_loss": last_valid_loss(train_log),
                    "hard_mean_solved_rate": hard_score(hard_json),
                    "train_log_exists": train_log.exists(),
                    "hard_report_exists": hard_json.exists(),
                }
            )

    rows.sort(
        key=lambda r: (
            -(r["hard_mean_solved_rate"] if isinstance(r["hard_mean_solved_rate"], (int, float)) else -1.0),
            (r["best_valid_loss"] if isinstance(r["best_valid_loss"], (int, float)) else 1e9),
        )
    )
    payload = {"rows": rows}
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"wrote": str(OUT), "num_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
