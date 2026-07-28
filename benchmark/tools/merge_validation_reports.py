#!/usr/bin/env python3
"""Merge multiple validate_instances reports into one (for generate_index --validation-report).

The 2-agent backbone and the 3-4 agent scaling axis are validated in separate runs
(separate report dirs). generate_index attaches status from a single report, so this
merges their ``results`` maps (later reports win on key conflicts) and recomputes the
pass/fail counts. Usage: merge_validation_reports.py out.json in1.json in2.json ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    out_path = Path(sys.argv[1])
    merged = {}
    for p in sys.argv[2:]:
        rep = json.loads(Path(p).read_text())
        merged.update(rep.get("results", {}))
    passed = sum(1 for v in merged.values() if v.get("success") and v.get("legal_plan"))
    report = {
        "merged_from": sys.argv[2:],
        "count": len(merged),
        "passed": passed,
        "failed": len(merged) - passed,
        "results": merged,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {passed}/{len(merged)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
