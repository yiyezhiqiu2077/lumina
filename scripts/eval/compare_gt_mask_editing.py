#!/usr/bin/env python3
"""Combine four GT-mask hard-lock evaluator summaries without regenerating images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "SUMMARY"), required=True)
    args = parser.parse_args()
    rows = []
    for label, summary in args.model:
        payload = json.loads(Path(summary).read_text(encoding="utf-8"))
        rows.append({
            "Model": label,
            "Edit Token Acc": payload.get("edit_token_accuracy"),
            "Inside L1": payload.get("inside_l1_target"),
            "Inside PSNR": payload.get("inside_psnr_target"),
            "Inside L1 vs Target Recon": payload.get("inside_l1_target_recon"),
            "Boundary L1": payload.get("boundary_l1_source_recon"),
            "Full PSNR": payload.get("full_psnr_target"),
            "Seconds / sample": payload.get("seconds"),
            "Outside Token Acc": payload.get("outside_token_accuracy"),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
