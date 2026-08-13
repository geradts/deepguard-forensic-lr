#!/usr/bin/env python3
"""DeepGuard DF40 large-scale benchmark orchestrator.

This script intentionally does NOT download the ~93 GB DF40 test set or model
weights. It validates the Drive workspace, records the exact experiment
configuration, and runs the official DF40/DeepfakeBench Xception evaluation
for the requested protocols once the user has placed the official processed
DF40 test data, dataset_json files, and checkpoint on Drive.

The official DF40 repository documents the processed test set, JSON manifests,
and Protocols 1-4. Keep dataset and checkpoint licenses/terms intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROTOCOLS = {
    "p2": ["FSAll_cdf", "FRAll_cdf", "EFSAll_cdf"],
    "p3": ["deepfacelab", "heygen", "whichisreal", "MidJourney", "stargan", "starganv2", "styleclip", "e4e", "CollabDiff"],
}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None, log: Path | None = None) -> int:
    print("$", " ".join(map(str, cmd)))
    with (log.open("a") if log else open(os.devnull, "w")) as lf:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True, type=Path)
    ap.add_argument("--df40-root", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--protocol", choices=["p2", "p3"], default="p3")
    ap.add_argument("--detector", default="xception")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.drive_root
    df40 = args.df40_root
    weights = args.weights
    out = root / "benchmark" / "df40" / args.detector / args.protocol
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    dataset_json = df40 / "preprocessing" / "dataset_json"
    detector_cfg = df40 / "training" / "config" / "detector" / f"{args.detector}.yaml"

    checks = {
        "drive_root": root.exists(),
        "df40_root": df40.exists(),
        "dataset_json": dataset_json.exists(),
        "detector_config": detector_cfg.exists(),
        "weights": weights.exists(),
    }
    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        print("MISSING:", ", ".join(missing))
        print("Expected dataset_json:", dataset_json)
        print("Expected detector config:", detector_cfg)
        print("Expected weights:", weights)
        return 2

    datasets = PROTOCOLS[args.protocol]
    missing_json = [d for d in datasets if not (dataset_json / f"{d}.json").exists()]
    if missing_json:
        print("Missing dataset JSON manifests:", missing_json)
        return 3

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "detector": args.detector,
        "datasets": datasets,
        "df40_root": str(df40),
        "dataset_json": str(dataset_json),
        "detector_config": str(detector_cfg),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "python": sys.version,
        "cwd": os.getcwd(),
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    cmd = [
        sys.executable,
        str(df40 / "training" / "test.py"),
        "--detector_path", str(detector_cfg),
        "--weights_path", str(weights),
        "--test_dataset", *datasets,
    ]
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print("DRY RUN:", " ".join(cmd))
        return 0

    rc = run(cmd, cwd=df40, log=log)
    (out / "exit_code.txt").write_text(str(rc))
    print("Exit code:", rc)
    print("Results/log:", out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
