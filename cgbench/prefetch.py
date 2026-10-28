"""Download shortlisted weights before the clock starts.

On a metered, 24-hour box the cheapest thing you can buy is a warm cache: a 183 GiB
shortlist at 1000 Mbps is ~26 minutes of pure download, at 300 Mbps it is ~85, and
at 100 Mbps it eats four and a half hours of a day you already paid for.

Run this the night before, or right after the instance boots and before `run_day.sh`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .catalog import load_shortlist
from .harness import OUT

GIB = 1024 ** 3


def plan(top: int | None = None) -> list[dict]:
    items = load_shortlist()
    if not items:
        raise SystemExit("no catalog shortlist; run `python -m cgbench catalog` first")
    if top:
        items = items[:top]
    return items


def run(top: int | None = None, dry_run: bool = False) -> int:
    items = plan(top)
    total_gib = sum(e.get("est_gib_bf16") or 0 for e in items)
    print(f"{'model':<48}{'est GiB':>9}")
    print("-" * 58)
    for e in items:
        print(f"{e['id'][:47]:<48}{e.get('est_gib_bf16') or 0:>9}")
    print(f"\ntotal ~{total_gib:.0f} GiB")
    for mbps in (1000, 300, 100):
        minutes = total_gib * 8.59 / (mbps / 1000) / 60
        print(f"  ~{minutes:.0f} min at {mbps} Mbps")

    if dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub missing: pip install huggingface_hub", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"\nendpoint: {endpoint}")
    failed = []
    revisions: dict[str, str | None] = {}
    for e in items:
        print(f"\n>> {e['id']}", flush=True)
        try:
            path = snapshot_download(
                repo_id=e["id"],
                endpoint=endpoint,
                max_workers=4,
                resume_download=True,
            )
            # snapshot dirs are named by commit sha: this is the weights pin, offline
            sha = Path(path).name if "snapshots" in Path(path).parts else None
            revisions[e["id"]] = sha
            print(f"   cached -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(e["id"])

    if revisions:
        p = OUT / "revisions.json"
        p.write_text(json.dumps(revisions, indent=2), encoding="utf-8")
        print(f"\nrevisions pinned -> {p}")

    if failed:
        print(f"\n{len(failed)} failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("\nall shortlisted weights cached")
    return 0
