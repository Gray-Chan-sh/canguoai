"""CLI: python -m cgbench <plan|env|catalog|phase|all|report|shots> [options]"""
from __future__ import annotations

import argparse
import sys

from .harness import env_info
from .spec import OBJECTIVES, PHASES


def cmd_plan(_args) -> int:
    total = sum(p["minutes"] for p in PHASES.values())
    print(f"24h plan — {total} min total\n")
    print(f"{'phase':<12}{'min':>5}  description")
    print("-" * 72)
    for name, p in PHASES.items():
        print(f"{name:<12}{p['minutes']:>5}  {p['desc']}")
    print("\nobjectives")
    print("-" * 72)
    for o in OBJECTIVES:
        print(f"{o['id']}  [{o['phase']}] {o['question']}")
        print(f"     -> {o['answer']}")
    return 0


def cmd_env(_args) -> int:
    import json

    print(json.dumps(env_info(), indent=2, ensure_ascii=False))
    return 0


def cmd_catalog(args) -> int:
    from .catalog import build

    build(top=args.top)
    return 0


def cmd_shots(_args) -> int:
    from .capture import shotlist

    print(shotlist())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cgbench")
    ap.add_argument("what", choices=["plan", "env", "catalog", "phase", "all", "report", "shots"])
    ap.add_argument("phase", nargs="?", help="phase name (for `phase`)")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic engine: validate the plan without weights")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="trade speed for headroom; changes what you are measuring")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="override the phase budget (minutes)")
    ap.add_argument("--from-catalog", action="store_true",
                    help="run the `models` phase over the HF shortlist")
    ap.add_argument("--top", type=int, default=8, help="catalog: shortlist size")
    args = ap.parse_args(argv)

    if args.what == "plan":
        return cmd_plan(args)
    if args.what == "env":
        return cmd_env(args)
    if args.what == "catalog":
        return cmd_catalog(args)
    if args.what == "shots":
        return cmd_shots(args)
    if args.what == "report":
        from .report import build
        build()
        return 0

    from . import probes

    resume = not args.no_resume
    if args.what == "phase":
        if not args.phase:
            print("usage: python -m cgbench phase <name>")
            return 2
        if args.budget_min:
            probes.run_phase(args.phase, dry_run=args.dry_run, resume=resume,
                             cpu_offload=args.cpu_offload, budget_min=args.budget_min,
                             from_catalog=args.from_catalog)
        else:
            probes.run_all(dry_run=args.dry_run, resume=resume,
                           cpu_offload=args.cpu_offload, phases=[args.phase],
                           from_catalog=args.from_catalog)
        return 0

    probes.run_all(dry_run=args.dry_run, resume=resume, cpu_offload=args.cpu_offload,
                   from_catalog=args.from_catalog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
