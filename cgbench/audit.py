"""Can this results directory be used to reconstruct the run?

Answers with checks, not prose. Two different questions get two different verdicts:

  reproduce  - same numbers, same machine, same weights, same code  (needs pins)
  replay     - what was run, in what order, with what outcome       (needs records)

A day of results that can be replayed but not reproduced is still useful, but you
must know which one you have before you quote a number from it.
"""
from __future__ import annotations

import json

from .harness import OUT, RUNS, rows


def _ok(cond: bool, name: str, detail: str = "") -> tuple[str, str, str]:
    return (("PASS" if cond else "WARN"), name, detail)


def collect() -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    rs = rows()

    env = next((r.get("env", {}) for r in rs if r["key"] == "env:start"), {})
    runs = [r for r in rs if r.get("status") in ("ok", "error")]

    checks.append(_ok(bool(runs), "results.jsonl has runs", f"{len(runs)} runs"))
    checks.append(_ok(bool(env), "env snapshot", "env:start record"))

    git = (env.get("git") or {}) if env else {}
    checks.append(_ok(bool(git.get("commit")), "code commit pinned",
                      git.get("commit") or "git metadata missing (tarball?)"))
    checks.append(_ok(git.get("dirty") is False, "working tree clean",
                      "dirty" if git.get("dirty") else "clean/unknown"))

    checks.append(_ok((OUT / "requirements.lock").exists(), "dependency lock",
                      "results/requirements.lock"))

    for key in ("torch", "cuda", "diffusers", "cc", "vram_gib"):
        checks.append(_ok(bool(env.get(key)), f"env.{key}", str(env.get(key))))

    revs = {r.get("model_revision") for r in runs if r.get("model")}
    unknown = [r.get("model") for r in runs if r.get("model") and not r.get("model_revision")]
    checks.append(_ok(bool(revs) and not unknown, "weights pinned (HF revision)",
                      f"{len(revs)} revision(s)" + (f", {len(unknown)} unresolved" if unknown else "")))

    bad = [r["key"] for r in runs if r.get("status") == "ok"
           and not (r.get("seconds") and r.get("peak_gib") is not None)]
    checks.append(_ok(not bad, "ok runs carry timing + memory", f"{len(bad)} incomplete"))

    no_meta = [r["key"] for r in runs if r.get("status") == "ok"
               and not all(k in r for k in ("width", "height", "frames", "fps", "steps", "prompt"))]
    checks.append(_ok(not no_meta, "ok runs carry their config", f"{len(no_meta)} incomplete"))

    errs = [r for r in runs if r.get("status") == "error"]
    thin = [r["key"] for r in errs if not r.get("error")]
    checks.append(_ok(not thin, "failures recorded with a reason",
                      f"{len(errs)} failures, {len(thin)} without an error string"))

    pub = OUT / "publish"
    for f in ("report.md", "metrics.csv", "frontier.json"):
        checks.append(_ok((pub / f).exists(), f"publish/{f}", ""))

    casts = list((OUT / "captures").glob("*.cast")) if (OUT / "captures").exists() else []
    checks.append(_ok(bool(casts), "podcast captures", f"{len(casts)} cast file(s)"))

    phases = {r.get("phase") for r in runs}
    checks.append(_ok(len(phases) >= 3, "phase coverage", ", ".join(sorted(p for p in phases if p))))

    return checks


def verdict(checks: list[tuple[str, str, str]]) -> tuple[str, str]:
    kinds = {k for k, _, _ in checks}
    if "FAIL" in kinds:
        return "cannot replay", "缺少基础记录，连过程都还原不了"
    if any(name in ("code commit pinned", "weights pinned (HF revision)", "dependency lock")
           and k != "PASS" for k, name, _ in checks):
        return "replay only", "能还原“跑了什么、结果如何”，但不能保证重跑得到同样的数"
    return "reproducible", "代码、依赖、权重都已钉死，同一台机器可以复现"


def main() -> int:
    checks = collect()
    width = max(len(n) for _, n, _ in checks) + 2
    print(f"\n{OUT}\n" + "=" * 74)
    for status, name, detail in checks:
        mark = {"PASS": " ok ", "WARN": "warn"}.get(status, status)
        print(f"[{mark}] {name.ljust(width)} {detail}")
    kind, why = verdict(checks)
    print("=" * 74)
    print(f"verdict: {kind} — {why}\n")
    return 0 if kind != "cannot replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
