"""Can this results directory be used to reconstruct the run?

Answers with checks, not prose. Two different questions get two different verdicts:

  reproducible - same code, same deps, same weights, same machine -> same numbers
  replay only  - what was run, in what order, with what outcome
  cannot replay- not even the above

The audit works on a live results/ directory or on a snapshot pulled to a laptop,
so the person reading the report can check the claim instead of trusting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .harness import OUT


def _rows(out: Path) -> list[dict]:
    f = out / "results.jsonl"
    if not f.exists():
        return []
    rs = []
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rs.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return rs


def _check(cond: bool, name: str, detail: str = "") -> dict:
    return {"status": "PASS" if cond else "WARN", "name": name, "detail": detail}


def _verify_manifest(out: Path, doc: dict) -> tuple[int, int, list[str]]:
    good = bad = 0
    broken: list[str] = []
    for f in doc.get("files", []):
        if not f.get("sha256"):
            continue
        p = out / f["path"]
        if not p.exists():
            broken.append(f["path"])
            bad += 1
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]:
            good += 1
        else:
            broken.append(f["path"])
            bad += 1
    return good, bad, broken[:5]


def collect(out: Path, verify: bool = True) -> tuple[list[dict], dict]:
    rs = _rows(out)
    checks: list[dict] = []

    runs = [r for r in rs if r.get("status") in ("ok", "error")]
    env = next((r.get("env", {}) for r in rs if r.get("key") == "env:start"), {})
    git = (env.get("git") or {}) if env else {}

    checks.append(_check(bool(runs), "results.jsonl has runs", f"{len(runs)} runs"))
    checks.append(_check(bool(env), "env snapshot", "env:start record"))
    checks.append(_check(bool(git.get("commit")), "code commit pinned",
                         git.get("commit") or "no git metadata (tarball?)"))
    checks.append(_check(git.get("dirty") is False, "working tree clean",
                         "dirty" if git.get("dirty") else "clean/unknown"))
    checks.append(_check((out / "requirements.lock").exists(), "dependency lock",
                         "requirements.lock"))
    for key in ("torch", "cuda", "diffusers", "cc", "vram_gib"):
        checks.append(_check(bool(env.get(key)), f"env.{key}", str(env.get(key))))

    unknown = [r.get("model") for r in runs if r.get("model") and not r.get("model_revision")]
    revs = {r.get("model_revision") for r in runs if r.get("model_revision")}
    checks.append(_check(not unknown, "weights pinned (HF revision)",
                         f"{len(revs)} revision(s)" + (f", {len(unknown)} unresolved" if unknown else "")))

    incomplete = [r["key"] for r in runs if r.get("status") == "ok"
                  and not (r.get("seconds") and r.get("peak_gib") is not None)]
    checks.append(_check(not incomplete, "ok runs carry timing + memory", f"{len(incomplete)} incomplete"))

    no_meta = [r["key"] for r in runs if r.get("status") == "ok"
               and not all(k in r for k in ("width", "height", "frames", "fps", "steps", "prompt"))]
    checks.append(_check(not no_meta, "ok runs carry their config", f"{len(no_meta)} incomplete"))

    errs = [r for r in runs if r.get("status") == "error"]
    no_reason = [r["key"] for r in errs if not r.get("error")]
    no_trace = [r["key"] for r in errs
                if r.get("traceback_file") and not (out / r["traceback_file"]).exists()]
    checks.append(_check(not no_reason, "failures carry a reason",
                         f"{len(errs)} failure(s), {len(no_reason)} without error text"))
    checks.append(_check(not no_trace, "failures carry a full traceback",
                         f"{len(no_trace)} missing file(s)"))

    pub = out / "publish"
    for f in ("report.md", "metrics.csv", "frontier.json"):
        checks.append(_check((pub / f).exists(), f"publish/{f}", ""))

    vids = [r for r in runs if r.get("status") == "ok" and str(r.get("video", "")).endswith(".mp4")]
    present = [r for r in vids if r.get("video") and (out / r["video"]).exists()]
    if vids:
        checks.append(_check(len(present) == len(vids), "generated clips present",
                             f"{len(present)}/{len(vids)} (media layer may not be pulled)"))
    else:
        checks.append(_check(False, "generated clips present", "no mp4 in this snapshot"))

    casts = list((out / "captures").glob("*.cast")) if (out / "captures").exists() else []
    checks.append(_check(bool(casts), "podcast captures", f"{len(casts)} cast file(s)"))

    phases = sorted({r["phase"] for r in runs if r.get("phase")})
    checks.append(_check(len(phases) >= 3, "phase coverage", ", ".join(phases)))

    manifest_doc = {}
    mpath = out / "manifest.json"
    checks.append(_check(mpath.exists(), "hash manifest", "manifest.json"))
    if mpath.exists():
        try:
            manifest_doc = json.loads(mpath.read_text(encoding="utf-8"))
            if verify:
                good, bad, broken = _verify_manifest(out, manifest_doc)
                checks.append(_check(bad == 0, "manifest hashes verify",
                                     f"{good} ok, {bad} bad" + (f" ({', '.join(broken)})" if broken else "")))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check(False, "manifest readable", str(exc)[:80]))

    return checks, {"env": env, "manifest": {k: v for k, v in manifest_doc.items() if k != "files"}}


def verdict(checks: list[dict]) -> tuple[str, str]:
    warn = {c["name"] for c in checks if c["status"] != "PASS"}
    if "results.jsonl has runs" in warn or "env snapshot" in warn:
        return "cannot replay", "缺少基础记录，连过程都还原不了"
    pins = {"code commit pinned", "working tree clean", "dependency lock",
            "weights pinned (HF revision)", "hash manifest"}
    if pins & warn:
        return "replay only", "能还原“跑了什么、结果如何”，但不保证重跑得到同样的数"
    return "reproducible", "代码、依赖、权重、清单都可核对，同机可复现"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cgbench audit")
    ap.add_argument("--results", default=None, help="audit a pulled snapshot instead of ./results")
    ap.add_argument("--no-verify", action="store_true", help="skip manifest hash verification")
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args(argv)

    out = Path(args.results).expanduser().resolve() if args.results else OUT
    checks, meta = collect(out, verify=not args.no_verify)
    kind, why = verdict(checks)

    doc = {"results": str(out), "verdict": kind, "reason": why, "checks": checks, **meta}
    try:
        (out / "audit.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        width = max(len(c["name"]) for c in checks) + 2
        print(f"\n{out}\n" + "=" * 76)
        for c in checks:
            mark = {"PASS": " ok ", "WARN": "warn"}.get(c["status"], c["status"])
            print(f"[{mark}] {c['name'].ljust(width)} {c['detail']}")
        print("=" * 76)
        print(f"verdict: {kind} — {why}\n")

    return 0 if kind != "cannot replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
