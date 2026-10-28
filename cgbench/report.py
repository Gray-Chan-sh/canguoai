"""Post-hoc analysis: turn results.jsonl into answers, not tables.

Three artefacts land in results/publish/:
  report.md      the day's conclusions, in the order the objectives were asked
  metrics.csv    one row per run, for pandas/Excel
  frontier.json  machine-readable envelope, for re-use in later decisions
"""
from __future__ import annotations

import csv
import json
import statistics as st
from datetime import datetime
from pathlib import Path

from .harness import OUT, rows
from .spec import LATENCY_BUDGET_S, MODELS, OBJECTIVES, RESOLUTIONS

PUBLISH = OUT / "publish"
PUBLISH.mkdir(parents=True, exist_ok=True)


def _ok(r: dict) -> bool:
    return r.get("status") == "ok"


def _runs() -> list[dict]:
    return [r for r in rows() if r.get("status") in ("ok", "error")]


def _by_phase(rs: list[dict], phase: str) -> list[dict]:
    return [r for r in rs if r.get("phase") == phase]


def _fmt(v, nd=1):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


# ------------------------------------------------------------------ objectives


def objective_o1(rs: list[dict]) -> dict:
    runs = [r for r in _by_phase(rs, "resolution") if _ok(r)]
    feasible = [r for r in runs if r["seconds"] <= LATENCY_BUDGET_S]
    best = max(feasible, key=lambda r: r["width"] * r["height"], default=None)
    per = {}
    for r in runs:
        per[f"{r['width']}x{r['height']}"] = {
            "seconds": r["seconds"],
            "s_per_mpframe": r.get("s_per_mpframe"),
            "peak_gib": r.get("peak_gib"),
        }
    fails = [r for r in _by_phase(rs, "resolution") if not _ok(r)]
    return {
        "max_feasible": f"{best['width']}x{best['height']}" if best else None,
        "budget_s": LATENCY_BUDGET_S,
        "per_resolution": per,
        "failed": [{"label": r["label"], "error": r.get("error", "")[:120]} for r in fails],
    }


def objective_o2(rs: list[dict]) -> dict:
    runs = [r for r in _by_phase(rs, "duration") if _ok(r)]
    if not runs:
        return {"max_seconds": None}
    best = max(runs, key=lambda r: r["frames"])
    ladder = [{"label": r["label"], "seconds": r["seconds"], "frames": r["frames"],
               "peak_gib": r.get("peak_gib")} for r in sorted(runs, key=lambda r: r["frames"])]
    fails = [r["label"] for r in _by_phase(rs, "duration") if not _ok(r)]
    return {"max_seconds": round(best["frames"] / best["fps"], 1),
            "max_frames": best["frames"], "ladder": ladder, "failed": fails}


def objective_o3(rs: list[dict]) -> dict:
    reps = [r for r in _by_phase(rs, "throughput") if _ok(r)]
    if not reps:
        return {"per_hour": None}
    times = [r["seconds"] for r in reps]
    med = st.median(times)
    cold = [r.get("load_s", 0) for r in reps if r.get("cold")]
    amort = st.mean(cold) if cold else 0.0
    return {
        "per_hour": round(3600 / med, 2) if med else None,
        "per_hour_with_load": round(3600 / (med + amort), 2) if med else None,
        "seconds_median": round(med, 1),
        "seconds_stdev": round(st.pstdev(times), 2) if len(times) > 1 else 0.0,
        "n": len(times),
    }


def objective_o4(rs: list[dict]) -> dict:
    peaks = [r["peak_gib"] for r in rs if _ok(r) and r.get("peak_gib")]
    errs = [r for r in rs if not _ok(r)]
    return {
        "peak_gib": max(peaks) if peaks else None,
        "peak_at": max((r for r in rs if _ok(r) and r.get("peak_gib")),
                       key=lambda r: r["peak_gib"], default={}).get("label"),
        "errors": [{"phase": r.get("phase"), "label": r.get("label"),
                    "error": r.get("error", "")[:100]} for r in errs][:12],
        "error_count": len(errs),
    }


def objective_o5(rs: list[dict]) -> dict:
    runs = [r for r in _by_phase(rs, "precision") if _ok(r)]
    table = {}
    for r in runs:
        table[r["dtype"]] = {
            "seconds": r["seconds"],
            "peak_gib": r.get("peak_gib"),
            "clip": (r.get("metrics") or {}).get("clip"),
            "temporal_err": (r.get("metrics") or {}).get("temporal_err"),
        }
    base = table.get("bfloat16", {}).get("seconds")
    for k, v in table.items():
        v["speedup_vs_bf16"] = round(base / v["seconds"], 2) if base and v["seconds"] else None
    return table


def _parse_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return datetime.min


def objective_o6(rs: list[dict]) -> dict:
    runs = [r for r in _by_phase(rs, "soak")]
    ok = [r for r in runs if _ok(r)]
    if len(ok) < 2:
        return {"n": len(ok)}
    ok.sort(key=lambda r: _parse_ts(r["ts"]))
    third = max(1, len(ok) // 3)
    head, tail = ok[:third], ok[-third:]
    h = st.median([r["seconds"] for r in head])
    t = st.median([r["seconds"] for r in tail])
    hp = st.median([r["peak_gib"] for r in head if r.get("peak_gib")] or [0])
    tp = st.median([r["peak_gib"] for r in tail if r.get("peak_gib")] or [0])
    return {
        "n": len(ok),
        "fail_rate_pct": round(100 * (len(runs) - len(ok)) / max(1, len(runs)), 1),
        "head_s": round(h, 1),
        "tail_s": round(t, 1),
        "slowdown_pct": round((t / h - 1) * 100, 1) if h else None,
        "peak_drift_gib": round(tp - hp, 2),
        "span_hours": round((_parse_ts(ok[-1]["ts"]) - _parse_ts(ok[0]["ts"])).total_seconds() / 3600, 1),
    }


def objective_o7(rs: list[dict]) -> list[dict]:
    out = []
    for r in rs:
        if not _ok(r) or not r.get("metrics"):
            continue
        m = r["metrics"]
        out.append({
            "phase": r["phase"], "label": r["label"], "model": r["model"],
            "res": f"{r['width']}x{r['height']}", "frames": r["frames"],
            "clip": m.get("clip"), "temporal_err": m.get("temporal_err"),
            "motion": m.get("motion"), "flicker": m.get("flicker"),
        })
    return out


# ------------------------------------------------------------------ rendering


def _table(header: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def build() -> dict:
    rs = _runs()
    if not rs:
        raise SystemExit("no runs in results/results.jsonl")

    env = next((r["env"] for r in rows() if r["key"] == "env:start"), {})
    frontier = {
        "env": env,
        "o1_resolution": objective_o1(rs),
        "o2_duration": objective_o2(rs),
        "o3_throughput": objective_o3(rs),
        "o4_memory": objective_o4(rs),
        "o5_precision": objective_o5(rs),
        "o6_stability": objective_o6(rs),
        "o7_quality": objective_o7(rs),
    }

    (PUBLISH / "frontier.json").write_text(
        json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV for pandas / Excel
    keys = ["key", "phase", "model", "label", "width", "height", "frames", "fps",
            "steps", "dtype", "status", "seconds", "s_per_it", "peak_gib", "load_s"]
    with (PUBLISH / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["clip", "temporal_err", "motion", "flicker", "sharpness", "error"])
        for r in rs:
            m = r.get("metrics") or {}
            w.writerow([r.get(k) for k in keys] +
                       [m.get("clip"), m.get("temporal_err"), m.get("motion"),
                        m.get("flicker"), m.get("sharpness"), r.get("error", "")])

    md = []
    md.append(f"# 视频生成能力评估 — {env.get('device', '?')}\n")
    md.append(f"- 架构 `{env.get('arch')}` / 计算能力 `{env.get('cc')}` / "
              f"显存 `{env.get('vram_gib')} GiB`")
    md.append(f"- torch `{env.get('torch')}` cuda `{env.get('cuda')}` "
              f"diffusers `{env.get('diffusers')}`")
    md.append(f"- 主机 `{env.get('host')}`，内核 `{env.get('kernel')}`\n")

    o1 = frontier["o1_resolution"]
    md.append("## O1 最大分辨率\n")
    md.append(f"在 ≤ {o1['budget_s']}s/条 的前提下，可达 **{o1['max_feasible'] or '无'}**。\n")
    if o1["per_resolution"]:
        md.append(_table(["分辨率", "耗时 s", "s/百万像素帧", "峰值 GiB"],
                         [[k, v["seconds"], v["s_per_mpframe"], v["peak_gib"]]
                          for k, v in o1["per_resolution"].items()]))
    if o1["failed"]:
        md.append("\n失败：" + "；".join(f"`{f['label']}` {f['error']}" for f in o1["failed"]))
    md.append("")

    o2 = frontier["o2_duration"]
    md.append("## O2 最长时长（720p）\n")
    md.append(f"最长成功 **{o2.get('max_seconds')}s**（{o2.get('max_frames')} 帧）。\n")
    if o2.get("ladder"):
        md.append(_table(["档位", "耗时 s", "帧数", "峰值 GiB"],
                         [[x["label"], x["seconds"], x["frames"], x["peak_gib"]]
                          for x in o2["ladder"]]))
    if o2.get("failed"):
        md.append("\n未完成：" + "、".join(o2["failed"]))
    md.append("")

    o3 = frontier["o3_throughput"]
    md.append("## O3 吞吐\n")
    md.append(f"- 稳态 **{o3.get('per_hour')} 条/小时**（中位 {o3.get('seconds_median')}s，"
              f"σ={o3.get('seconds_stdev')}s，n={o3.get('n')}）")
    md.append(f"- 含模型加载摊销：{o3.get('per_hour_with_load')} 条/小时\n")

    o4 = frontier["o4_memory"]
    md.append("## O4 显存\n")
    md.append(f"- 全场景峰值 **{o4.get('peak_gib')} GiB**（`{o4.get('peak_at')}`）")
    md.append(f"- 失败 {o4.get('error_count')} 次" + (
        "：" + "；".join(f"{e['phase']}/{e['label']}" for e in o4["errors"][:5])
        if o4["errors"] else ""))
    md.append("")

    o5 = frontier["o5_precision"]
    md.append("## O5 精度\n")
    if o5:
        md.append(_table(["精度", "耗时 s", "相对 bf16", "峰值 GiB", "CLIP", "时序误差"],
                         [[k, v["seconds"], v["speedup_vs_bf16"], v["peak_gib"], v["clip"],
                           v["temporal_err"]] for k, v in o5.items()]))
    else:
        md.append("未采集到 precision 阶段数据。")
    md.append("")

    o6 = frontier["o6_stability"]
    md.append("## O6 24h 稳定性\n")
    if o6.get("n"):
        md.append(f"- 样本 {o6['n']} 条，跨 {o6.get('span_hours')} 小时，失败率 {o6.get('fail_rate_pct')}%")
        md.append(f"- 起步 {o6.get('head_s')}s → 末尾 {o6.get('tail_s')}s，"
                  f"**变慢 {o6.get('slowdown_pct')}%**")
        md.append(f"- 峰值显存漂移 {o6.get('peak_drift_gib')} GiB（泄漏信号）")
    else:
        md.append("未采集到 soak 阶段数据。")
    md.append("")

    md.append("## O7 质量代理（按配置）\n")
    q = frontier["o7_quality"]
    if q:
        md.append(_table(["阶段", "配置", "分辨率", "帧", "CLIP", "时序误差", "运动", "闪烁"],
                         [[x["phase"], x["label"], x["res"], x["frames"], x["clip"],
                           x["temporal_err"], x["motion"], x["flicker"]] for x in q[:40]]))
    md.append("")

    md.append("## 结论\n")
    md.append(f"1. 720p 5s 一条约 {o3.get('seconds_median')}s，"
              f"一天（按 20 小时有效机时）可产出约 "
              f"{round(20 * (o3.get('per_hour') or 0))} 条。")
    md.append(f"2. 分辨率上限 {o1['max_feasible']}；再往上要么超时要么 OOM。")
    md.append(f"3. 显存峰值 {o4.get('peak_gib')} GiB，距离 128GB 还有 "
              f"{round(128 - (o4.get('peak_gib') or 0), 1)} GiB 余量——"
              f"余量在哪，就说明瓶颈不在显存。")
    md.append(f"4. 长跑 {'稳定' if (o6.get('slowdown_pct') or 0) < 5 else '存在降频'}，"
              f"衰减 {o6.get('slowdown_pct')}%。")

    (PUBLISH / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten -> {PUBLISH}")
    return frontier


if __name__ == "__main__":
    build()
