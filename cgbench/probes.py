"""Phase runners. Each phase is a bounded experiment that can be resumed.

An engine is loaded once and reused for every config that needs it: on a 24h run,
reloading a 14B checkpoint per config would quietly eat hours. Switching model
drops the previous one first, because 128GB is generous but not infinite.
"""
from __future__ import annotations

import gc
import time
import traceback

from . import metrics
from .engines import build_engine
from .harness import VIDEOS, Budget, Sampler, emit, env_info, model_revision, skip, write_lockfile
from .spec import MODELS, NEGATIVE, PHASES, PROMPTS, RESOLUTIONS

# Dynamic registry: spec.MODELS plus whatever the HF screening shortlisted.
_MODELS: dict = {}


def _registry() -> dict:
    if _MODELS:
        return _MODELS
    from .catalog import load_shortlist

    _MODELS.update(MODELS)
    for e in load_shortlist():
        _MODELS[e["key"]] = {
            "id": e["id"],
            "label": e.get("label", e["key"]),
            "fps": e.get("fps", 24),
            "steps": e.get("steps", 30),
            "guidance": e.get("guidance", 5.0),
            "multiple": e.get("multiple", 16),
            "note": "from catalog",
        }
    return _MODELS


class EnginePool:
    def __init__(self, dry_run: bool = False, cpu_offload: bool = False):
        self.dry_run = dry_run
        self.cpu_offload = cpu_offload
        self.key = None
        self.engine = None
        self.load_s = 0.0
        self.loads = 0

    def get(self, mcfg: dict):
        key = (mcfg["model_key"], mcfg.get("dtype"), mcfg.get("quant"), self.cpu_offload)
        if key == self.key and self.engine is not None:
            return self.engine, 0.0
        self.release()
        cfg = dict(mcfg)
        if self.dry_run:
            cfg["engine"] = "fake"
            cfg["speed"] = 0.02
        cfg["cpu_offload"] = self.cpu_offload
        eng = build_engine(cfg)
        t0 = time.perf_counter()
        eng.load()
        dt = time.perf_counter() - t0
        self.key, self.engine, self.load_s = key, eng, dt
        self.loads += 1
        print(f"  [load] {mcfg['model_key']} {mcfg.get('dtype')} in {dt:.1f}s", flush=True)
        return eng, dt

    def release(self):
        if self.engine is not None:
            del self.engine
            self.engine = None
        self.key = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _model_cfg(model_key: str, dtype: str, quant: str | None = None) -> dict:
    reg = _registry()
    if model_key not in reg:
        raise KeyError(f"unknown model: {model_key} (run `python -m cgbench catalog` first)")
    m = reg[model_key]
    return {
        "model_key": model_key,
        "model_id": m["id"],
        "label": m["label"],
        "fps": m["fps"],
        "dtype": dtype,
        "quant": quant,
        "multiple": m["multiple"],
    }


def run_one(pool: EnginePool, cfg: dict, phase: str, idx: int, resume: bool = True) -> dict | None:
    model_key = cfg["model_key"]
    dtype = cfg.get("dtype", "bfloat16")
    label = cfg["label"]
    key = f"{phase}:{model_key}:{label}:{dtype}"
    if skip(key, resume):
        return None

    mcfg = _model_cfg(model_key, dtype, cfg.get("quant"))
    w, h = RESOLUTIONS[cfg["res"]]
    mult = mcfg["multiple"]
    w, h = (w // mult) * mult, (h // mult) * mult

    fps = mcfg["fps"]
    frames = max(5, int(round(fps * cfg["seconds"])))
    frames = 4 * round((frames - 1) / 4) + 1

    prompt = PROMPTS[idx % len(PROMPTS)]
    reg = _registry()[model_key]
    steps = int(cfg.get("steps", reg["steps"]))
    guidance = float(cfg.get("guidance", reg["guidance"]))
    seed = int(cfg.get("seed", 1234))
    out_path = VIDEOS / f"{phase}-{model_key}-{label}".replace("/", "_")

    rec = {
        "phase": phase, "model": model_key, "model_id": mcfg["model_id"],
        "model_revision": model_revision(mcfg["model_id"]),
        "label": label, "width": w, "height": h, "frames": frames, "fps": fps,
        "steps": steps, "guidance": guidance, "dtype": dtype, "seed": seed,
        "prompt": prompt, "dry_run": pool.dry_run,
    }

    try:
        engine, load_s = pool.get(mcfg)
        with Sampler(f"{phase}-{model_key}-{label}"):
            res = engine.generate(
                prompt=prompt, negative=NEGATIVE, width=w, height=h,
                frames=frames, steps=steps, guidance=guidance, seed=seed,
                out_path=out_path,
            )
        m = metrics.summarize(res.frames, prompt)
        mpf = (w * h * frames) / 1e6
        rec.update(
            status="ok",
            load_s=round(load_s, 1),
            cold=False if load_s == 0 else True,
            seconds=round(res.seconds, 2),
            s_per_it=round(res.seconds / max(1, steps), 3),
            s_per_mpframe=round(res.seconds / max(1e-6, mpf), 3),
            peak_gib=res.peak_gib,
            video=res.video_path,
            metrics=m,
            engine_info=res.info,
        )
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}",
                   traceback=traceback.format_exc()[-1200:])

    emit(key, rec)
    if rec["status"] == "ok":
        print(f"  [{phase}] {model_key:<9} {label:<6} {w}x{h} {frames}f "
              f"{rec['seconds']}s peak={rec['peak_gib']}GiB", flush=True)
    else:
        print(f"  [{phase}] {model_key:<9} {label:<6} FAILED: {rec['error'][:90]}", flush=True)
    return rec


def run_phase(name: str, dry_run: bool = False, resume: bool = True,
              cpu_offload: bool = False, budget_min: float | None = None,
              from_catalog: bool = False) -> dict:
    spec = PHASES[name]
    minutes = budget_min if budget_min is not None else spec["minutes"]
    if dry_run and budget_min is None:
        # compress the clock, not the logic: every phase still runs end to end
        minutes = max(0.2, min(minutes * 0.01, 0.8))
    budget = Budget(minutes * 60, name)

    model_list = spec.get("models") or [spec["model"]]
    if from_catalog and name == "models":
        keys = [k for k in _registry() if k not in MODELS]
        if keys:
            model_list = keys
            print(f"[catalog] testing {len(keys)} shortlisted models")
    pool = EnginePool(dry_run=dry_run, cpu_offload=cpu_offload)
    started = time.time()
    n_ok = n_err = 0

    print(f"\n=== phase {name} ({minutes:.0f} min) - {spec['desc']}", flush=True)

    idx = 0
    first_pass = True
    while first_pass or spec.get("loop"):
        first_pass = False
        for model_key in model_list:
            for cfg in spec["configs"]:
                if budget.expired():
                    break
                rec = run_one(pool, {**cfg, "model_key": model_key}, name, idx, resume)
                idx += 1
                if rec is None:
                    continue
                n_ok += rec["status"] == "ok"
                n_err += rec["status"] != "ok"
            if budget.expired():
                break
        if budget.expired():
            break

    pool.release()
    summary = {"phase": name, "budget_min": minutes,
               "used_min": round((time.time() - started) / 60, 1),
               "runs_ok": n_ok, "runs_err": n_err, "loads": pool.loads}
    emit(f"phase:{name}", summary)
    print(f"=== {name}: {n_ok} ok / {n_err} err, {summary['used_min']} min, "
          f"{pool.loads} model load(s)\n", flush=True)
    return summary


def run_all(dry_run: bool = False, resume: bool = True, cpu_offload: bool = False,
            phases: list[str] | None = None, from_catalog: bool = False) -> None:
    from .harness import DONE

    if "env:start" not in DONE:
        emit("env:start", {"env": env_info()})
        lock = write_lockfile()
        if lock:
            print(f"[env] dependency lock -> {lock}")

    for name in (phases or list(PHASES)):
        if name not in PHASES:
            print(f"unknown phase: {name}")
            continue
        run_phase(name, dry_run=dry_run, resume=resume, cpu_offload=cpu_offload,
                  from_catalog=from_catalog)
