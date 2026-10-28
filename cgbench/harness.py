"""Core plumbing: env capture, JSONL store with resume, timing, telemetry sampling."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CGBENCH_OUT", ROOT / "results"))
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "results.jsonl"
VIDEOS = OUT / "videos"
VIDEOS.mkdir(exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sh(cmd: str, timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


# ---------------------------------------------------------------- result store


def _load_done() -> set[str]:
    done: set[str] = set()
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001
                continue
    return done


DONE = _load_done()


def emit(key: str, payload: dict) -> None:
    rec = {"key": key, "ts": now(), **payload}
    with RUNS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()          # a reader on the other end of an ssh pipe must see it now
    DONE.add(key)


def skip(key: str, resume: bool = True) -> bool:
    return resume and key in DONE


def rows() -> list[dict]:
    if not RUNS.exists():
        return []
    out = []
    for line in RUNS.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------- environment


def env_info() -> dict:
    info = {
        "host": platform.node(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "kernel": platform.release(),
        "started": now(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(
                device=p.name,
                cc=f"{p.major}.{p.minor}",
                vram_gib=round(p.total_memory / 2**30, 1),
                sm_count=p.multi_processor_count,
            )
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)[:200]

    for mod in ("diffusers", "transformers", "accelerate", "av", "cv2"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None

    info["driver"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    info["ffmpeg"] = sh("ffmpeg -version 2>&1 | head -1")
    info["disk_free_gib"] = round(shutil.disk_usage(OUT).free / 2**30, 1)
    return info


# ---------------------------------------------------------------- telemetry


SMI_FIELDS = "timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu"


class Sampler:
    """Background nvidia-smi writer. No-ops when nvidia-smi is missing."""

    def __init__(self, tag: str, interval_ms: int = 1000):
        self.path = OUT / f"smi-{tag}.csv"
        self.interval_ms = interval_ms
        self.proc = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._fh = self.path.open("w", encoding="utf-8")
            try:
                self.proc = subprocess.Popen(
                    ["nvidia-smi", f"--query-gpu={SMI_FIELDS}",
                     "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                    stdout=self._fh, stderr=subprocess.DEVNULL,
                )
            except Exception:  # noqa: BLE001
                self.proc = None
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        if getattr(self, "_fh", None):
            self._fh.close()
        return False

    def summary(self) -> dict:
        if not self.path.exists():
            return {}
        vals: list[dict] = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                vals.append({"util": float(parts[1]), "mem": float(parts[2]),
                             "clk": float(parts[3]), "pwr": float(parts[4]),
                             "temp": float(parts[5])})
            except ValueError:
                continue
        if not vals:
            return {}

        def pct(k, q):
            v = sorted(x[k] for x in vals)
            return round(v[min(len(v) - 1, int(len(v) * q))], 1)

        return {
            "samples": len(vals),
            "util_p50": pct("util", 0.5),
            "mem_peak_mib": round(max(x["mem"] for x in vals), 0),
            "clk_p50": pct("clk", 0.5),
            "pwr_p50": pct("pwr", 0.5),
            "pwr_peak": round(max(x["pwr"] for x in vals), 1),
            "temp_peak": round(max(x["temp"] for x in vals), 1),
        }


# ---------------------------------------------------------------- timing


class Budget:
    """Soft deadline for one phase. Stops launching new runs, never kills a run."""

    def __init__(self, seconds: float, name: str = ""):
        self.name = name
        self.deadline = time.time() + seconds
        self.total = seconds

    def left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def spent(self) -> float:
        return self.total - self.left()

    def expired(self) -> bool:
        return self.left() <= 0


def timed(fn):
    t0 = time.perf_counter()
    val = fn()
    return val, time.perf_counter() - t0
