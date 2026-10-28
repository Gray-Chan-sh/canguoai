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


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


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


def _git_info() -> dict:
    """Which code produced these numbers. A tarball without .git returns blanks.

    Uses the list form on purpose: ROOT contains spaces on a typical Windows path,
    and `git -C <path>` through a shell silently fails there.
    """
    def run(*args):
        try:
            p = subprocess.run(["git", "-C", str(ROOT), *args],
                               capture_output=True, text=True, timeout=10)
            return p.returncode, p.stdout.strip()
        except Exception:  # noqa: BLE001
            return 1, ""

    rc, commit = run("rev-parse", "--short", "HEAD")
    if rc != 0 or not commit:
        return {"commit": None, "dirty": None}
    rc2, dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if rc2 == 0 else None}


def write_manifest(extra: dict | None = None) -> Path:
    """A hash manifest is what makes "complete" checkable rather than claimed.

    Large artefacts (mp4) are listed with size but not hashed: hashing hundreds of
    megabytes on a metered box costs more than it proves. Everything small is hashed.
    """
    import hashlib

    files = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "manifest.json":
            continue
        rel = str(p.relative_to(OUT))
        size = p.stat().st_size
        if p.suffix == ".mp4" or size > 32 * 2**20:
            files.append({"path": rel, "size": size, "sha256": None,
                          "unhashed": "large or binary media"})
            continue
        files.append({"path": rel, "size": size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})

    doc = {
        "generated": now(),
        "cgbench_version": _version(),
        "git": _git_info(),
        "revisions": _REVS,
        "count": len(files),
        "bytes": sum(f["size"] for f in files),
        "files": files,
        **(extra or {}),
    }
    path = OUT / "manifest.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_lockfile() -> Path | None:
    """A single diffusers version string does not pin transitive deps."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return None
    if not out.strip():
        return None
    p = OUT / "requirements.lock"
    p.write_text(out, encoding="utf-8")
    return p


_REVS: dict[str, str | None] = {}


def _cache_revision(repo_id: str) -> str | None:
    """Offline fallback: the local HF cache already knows which snapshot is on disk."""
    base = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    d = base / ("models--" + repo_id.replace("/", "--"))
    ref = d / "refs" / "main"
    if ref.exists():
        return ref.read_text(encoding="utf-8").strip()
    snaps = d / "snapshots"
    if snaps.exists():
        found = sorted(p.name for p in snaps.iterdir() if p.is_dir())
        if found:
            return found[-1]
    return None


def model_revision(repo_id: str) -> str | None:
    """Pin the weights: HF repos are updated in place under the same id.

    Online lookup first, then the prefetched revisions file, then the local cache.
    Three sources because the box may be offline, rate limited, or air-gapped.
    """
    if repo_id in _REVS:
        return _REVS[repo_id]

    sha = None
    revfile = OUT / "revisions.json"
    if revfile.exists():
        try:
            sha = json.loads(revfile.read_text(encoding="utf-8")).get(repo_id)
        except Exception:  # noqa: BLE001
            sha = None
    if not sha:
        try:
            from huggingface_hub import model_info

            sha = model_info(repo_id, timeout=20).sha
        except Exception:  # noqa: BLE001
            sha = None
    if not sha:
        sha = _cache_revision(repo_id)

    _REVS[repo_id] = sha
    return sha


def _cgroup_limit_bytes() -> int | None:
    """Container memory cap. On GB10 this is not 'just RAM': it caps the unified pool,
    so a 64 GiB cap makes the 128 GiB card behave like a 64 GiB card, with an OOM that
    looks like a VRAM error."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            continue
        if v == "max":
            return None
        if v.isdigit():
            n = int(v)
            # cgroup v1 uses a huge sentinel for "unlimited"
            return None if n > 2**60 else n
    return None


def _meminfo() -> dict:
    try:
        out = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.strip().split()[0]) * 1024
        return out
    except Exception:  # noqa: BLE001
        return {}


def memory_facts(vram_gib: float | None = None) -> dict:
    """How much of the box this process can actually reach, versus what the GPU claims."""
    gib = lambda b: round(b / 2**30, 1) if b else None  # noqa: E731
    mem = _meminfo()
    total, avail = mem.get("MemTotal"), mem.get("MemAvailable")
    cap = _cgroup_limit_bytes()
    container = Path("/.dockerenv").exists() or "docker" in sh("head -c 4000 /proc/1/cgroup 2>/dev/null")
    unified = bool(vram_gib and total and abs(vram_gib - total / 2**30) / (total / 2**30) < 0.15)
    # "Effective pool" is only meaningful for unified memory; on a discrete card the
    # cgroup cap limits host RAM, not VRAM.
    candidates = [x for x in (total, cap) if x] + ([vram_gib * 2**30] if unified and vram_gib else [])
    return {
        "container": bool(container),
        "mem_total_gib": gib(total),
        "mem_available_gib": gib(avail),
        "cgroup_cap_gib": gib(cap),
        "vram_gib": vram_gib,
        "effective_pool_gib": gib(min(candidates)) if unified and candidates else None,
        "unified": unified,
    }


def env_info() -> dict:
    info = {
        "host": platform.node(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "kernel": platform.release(),
        "started": now(),
        "cgbench_version": _version(),
        "git": _git_info(),
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
    info["memory"] = memory_facts(info.get("vram_gib"))
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
