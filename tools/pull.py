"""Pull results from the running box to this machine, without disturbing the run.

Design rules, because the box is mid-benchmark:

  * never touch a file that is still being written (videos are filtered by mtime)
  * never pull more than a few MB, so a 1 Gbps prefetch keeps its bandwidth
  * tolerate a half-written last JSONL line (the reader already skips bad lines)
  * never modify anything remote: the whole remote command is a read-only `tar`

Usage (runs locally, e.g. on Windows):

    python tools/pull.py --host 1.2.3.4 --port 22 --user root --key ~/.ssh/id_ed25519
    python tools/pull.py --host ... --interval 600 --dest D:/cg-results
    python tools/pull.py --host ... --once --with-videos

Layout created locally:

    <dest>/latest/          always the newest snapshot, overwritten
    <dest>/history/*.tgz    timestamped snapshots (small, keep them all)
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

SMALL = [
    "results/results.jsonl",
    "results/env.json",
    "results/catalog.json",
    "results/day.log",
    "results/publish",
    "results/captures",      # podcast material: .cast and .png are small
]
# mp4 lives in its own, mtime-guarded layer: desktop recordings and clips are MBs,
# and pulling them every 10 minutes would compete with a running prefetch.
VIDEO_DIRS = ["results/videos", "results/captures", "results/publish/thumbs"]
VIDEO_GLOBS = ["*.mp4", "frame-*.png"]


def _ssh_base(args) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=15", "-p", str(args.port)]
    if args.key:
        cmd += ["-i", str(Path(args.key).expanduser())]
    cmd.append(f"{args.user}@{args.host}")
    return cmd


def _run_remote(args, remote_cmd: str, out_path: Path) -> bool:
    cmd = _ssh_base(args) + [remote_cmd]
    with out_path.open("wb") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode(errors="ignore").strip()[:300]
        print(f"  ssh failed ({p.returncode}): {err}")
        return False
    return out_path.stat().st_size > 0


def _cd(remote_dir: str) -> str:
    """`cd '~/x'` would not expand the tilde, so handle ~ explicitly."""
    if remote_dir.startswith("~"):
        rest = remote_dir[1:].lstrip("/")
        return f'cd "$HOME"/{shlex.quote(rest)}' if rest else 'cd "$HOME"'
    return f"cd {shlex.quote(remote_dir)}"


def _small_cmd(remote_dir: str) -> str:
    paths = " ".join(shlex.quote(p) for p in SMALL)
    # --ignore-failed-read: a phase that has not produced publish/ yet must not fail the pull
    # --exclude: keep the minute-level layer tiny; mp4 rides in the video layer
    # $(ls ...): one telemetry csv per run, ~20 KB each - the thermal curve lives here
    return (f"{_cd(remote_dir)} && "
            f"tar czf - --ignore-failed-read --exclude='*.mp4' {paths} "
            f"$(ls results/smi-*.csv 2>/dev/null) 2>/dev/null")


def _video_cmd(remote_dir: str, min_age_min: int) -> str:
    # only files untouched for N minutes: never grab an mp4 that is still being written
    dirs = " ".join(shlex.quote(d) for d in VIDEO_DIRS)
    name_expr = " -o ".join(f"-name '{g}'" for g in VIDEO_GLOBS)
    return (f"{_cd(remote_dir)} && "
            f"find {dirs} -type f \\( {name_expr} \\) -mmin +{min_age_min} -print0 "
            f"2>/dev/null | tar czf - --null --files-from=- 2>/dev/null")


def _extract(tgz: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            for member in tf:
                if member.name.startswith("/") or ".." in member.name.split("/"):
                    continue  # refuse absolute/parent paths
                tf.extract(member, dest)
                n += 1
    except (tarfile.TarError, EOFError):
        return n  # a truncated archive still yields whatever was complete
    return n


def _count_runs(dest: Path) -> int:
    f = dest / "latest" / "results" / "results.jsonl"
    if not f.exists():
        return 0
    with f.open(encoding="utf-8", errors="ignore") as fh:
        return sum(1 for line in fh if line.strip().endswith("}"))


def pull_once(args) -> int:
    dest = Path(args.dest).expanduser()
    history = dest / "history"
    history.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tgz = history / f"snap-{ts}.tgz"

    if not _run_remote(args, _small_cmd(args.remote_dir), tgz):
        return 0
    n = _extract(tgz, dest / "latest")

    size_kb = tgz.stat().st_size / 1024
    runs = _count_runs(dest)
    print(f"[{datetime.now():%H:%M:%S}] {n} files, {size_kb:.0f} KB, {runs} runs recorded")

    if args.with_videos:
        vtgz = history / f"videos-{ts}.tgz"
        if _run_remote(args, _video_cmd(args.remote_dir, args.video_age_min), vtgz):
            mb = vtgz.stat().st_size / 2**20
            m = _extract(vtgz, dest / "latest")
            print(f"           + {m} media files, {mb:.1f} MB")
            vtgz.unlink(missing_ok=True)   # extracted already; keeping it doubles local disk
        else:
            vtgz.unlink(missing_ok=True)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="root")
    ap.add_argument("--key", default=None, help="path to private key")
    ap.add_argument("--remote-dir", default="~/canguo-videobench")
    ap.add_argument("--dest", default="cg-results")
    ap.add_argument("--interval", type=int, default=600,
                    help="seconds between pulls; 0 = once")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--with-videos", action="store_true",
                    help="also pull finished mp4 / key frames (mtime-guarded, deleted after extract)")
    ap.add_argument("--video-age-min", type=int, default=3,
                    help="only media untouched for this many minutes")
    args = ap.parse_args()

    if args.once or args.interval <= 0:
        return pull_once(args)

    print(f"pulling every {args.interval}s -> {args.dest}  (ctrl-c to stop)")
    while True:
        try:
            pull_once(args)
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  pull error: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
