"""Publish results: push the publishable subset to a git repo and/or a webhook.

Only the small stuff travels: report.md, metrics.csv, frontier.json, results.jsonl
and a handful of thumbnails. Videos stay on the box - pushing gigabytes from a
metered instance is the single easiest way to waste a day.

Secrets never land in .git/config:
  CGBENCH_RESULTS_REMOTE   https://github.com/you/canguo-results.git
  CGBENCH_GIT_TOKEN        fine-grained PAT, repo scope on that one repo
  CGBENCH_WEBHOOK          feishu/wecom/slack incoming webhook URL
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLISH = Path(os.environ.get("CGBENCH_OUT", ROOT / "results")) / "publish"
THUMBS = 6


def summary() -> str:
    f = PUBLISH / "frontier.json"
    if not f.exists():
        return "cgbench: no results yet"
    d = json.loads(f.read_text(encoding="utf-8"))
    env = d.get("env", {})
    o1, o2, o3 = d.get("o1_resolution", {}), d.get("o2_duration", {}), d.get("o3_throughput", {})
    o4, o6 = d.get("o4_memory", {}), d.get("o6_stability", {})
    return (
        f"视频生成能力评估完成 — {env.get('device', '?')}\n"
        f"分辨率上限: {o1.get('max_feasible')} (≤{o1.get('budget_s')}s/条)\n"
        f"时长上限: {o2.get('max_seconds')}s @720p\n"
        f"吞吐: {o3.get('per_hour')} 条/小时 (中位 {o3.get('seconds_median')}s)\n"
        f"显存峰值: {o4.get('peak_gib')} GiB / 128 GiB\n"
        f"24h 衰减: {o6.get('slowdown_pct')}%  失败率: {o6.get('fail_rate_pct')}%"
    )


def _git(*args, env=None, cwd=PUBLISH):
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                       text=True, env=env)
    if p.returncode != 0:
        print(f"git {' '.join(args)} -> {p.returncode}\n{p.stderr[:400]}")
    return p.returncode == 0


def push_git() -> bool:
    remote = os.environ.get("CGBENCH_RESULTS_REMOTE")
    if not remote:
        print("CGBENCH_RESULTS_REMOTE not set; skipping git push")
        return False

    src_results = PUBLISH.parent
    for name in ("results.jsonl", "frontier.json", "report.md", "metrics.csv"):
        s = src_results / name if (src_results / name).exists() else PUBLISH / name
        if s.exists() and s.parent != PUBLISH:
            shutil.copy2(s, PUBLISH / name)

    vids = sorted((PUBLISH.parent / "videos").glob("*.mp4"))[:THUMBS]
    if vids:
        (PUBLISH / "thumbs").mkdir(exist_ok=True)
        for v in vids:
            shutil.copy2(v, PUBLISH / "thumbs" / v.name)

    if not (PUBLISH / ".git").exists():
        _git("init", "-b", "main")
        _git("remote", "add", "origin", remote)

    _git("add", "-A")
    _git("-c", "user.name=cgbench", "-c", "user.email=cgbench@local",
         "commit", "-m", f"results: {summary().splitlines()[0]}")

    env = os.environ.copy()
    token = os.environ.get("CGBENCH_GIT_TOKEN")
    args = ["push", "origin", "HEAD:main"]
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        args = ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
                *args]
    ok = _git(*args, env=env)
    print("git push:", "ok" if ok else "failed")
    return ok


def push_webhook() -> bool:
    url = os.environ.get("CGBENCH_WEBHOOK")
    if not url:
        print("CGBENCH_WEBHOOK not set; skipping webhook")
        return False
    text = summary()
    fmt = os.environ.get("CGBENCH_WEBHOOK_FORMAT", "feishu")
    if fmt == "feishu":
        body = {"msg_type": "text", "content": {"text": text}}
    elif fmt == "wecom":
        body = {"msgtype": "text", "text": {"content": text}}
    elif fmt == "slack":
        body = {"text": text}
    else:
        body = {"text": text}

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print("webhook:", resp.status, resp.read(200).decode(errors="ignore"))
        return True
    except Exception as exc:  # noqa: BLE001
        print("webhook failed:", exc)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--webhook", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.summary or not (args.git or args.webhook):
        print(summary())
        return 0
    ok = True
    if args.git:
        ok &= push_git()
    if args.webhook:
        ok &= push_webhook()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
