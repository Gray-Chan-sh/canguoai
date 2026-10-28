"""Capture material for the podcast: terminal casts, screenshots, screen recordings.

A rented box is headless, so "screen recording" means three different things:

  1. terminal cast  - always works, record any command into an asciinema v2 file
  2. GUI screenshot - only when an X display exists (DGX OS desktop session)
  3. desktop video  - ffmpeg x11grab, same precondition

Anything unavailable is reported as unavailable rather than faked. A missing
screenshot is a fact about the setup; a fabricated one is a lie in the edit.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .harness import OUT, VIDEOS

CAP = OUT / "captures"
CAP.mkdir(parents=True, exist_ok=True)
COLS, ROWS = 120, 36


# --------------------------------------------------------------- capabilities


def detect() -> dict:
    return {
        "display": os.environ.get("DISPLAY"),
        "pty": "pty" in sys.builtin_module_names or _has("pty"),
        "script": shutil.which("script"),
        "ffmpeg": shutil.which("ffmpeg"),
        "screenshot": next((shutil.which(t) for t in
                            ("gnome-screenshot", "scrot", "import", "grim")
                            if shutil.which(t)), None),
        "x11grab": bool(os.environ.get("DISPLAY")) and bool(shutil.which("ffmpeg")),
    }


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------- terminal cast


class TerminalRecorder:
    """Write an asciinema v2 .cast while a command runs in a pty."""

    def __init__(self, name: str, cols: int = COLS, rows: int = ROWS):
        self.path = CAP / f"{name}.cast"
        self.cols, self.rows = cols, rows

    def run(self, argv: list[str]) -> int:
        import pty
        import select

        fh = self.path.open("w", encoding="utf-8")
        fh.write(json.dumps({
            "version": 2, "width": self.cols, "height": self.rows,
            "timestamp": int(time.time()),
            "env": {"SHELL": os.environ.get("SHELL", "/bin/bash"),
                    "TERM": os.environ.get("TERM", "xterm-256color")},
        }) + "\n")
        fh.flush()

        pid, fd = pty.fork()
        if pid == 0:  # child
            os.execvp(argv[0], argv)

        start = time.time()
        try:
            while True:
                try:
                    r, _, _ = select.select([fd], [], [], 0.2)
                except OSError:
                    break
                if fd in r:
                    try:
                        data = os.read(fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    txt = data.decode("utf-8", errors="replace")
                    fh.write(json.dumps([round(time.time() - start, 4), "o", txt]) + "\n")
                    fh.flush()
                    sys.stdout.write(txt)
                    sys.stdout.flush()
        finally:
            fh.close()
            try:
                _, status = os.waitpid(pid, 0)
            except Exception:  # noqa: BLE001
                status = 0
        return status


def record(name: str, argv: list[str]) -> int:
    if not _has("pty"):
        print(f"[capture] pty unavailable; running without recording: {' '.join(argv)}")
        return subprocess.call(argv)
    rec = TerminalRecorder(name)
    print(f"[capture] recording -> {rec.path}")
    rc = rec.run(argv)
    print(f"\n[capture] cast saved: {rec.path}")
    return rc


# --------------------------------------------------------------- visuals


def screenshot(name: str) -> str | None:
    caps = detect()
    if not caps["display"]:
        print("[capture] no DISPLAY; screenshot skipped (headless session)")
        return None
    out = CAP / f"{name}.png"
    if caps["screenshot"] == "gnome-screenshot":
        subprocess.run(["gnome-screenshot", "-f", str(out)], check=False)
    elif caps["screenshot"] == "scrot":
        subprocess.run(["scrot", str(out)], check=False)
    elif caps["screenshot"] == "grim":
        subprocess.run(["grim", str(out)], check=False)
    elif caps["screenshot"] == "import":
        subprocess.run(["import", "-window", "root", str(out)], check=False)
    else:
        print("[capture] no screenshot tool found")
        return None
    return str(out) if out.exists() else None


def screen_record(seconds: int, name: str = "desktop") -> str | None:
    caps = detect()
    if not caps["x11grab"]:
        print("[capture] no DISPLAY or no ffmpeg; screen recording unavailable")
        return None
    out = CAP / f"{name}.mp4"
    cmd = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", "15", "-i", os.environ["DISPLAY"],
           "-t", str(seconds), "-pix_fmt", "yuv420p", str(out)]
    print(f"[capture] recording desktop {seconds}s -> {out}")
    subprocess.run(cmd, check=False, capture_output=True)
    return str(out) if out.exists() else None


def contact_sheet(name: str, frames: int = 6) -> str | None:
    """Grid of evenly spaced frames from a generated clip: the best still for a thumbnail."""
    try:
        import av
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    clips = sorted(VIDEOS.glob(f"*{name}*.mp4"))
    if not clips:
        return None
    src = clips[0]
    try:
        container = av.open(str(src))
        stream = container.streams.video[0]
        total = stream.frames or 0
        picks = []
        idx = 0
        want = {int(i * max(0, total - 1) / max(1, frames - 1)) for i in range(frames)} if total else set()
        for frame in container.decode(stream):
            if not total or idx in want:
                picks.append(frame.to_image())
            idx += 1
            if len(picks) >= frames:
                break
        container.close()
        if not picks:
            return None
        w, h = picks[0].size
        tw, th = w // 2, h // 2
        sheet = Image.new("RGB", (tw * min(3, len(picks)), th * (1 + (len(picks) - 1) // 3)))
        for i, im in enumerate(picks[:6]):
            sheet.paste(im.resize((tw, th)), ((i % 3) * tw, (i // 3) * th))
        out = CAP / f"{name}-sheet.png"
        sheet.save(out)
        return str(out)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------- storyboard


def shotlist() -> str:
    caps = detect()
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 播客素材清单（拍摄脚本）",
        "",
        f"生成于 {ts}",
        f"- DISPLAY: `{caps['display'] or '无（纯 SSH）'}`",
        f"- 终端录制: {'可用' if caps['pty'] else '不可用'}   "
        f"截图: {caps['screenshot'] or '不可用'}   "
        f"桌面录屏: {'可用' if caps['x11grab'] else '不可用'}",
        "",
        "每个镜头都给了**命令**、**建议时长**、**看点**和**解说提示**。",
        "终端镜头用 `python -m cgbench.capture run <名字> -- <命令>` 录制，",
        "产物在 `results/captures/*.cast`，可直接用 asciinema player 播放或转 GIF。",
        "",
    ]
    shots = [
        ("01-cold-open", "nvidia-smi", 20,
         "128GB 统一内存第一次出现在屏幕上",
         "开场就放这个数字。观众对'128G'没有直觉，对'一台能放在桌上的机器'有。"),
        ("02-catalog", "python -m cgbench catalog --top 8", 60,
         "几十个候选模型被筛成一页，被拒的写着原因",
         "这是全片最有信息量的一段：不是'能不能跑'，是'生态允许你跑什么'。"),
        ("03-smoke", "python -m cgbench phase smoke", 120,
         "第一条视频生成出来的瞬间",
         "强调这不是调 API，是本地 DiT 在跑。"),
        ("04-resolution", "python -m cgbench phase resolution", 180,
         "480p→720p→1080p，耗时跳变",
         "分辨率每翻一倍，时间不是翻倍，是翻更多——VAE 和 attention 都不线性。"),
        ("05-duration", "python -m cgbench phase duration", 180,
         "20 秒那档超时或 OOM 的那一刻",
         "这是'上限'第一次具象化。失败比成功更适合做叙事高潮。"),
        ("06-precision", "python -m cgbench phase precision", 120,
         "fp8 unsupported 的报错原文",
         "不要剪掉失败。arm64 + sm_121 的生态缺口，是这台机器最真实的部分。"),
        ("07-report", "python -m cgbench report", 60,
         "七个问题逐条被回答",
         "把 O1–O7 当成节目结构，一条一条过。"),
        ("08-outro", "cat results/publish/report.md", 30,
         "最终的边界表",
         "收尾给一个可比较的数字，别给形容词。"),
    ]
    for name, cmd, secs, look, tip in shots:
        lines += [
            f"## {name}",
            f"- 命令：`{cmd}`",
            f"- 建议时长：{secs}s",
            f"- 看点：{look}",
            f"- 解说：{tip}",
            "",
        ]
    lines += [
        "## 素材落点",
        "- 终端：`results/captures/<名字>.cast` → asciinema player / `agg` 转 GIF",
        "- 桌面：`results/captures/desktop.mp4`（有 DISPLAY 时）",
        "- 静帧：`results/captures/*.png`、`results/captures/*-sheet.png`（成片关键帧九宫格，可直接做封面）",
        "- 数据：`results/publish/report.md`、`metrics.csv`",
        "",
        "## 没有 GUI 怎么办",
        "纯 SSH 下截图和桌面录屏不可用，这是事实而不是缺陷。替代方案：",
        "1. 终端 cast 已经是完整叙事，配旁白足够；",
        "2. 用 `--sheet` 生成成片关键帧九宫格，做封面和插图；",
        "3. 需要真桌面时，在灿果控制台开 VNC/远程桌面再跑 `screen_record()`。",
    ]
    return "\n".join(lines)


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="cgbench.capture")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="record a command into an asciinema cast")
    p.add_argument("name")
    p.add_argument("argv", nargs=argparse.REMAINDER)

    p = sub.add_parser("shot", help="screenshot the X display")
    p.add_argument("name", nargs="?", default="shot")

    p = sub.add_parser("record", help="record the desktop for N seconds")
    p.add_argument("seconds", type=int, nargs="?", default=30)
    p.add_argument("--name", default="desktop")

    p = sub.add_parser("sheet", help="contact sheet from a generated clip")
    p.add_argument("name")
    p.add_argument("--frames", type=int, default=6)

    p = sub.add_parser("detect", help="what capture is possible here")
    p = sub.add_parser("shots", help="print the storyboard")

    args = ap.parse_args()
    if args.cmd == "run":
        argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
        return record(args.name, argv)
    if args.cmd == "shot":
        print(screenshot(args.name) or "no screenshot")
        return 0
    if args.cmd == "record":
        print(screen_record(args.seconds, args.name) or "no recording")
        return 0
    if args.cmd == "sheet":
        print(contact_sheet(args.name, args.frames) or "no sheet")
        return 0
    if args.cmd == "detect":
        print(json.dumps(detect(), indent=2))
        return 0
    if args.cmd == "shots":
        print(shotlist())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
