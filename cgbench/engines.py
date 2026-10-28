"""Model adapters. One job: turn a config into frames, and record what it cost.

Deliberately defensive. Diffusers pipelines differ in accepted kwargs across
versions and models, so we filter the call signature instead of assuming one.
Anything unsupported (fp8, a kwarg, a whole pipeline) is recorded as a finding,
never a crash: on a 24h run, an exception in one config must not end the day.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path

from .harness import VIDEOS, Sampler

DTYPES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


@dataclass
class GenResult:
    frames: list = field(default_factory=list)
    video_path: str | None = None
    seconds: float = 0.0
    load_seconds: float = 0.0
    peak_gib: float | None = None
    info: dict = field(default_factory=dict)


def save_video(frames, path: Path, fps: int) -> str | None:
    """Try diffusers' exporter, fall back to PNG frames, then to .npy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video

        export_to_video(frames, str(path.with_suffix(".mp4")), fps=fps)
        return str(path.with_suffix(".mp4"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import av

        p = path.with_suffix(".mp4")
        with av.open(str(p), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = frames[0].width
            stream.height = frames[0].height
            stream.pix_fmt = "yuv420p"
            for im in frames:
                for packet in stream.encode(av.VideoFrame.from_image(im)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return str(p)
    except Exception:  # noqa: BLE001
        pass
    try:
        d = path.with_suffix("")
        d.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(frames):
            im.save(d / f"{i:04d}.png")
        return str(d)
    except Exception:  # noqa: BLE001
        return None


class FakeEngine:
    """No weights, synthetic cost. Lets the whole 24h plan be validated offline.

    `speed` compresses the synthetic cost so a full-day plan can be exercised in
    a couple of minutes; it does not change any of the control flow.
    """

    name = "fake"
    base_load_s = 3.0

    def __init__(self, cfg):
        self.cfg = cfg
        self.speed = float(cfg.get("speed", 1.0))

    def load(self) -> float:
        t0 = time.perf_counter()
        time.sleep(self.base_load_s * self.speed)
        return time.perf_counter() - t0

    def generate(self, *, prompt, negative, width, height, frames, steps,
                 guidance, seed, out_path) -> GenResult:
        import numpy as np
        from PIL import Image

        cost = (width * height * frames * steps) / 4.0e8  # ~0.4 GP/s effective
        t0 = time.perf_counter()
        time.sleep(min(cost * self.speed, 60.0))
        rng = np.random.default_rng(seed)
        base = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
        imgs = [
            Image.fromarray(base).resize((width, height)).point(
                lambda v, o=i * 3: (v + o) % 255)
            for i in range(frames)
        ]
        secs = time.perf_counter() - t0
        p = save_video(imgs, out_path, self.cfg.get("fps", 24))
        return GenResult(frames=imgs, video_path=p, seconds=secs,
                         load_seconds=self.base_load_s,
                         peak_gib=round(width * height * frames * 4 / 2**30, 2),
                         info={"engine": "fake", "cost_model": "0.4 GP/s"})


class DiffusersVideoEngine:
    """Generic diffusers text-to-video adapter."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg["model_key"]
        self.pipe = None
        self.notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> float:
        import torch
        from diffusers import DiffusionPipeline

        t0 = time.perf_counter()
        dtype_name = self.cfg.get("dtype", "bfloat16")
        dtype = getattr(torch, DTYPES.get(dtype_name, "bfloat16"))

        kwargs = {"torch_dtype": dtype}
        if self.cfg.get("cpu_offload"):
            kwargs["low_cpu_mem_usage"] = True

        pipe = DiffusionPipeline.from_pretrained(self.cfg["model_id"], **kwargs)

        if self.cfg.get("cpu_offload"):
            pipe.enable_model_cpu_offload()
            self.notes.append("cpu_offload")
        else:
            pipe.to("cuda")

        for opt in ("enable_vae_tiling", "enable_attention_slicing"):
            if hasattr(pipe, opt):
                try:
                    getattr(pipe, opt)()
                    self.notes.append(opt.replace("enable_", ""))
                except Exception:  # noqa: BLE001
                    pass

        if dtype_name == "fp8":
            self.notes.append(self._try_fp8(pipe))

        self.pipe = pipe
        self.load_seconds = time.perf_counter() - t0
        return self.load_seconds

    def _try_fp8(self, pipe) -> str:
        """Weight-only fp8 is an ecosystem question, so record the answer instead of assuming."""
        try:
            import torchao  # noqa: F401
        except Exception:  # noqa: BLE001
            return "fp8: torchao missing -> fell back to bf16"
        try:
            from torchao.quantization import quantize_, float8_weight_only

            quantize_(pipe.transformer, float8_weight_only())
            return "fp8: applied via torchao float8_weight_only"
        except Exception as exc:  # noqa: BLE001
            return f"fp8: unsupported ({type(exc).__name__}: {str(exc)[:80]})"

    # -- inference ---------------------------------------------------------
    def generate(self, *, prompt, negative, width, height, frames, steps,
                 guidance, seed, out_path) -> GenResult:
        import torch

        pipe = self.pipe
        gen = torch.Generator(device="cuda").manual_seed(seed)

        # Wan/Hunyuan expect 4n+1 frames; asking for 4n+1 is always safe.
        n = 4 * round((frames - 1) / 4) + 1
        n = max(5, n)

        wanted = {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "num_frames": n,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": gen,
            "output_type": "pil",
        }
        accepted = set(inspect.signature(pipe.__call__).parameters)
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        dropped = sorted(set(wanted) - set(kwargs))

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with Sampler(f"gen-{out_path.stem}") as smp:
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = pipe(**kwargs)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t0

        peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        pil = out.frames[0]
        path = save_video(pil, out_path, self.cfg.get("fps", 24))

        return GenResult(
            frames=pil,
            video_path=path,
            seconds=secs,
            load_seconds=getattr(self, "load_seconds", 0.0),
            peak_gib=peak,
            info={
                "engine": "diffusers",
                "requested_frames": frames,
                "actual_frames": n,
                "dropped_kwargs": dropped,
                "notes": self.notes,
                "smi": smp.summary(),
            },
        )


def build_engine(cfg: dict):
    if cfg.get("engine") == "fake":
        return FakeEngine(cfg)
    return DiffusersVideoEngine(cfg)
