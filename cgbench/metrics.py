"""Objective proxies for "is this video any good". No human in the loop.

These are not quality scores in the artistic sense. They are cheap, repeatable
signals that catch the failure modes that actually show up at the edges of a
hardware envelope: frozen video, drift, flicker, mush, and prompt mismatch.

All metrics run on downscaled frames so they cost seconds, not minutes.
"""
from __future__ import annotations

import math

import numpy as np

_CACHE: dict = {}
PROBE_SIZE = 256


def _np(frames, size: int = PROBE_SIZE) -> list[np.ndarray]:
    out = []
    for im in frames:
        f = im
        if hasattr(f, "resize"):
            f = f.resize((size, size))
            out.append(np.asarray(f.convert("RGB"), dtype=np.float32))
        else:
            out.append(np.asarray(f, dtype=np.float32))
    return out


def _gray(f: np.ndarray) -> np.ndarray:
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])


def clip_score(frames, prompt: str, model_id: str = "openai/clip-vit-base-patch32") -> float | None:
    """Prompt/video alignment. Mean over 4 evenly spaced frames."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        key = f"clip:{model_id}"
        if key not in _CACHE:
            _CACHE[key] = (
                CLIPModel.from_pretrained(model_id).eval(),
                CLIPProcessor.from_pretrained(model_id),
            )
        model, proc = _CACHE[key]
        idx = np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)
        picks = [frames[i] for i in idx]
        inputs = proc(text=[prompt], images=picks, return_tensors="pt", padding=True)
        with torch.inference_mode():
            out = model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return round(float((img @ txt.T).mean()) * 100, 2)
    except Exception:  # noqa: BLE001
        return None


def temporal_consistency(arr: list[np.ndarray]) -> float | None:
    """Warp error between consecutive frames. Lower is steadier; 0 would be perfect."""
    try:
        import cv2

        errs = []
        prev = _gray(arr[0])
        for cur in arr[1:]:
            g = _gray(cur)
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = prev.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR)
            band = slice(4, h - 4), slice(4, w - 4)
            errs.append(float(np.abs(warped[band] - g[band]).mean()))
            prev = g
        return round(float(np.mean(errs)), 3)
    except Exception:  # noqa: BLE001
        return None


def motion_magnitude(arr: list[np.ndarray]) -> float | None:
    """Mean optical-flow magnitude. Near zero means a still image, not a video."""
    try:
        import cv2

        mags = []
        prev = _gray(arr[0])
        for cur in arr[1:]:
            g = _gray(cur)
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mags.append(float(np.linalg.norm(flow, axis=-1).mean()))
            prev = g
        return round(float(np.mean(mags)), 3)
    except Exception:  # noqa: BLE001
        return None


def flicker(arr: list[np.ndarray]) -> float | None:
    """Frame-to-frame brightness jump, normalised. High values read as strobing."""
    try:
        means = np.array([_gray(f).mean() for f in arr])
        diffs = np.abs(np.diff(means))
        return round(float(diffs.mean() / max(1e-6, means.mean())), 5)
    except Exception:  # noqa: BLE001
        return None


def sharpness(arr: list[np.ndarray]) -> float | None:
    try:
        import cv2

        vals = [float(cv2.Laplacian(_gray(f).astype(np.uint8), cv2.CV_64F).var()) for f in arr]
        return round(float(np.mean(vals)), 2)
    except Exception:  # noqa: BLE001
        return None


def summarize(frames, prompt: str) -> dict:
    if not frames:
        return {}
    try:
        arr = _np(frames)
    except Exception:  # noqa: BLE001  (numpy/PIL missing should not kill a run)
        return {}
    return {
        "clip": clip_score(frames, prompt),
        "temporal_err": temporal_consistency(arr),
        "motion": motion_magnitude(arr),
        "flicker": flicker(arr),
        "sharpness": sharpness(arr),
        "n_frames_scored": len(arr),
    }
