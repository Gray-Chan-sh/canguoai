"""Screen the HuggingFace video-model catalogue against what a GB10 can actually run.

The question is not "what is popular", it is "what can this box run this week,
with the stack that exists on arm64". So the filter is deliberately harsh:

  * must be a diffusers repo (ComfyUI/GGUF-only weights are a different runtime)
  * must not be gated (a 24h run cannot wait on a manual approval)
  * must have published parameter counts (otherwise size is a guess)
  * must fit in ~90 GiB at bf16, or ~45 GiB if an fp8 build exists

Everything rejected is kept with a reason. The rejection list is evidence too:
"87 个候选里只有 9 个能直接跑" is a finding for the podcast.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .harness import OUT

HF = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CATALOG = OUT / "catalog.json"
CACHE = OUT / ".hf-cache.json"
CACHE_TTL_S = 24 * 3600

PIPELINES = ("text-to-video", "image-to-video")
HEADROOM_GIB = 90.0     # leave ~38 GiB of the unified pool for KV, VAE tiles, OS
EXTRA_GIB = 3.0         # VAE + text encoder + tokenizer, on top of the DiT
MAX_CANDIDATES = 120

FAMILY = [
    # (pattern, fps, steps, guidance, multiple)
    (r"wan2\.2.*a14b|wan2\.1.*14b", 16, 30, 5.0, 16),
    (r"wan2\.2", 24, 30, 5.0, 16),
    (r"wan2\.1", 16, 30, 5.0, 16),
    (r"ltx", 24, 30, 3.0, 32),
    (r"hunyuan", 24, 30, 6.0, 16),
    (r"cogvideo", 8, 30, 6.0, 8),
    (r"mochi", 30, 30, 4.5, 16),
]
DEFAULT = (24, 30, 5.0, 16)

# Repos that publish no safetensors metadata. Values are from the model cards and
# are marked in the output so nobody mistakes them for measured numbers.
OVERRIDES = {
    "tencent/HunyuanVideo": (13.0, "model card: 13B, repo ships .pt weights"),
    "Lightricks/LTX-Video": None,  # fills itself in when metadata is present
}


def _get(url: str, attempts: int = 4) -> dict:
    for i in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": "cgbench-catalog/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and i < attempts - 1:
                time.sleep(1.5 * (2 ** i))   # 1.5s, 3s, 6s
                continue
            raise
        except Exception:
            if i < attempts - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            raise
    raise RuntimeError("unreachable")


def _cache_load() -> dict:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _cache_save(cache: dict) -> None:
    try:
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _family(model_id: str):
    low = model_id.lower()
    for pat, fps, steps, g, mult in FAMILY:
        if re.search(pat, low):
            return fps, steps, g, mult
    return DEFAULT


def _is_gguf_comfy(model: dict) -> bool:
    tags = [t.lower() for t in model.get("tags", [])]
    return any(t in ("gguf", "comfyui") for t in tags) or "gguf" in model["id"].lower()


def _estimate_gib(params: int) -> float:
    return round(params * 2 / 2**30 + EXTRA_GIB, 1)


def scan(limit: int = MAX_CANDIDATES, detail_cap: int = 60) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    seen: dict[str, dict] = {}
    for pipe in PIPELINES:
        url = f"{HF}/api/models?pipeline_tag={pipe}&sort=downloads&direction=-1&limit={limit}"
        try:
            for m in _get(url):
                seen.setdefault(m["id"], m)
        except Exception as exc:  # noqa: BLE001
            print(f"[catalog] {pipe} listing failed: {exc}", flush=True)

    listed = sorted(seen.values(), key=lambda m: m.get("downloads", 0), reverse=True)

    # Popularity is a bad sole filter: HunyuanVideo has few downloads on the mirror
    # but is a reference model for this class. Always examine known families too.
    def known_family(mid: str) -> bool:
        low = mid.lower()
        return any(re.search(p, low) for p, *_ in FAMILY)

    head = listed[:detail_cap]
    extra = [m for m in listed if known_family(m["id"]) and m not in head]
    detail_set = (head + extra)[: detail_cap + 20]
    print(f"[catalog] {len(listed)} unique repos; fetching detail for {len(detail_set)}", flush=True)

    out = []
    for mid, m in seen.items():
        out.append({
            "key": re.sub(r"[^a-z0-9]+", "-", mid.split("/")[-1].lower()).strip("-")[:40],
            "id": mid,
            "pipeline": m.get("pipeline_tag"),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "last_modified": m.get("lastModified"),
            "gated": bool(m.get("gated")),
            "library": m.get("library_name"),
            "tags": m.get("tags", []),
            "params_b": None,
            "est_gib_bf16": None,
            "tier": "unknown",
            "runnable": False,
            "why": "not examined (low downloads)",
            "_examined": False,
        })
    index = {e["id"]: e for e in out}

    cache = _cache_load()
    now = time.time()

    def detail(m: dict):
        mid = m["id"]
        hit = cache.get(mid)
        if hit and now - hit.get("_ts", 0) < CACHE_TTL_S:
            return mid, hit["data"]
        try:
            # blobs=true is what actually carries `safetensors` + per-file sizes;
            # the plain endpoint omits them for a lot of repos.
            full = _get(f"{HF}/api/models/{mid}?blobs=true")
            cache[mid] = {"_ts": now, "data": full}
            return mid, full
        except Exception as exc:  # noqa: BLE001
            return mid, {"__error__": f"{type(exc).__name__}: {str(exc)[:100]}"}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for n, (mid, full) in enumerate(ex.map(detail, detail_set), 1):
            e = index[mid]
            e["_examined"] = True
            if "__error__" in full:
                e["why"] = f"detail failed: {full['__error__']}"
                continue
            st = full.get("safetensors") or {}
            params = st.get("total") or (sum((st.get("parameters") or {}).values()) or None)
            e["gated"] = bool(full.get("gated"))
            e["library"] = full.get("library_name")
            e["license"] = (full.get("cardData") or {}).get("license")
            if params:
                e["params_b"] = round(params / 1e9, 2)
                e["est_gib_bf16"] = _estimate_gib(params)
            elif mid in OVERRIDES and OVERRIDES[mid]:
                pb, why = OVERRIDES[mid]
                e["params_b"] = pb
                e["est_gib_bf16"] = _estimate_gib(int(pb * 1e9))
                e["why"] = f"params from override ({why})"
                e["params_source"] = "override"
            if n % 15 == 0:
                print(f"[catalog]   {n}/{len(detail_set)}", flush=True)

    for e in out:
        if not e.get("_examined"):
            continue
        if e["why"].startswith("detail failed"):
            continue
        reasons = []
        low = e["id"].lower()
        if any(k in low for k in ("s2v", "vace", "video2video", "v2v", "-edit")):
            reasons.append("needs extra conditioning input (audio/reference video)")
        if e["gated"]:
            reasons.append("gated")
        if _is_gguf_comfy({"id": e["id"], "tags": e["tags"]}):
            reasons.append("gguf/comfyui runtime, not diffusers")
        elif e["library"] != "diffusers":
            reasons.append(f"library={e['library']}")
        if e["params_b"] is None:
            reasons.append("no safetensors params")
        elif e["est_gib_bf16"] > HEADROOM_GIB:
            reasons.append(f"bf16 {e['est_gib_bf16']}GiB > {HEADROOM_GIB:.0f}GiB")
        e["why"] = "; ".join(reasons) or (
            "ok (params from override)" if e.get("params_source") == "override" else "ok")
        e["runnable"] = not reasons
        e["tier"] = ("bf16" if e["runnable"] else
                     "needs-quant" if e["est_gib_bf16"] and e["est_gib_bf16"] > HEADROOM_GIB else "no")

    out.sort(key=lambda e: (e["runnable"], e["downloads"]), reverse=True)
    _cache_save(cache)
    return out


def shortlist(entries: list[dict], top: int = 8, include_i2v: bool = False) -> list[dict]:
    """Best runnable models, one per family first, then fill by score.

    Default is text-to-video only: an image-to-video pipeline needs a conditioning
    frame, which would make the speed numbers incomparable across models.
    """
    import math

    def fam(e):
        low = e["id"].lower()
        for pat, *_ in FAMILY:
            if re.search(pat, low):
                return pat
        return "other"

    def is_t2v(e):
        if e.get("pipeline") == "text-to-video" or "text-to-video" in (e.get("tags") or []):
            return True
        # LTX-Video and friends are dual-purpose but tagged image-to-video; the
        # repo name is the tell. Pure i2v checkpoints carry it explicitly.
        name = e["id"].lower()
        i2v_markers = ("i2v", "img2vid", "image2video", "image-to-video", "svd")
        return e.get("pipeline") == "image-to-video" and not any(k in name for k in i2v_markers)

    ok = [e for e in entries if e["runnable"] and (include_i2v or is_t2v(e))]

    def score(e):
        dl = math.log10(max(1, e.get("downloads", 0)))
        lk = math.log10(max(1, e.get("likes", 0)))
        return dl + 1.5 * lk

    ok.sort(key=score, reverse=True)

    def duplicate(e, picked):
        """Same family and same ballpark size = the same experiment twice."""
        for p in picked:
            if fam(p) != fam(e):
                continue
            a, b = e.get("params_b") or 0, p.get("params_b") or 0
            if a and b and abs(a - b) / max(a, b) < 0.15:
                return True
        return False

    picked: list[dict] = []
    used: set[str] = set()
    for e in ok:                      # pass 1: coverage across families
        f = fam(e)
        if f not in used:
            used.add(f)
            picked.append(e)
    for e in ok:                      # pass 2: fill by score, no near-duplicates
        if len(picked) >= top:
            break
        if e not in picked and not duplicate(e, picked):
            picked.append(e)
    return picked[:top]


def build(top: int = 8, write: bool = True, include_i2v: bool = False) -> dict:
    entries = scan()
    picked = shortlist(entries, top, include_i2v=include_i2v)

    for e in picked:
        fps, steps, g, mult = _family(e["id"])
        e.update(fps=fps, steps=steps, guidance=g, multiple=mult,
                 label=e["id"].split("/")[-1][:28])
        # keys must be unique even when two repos share a basename
        base = e["key"]
        n = 1
        while any(p["key"] == e["key"] and p["id"] != e["id"] for p in picked):
            n += 1
            e["key"] = f"{base}-{n}"

    doc = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "endpoint": HF,
        "headroom_gib": HEADROOM_GIB,
        "total_scanned": len(entries),
        "runnable": sum(1 for e in entries if e["runnable"]),
        "shortlist_keys": [e["key"] for e in picked],
        "entries": entries,
    }
    if write:
        CATALOG.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scanned {doc['total_scanned']} models via {HF}")
    print(f"runnable on GB10: {doc['runnable']}\n")
    print(f"{'model':<44}{'params':>7} {'bf16 GiB':>9}  why")
    print("-" * 92)
    for e in entries[:30]:
        p = f"{e['params_b']}B" if e["params_b"] else "?"
        g = e["est_gib_bf16"] or 0
        mark = "*" if e in picked else " "
        print(f"{mark}{e['id'][:43]:<43}{p:>7} {g:>9}  {e['why'][:40]}")
    print(f"\nshortlist ({len(picked)}):")
    for e in picked:
        print(f"  - {e['key']:<28} {e['id']}  ({e['est_gib_bf16']} GiB bf16)")
    if write:
        print(f"\nwritten -> {CATALOG}")
    return doc


def load_shortlist() -> list[dict]:
    if not CATALOG.exists():
        return []
    doc = json.loads(CATALOG.read_text(encoding="utf-8"))
    keys = set(doc.get("shortlist_keys", []))
    return [e for e in doc.get("entries", []) if e["key"] in keys]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    build(top=ap.parse_args().top)
