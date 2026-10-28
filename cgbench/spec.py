"""The contract for the 24h run: what we are trying to answer, and the grids we run.

Everything in OBJECTIVES must be answerable from results.jsonl alone. If a number
cannot be produced by report.py, it does not belong here.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# O1..O7 are the whole point of the day. Each maps to a phase below.
OBJECTIVES = [
    {
        "id": "O1",
        "question": "在可接受时延内，能生成的最大分辨率是多少？",
        "phase": "resolution",
        "answer": "预约时延预算（默认 600s/条）内完成的最高分辨率，以及每一档的秒/千像素",
    },
    {
        "id": "O2",
        "question": "720p 下能生成的最长片段是多少秒？",
        "phase": "duration",
        "answer": "逐档时长到失败为止，记录最后一次成功的 frames/seconds 与峰值显存",
    },
    {
        "id": "O3",
        "question": "稳态吞吐是多少条/小时？",
        "phase": "throughput",
        "answer": "固定最优配置连跑，取中位数；含加载摊销与不含加载摊销两个口径",
    },
    {
        "id": "O4",
        "question": "显存天花板在哪，是 OOM 还是掉速？",
        "phase": "resolution",
        "answer": "每档峰值显存 + 失败点；源于同一次扫描，不重复测",
    },
    {
        "id": "O5",
        "question": "降精度能换回多少速度，代价是什么？",
        "phase": "precision",
        "answer": "bf16 vs fp16 vs fp8 的耗时比，以及质量代理指标的差值",
    },
    {
        "id": "O6",
        "question": "连续跑一天会不会降频/泄漏/失败？",
        "phase": "soak",
        "answer": "首小时 vs 末小时的 s/it、峰值显存漂移、失败率、温度/功耗曲线",
    },
    {
        "id": "O7",
        "question": "参数变化时质量怎么退化？",
        "phase": "duration",
        "answer": "CLIP 对齐度、时序一致性、运动幅度随分辨率/时长/步数的变化",
    },
]

# --------------------------------------------------------------------------
# Model registry. fps/resolution are the model's native operating point, not a limit.
MODELS = {
    "wan22-5b": {
        "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "label": "Wan2.2 TI2V 5B",
        "fps": 24,
        "steps": 30,
        "guidance": 5.0,
        "multiple": 16,
        "note": "最快的一档，先把流程和指标跑通",
    },
    "wan22-14b": {
        "id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "label": "Wan2.2 T2V A14B (MoE)",
        "fps": 16,
        "steps": 30,
        "guidance": 5.0,
        "multiple": 16,
        "note": "MoE，128GB 统一内存的主战场",
    },
    "wan21-14b": {
        "id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "label": "Wan2.1 T2V 14B",
        "fps": 16,
        "steps": 30,
        "guidance": 5.0,
        "multiple": 16,
        "note": "稠密 14B，和 MoE 做对照",
    },
    "ltx": {
        "id": "Lightricks/LTX-Video",
        "label": "LTX-Video",
        "fps": 24,
        "steps": 30,
        "guidance": 3.0,
        "multiple": 32,
        "note": "速度基线，单位时间产出最高",
    },
    "hunyuan": {
        "id": "tencent/HunyuanVideo",
        "label": "HunyuanVideo",
        "fps": 24,
        "steps": 30,
        "guidance": 6.0,
        "multiple": 16,
        "note": "画质档，代价是慢",
    },
}

RESOLUTIONS = {
    "480p": (832, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1088),
}

PROMPTS = [
    "A slow dolly shot through a rain-soaked neon alley, reflections on wet asphalt, cinematic",
    "A hummingbird hovering beside a red flower, wings blurred, shallow depth of field, macro",
    "Aerial view of a winding mountain road at sunrise, low fog in the valley, gradual reveal",
    "A chef tossing vegetables in a wok, flames rising, steam and motion blur, close-up",
]

NEGATIVE = "blurry, low quality, static, watermark, text, distorted faces, jitter"

# --------------------------------------------------------------------------
# Phase budgets sum to 1440 min = 24h. They are soft: a phase stops launching new
# runs at its deadline and keeps whatever finished.
PHASES = {
    "smoke": {
        "minutes": 20,
        "desc": "单条最短路径，确认管线/编码/指标全链路可用",
        "model": "wan22-5b",
        "configs": [
            {"label": "smoke", "res": "480p", "seconds": 2.0, "steps": 10},
        ],
    },
    "resolution": {
        "minutes": 150,
        "desc": "分辨率阶梯：480p -> 720p -> 1080p，固定 5s",
        "model": "wan22-5b",
        "configs": [
            {"label": "480p", "res": "480p", "seconds": 5.0, "steps": 30},
            {"label": "720p", "res": "720p", "seconds": 5.0, "steps": 30},
            {"label": "1080p", "res": "1080p", "seconds": 5.0, "steps": 30},
        ],
    },
    "duration": {
        "minutes": 180,
        "desc": "时长阶梯：720p 下 2/5/10/20 秒，固定 30 步",
        "model": "wan22-5b",
        "configs": [
            {"label": "2s", "res": "720p", "seconds": 2.0, "steps": 30},
            {"label": "5s", "res": "720p", "seconds": 5.0, "steps": 30},
            {"label": "10s", "res": "720p", "seconds": 10.0, "steps": 30},
            {"label": "20s", "res": "720p", "seconds": 20.0, "steps": 30},
        ],
    },
    "precision": {
        "minutes": 120,
        "desc": "精度对比：bf16 / fp16 / fp8（fp8 不可用时如实记录）",
        "model": "wan22-5b",
        "configs": [
            {"label": "bf16", "res": "720p", "seconds": 5.0, "steps": 30, "dtype": "bfloat16"},
            {"label": "fp16", "res": "720p", "seconds": 5.0, "steps": 30, "dtype": "float16"},
            {"label": "fp8", "res": "720p", "seconds": 5.0, "steps": 30, "dtype": "fp8"},
        ],
    },
    "throughput": {
        "minutes": 120,
        "desc": "固定最优配置连跑 6 条，测稳态吞吐与方差",
        "model": "wan22-5b",
        "configs": [
            {"label": f"rep{i}", "res": "720p", "seconds": 5.0, "steps": 30} for i in range(6)
        ],
    },
    "models": {
        "minutes": 240,
        "desc": "四模型横评，同一组参数，比速度与质量代理",
        "model": None,  # iterate MODELS
        "configs": [
            {"label": "std", "res": "720p", "seconds": 5.0, "steps": 30},
        ],
        "models": ["wan22-5b", "wan22-14b", "ltx", "hunyuan"],
    },
    "soak": {
        "minutes": 610,
        "desc": "剩余时间全部做稳态：固定配置循环，看降频/泄漏/失败率",
        "model": "wan22-5b",
        "configs": [
            {"label": "soak", "res": "720p", "seconds": 5.0, "steps": 30},
        ],
        "loop": True,
    },
}

# A run is considered failed (for "max resolution") if it exceeds this wall time,
# even if it would eventually finish. 10 minutes is the practical patience limit.
LATENCY_BUDGET_S = 600
