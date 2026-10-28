# canguo-videobench — 单台 DGX Spark 的视频生成上限测试

在灿果Spark（整机独占、DGX OS、arm64、128GB 统一内存）上跑**一天**，回答七个问题。
不是"能不能跑通"，是**这台机器生成视频的能力边界在哪**。

---

## 测试目标（全部可从结果文件复算）

| ID | 问题 | 判据 |
|---|---|---|
| **O1** | 可接受时延内最大分辨率是多少？ | ≤600s/条 的前提下能完成的最高档 |
| **O2** | 720p 下最长能生成多少秒？ | 逐档加到失败为止 |
| **O3** | 稳态吞吐多少条/小时？ | 固定配置连跑 6 条的中位数 |
| **O4** | 显存天花板在哪，OOM 还是掉速？ | 全场景峰值 + 失败点 |
| **O5** | 降精度换回多少速度，代价是什么？ | bf16/fp16/fp8 耗时比 + 质量代理差值 |
| **O6** | 连续跑一天会降频/泄漏/失败吗？ | 首段 vs 末段的 s/it、显存漂移、失败率 |
| **O7** | 参数变化时质量怎么退化？ | CLIP 对齐、时序误差、运动幅度、闪烁 |

`python -m cgbench plan` 会把上面的表和 24 小时排程打印出来。

---

## 一天的排程（合计 1440 分钟）

| 阶段 | 分钟 | 内容 |
|---|---|---|
| `smoke` | 20 | 单条最短路径，打通管线/编码/指标 |
| `resolution` | 150 | 480p → 720p → 1080p，固定 5s |
| `duration` | 180 | 720p 下 2/5/10/20 秒 |
| `precision` | 120 | bf16 / fp16 / fp8 |
| `throughput` | 120 | 最优配置连跑 6 条 |
| `models` | 240 | 筛选出的 8 个模型横评（`--from-catalog`） |
| `soak` | 610 | 剩余时间固定配置循环，测稳态 |

阶段是**软预算**：到点就不再发起新任务，已完成的全部保留。所以某一段超时不会毁掉一天。

---

## 模型筛选（每天开跑前先做）

```bash
python3 -m cgbench catalog --top 8
```

扫 HuggingFace 上 `text-to-video` / `image-to-video` 的**全部主流仓库**（当前 240 个），
再对前 80 个拉详情，按四条硬规则筛：

| 规则 | 理由 |
|---|---|
| 必须是 diffusers 仓库 | ComfyUI/GGUF 权重是另一套运行时，不在同一天里比 |
| 不能是 gated | 24 小时里等不到人工审批 |
| 必须有参数量 | 否则显存占用只能靠猜 |
| bf16 ≤ 90 GiB | 128GB 统一内存要给 VAE、KV、OS 留余量 |
| 不能需要额外条件输入 | `S2V`/`VACE`/`v2v` 要音频或参考视频，跑出来的数不可比 |

最近一次实跑结果（镜像 `hf-mirror.com`，240 个仓库）：

```
runnable on GB10: 20 / 240
shortlist (8):
  ltx-video                            6.6 GiB
  wan2-1-t2v-14b                      29.6 GiB
  fastvideo-fasth3-4-step-preview     68.3 GiB
  wan2-1-t2v-1-3b                      5.6 GiB
  wan2-2-ti2v-5b-diffusers            12.3 GiB
  hunyuanvideo-1-5-diffusers-720p-t2v 18.5 GiB
  ltx-2                               38.2 GiB
  animatelcm                           3.8 GiB
```

**被拒的也全部留档**（`results/catalog.json` 里有 `why`）：gated、GGUF、参数量过大、
需要条件输入……这张拒绝清单本身就是结论——"生态允许你跑什么"比"模型有多强"更决定一天的产出。

国内主机连不上 huggingface.co，代码默认走 `HF_ENDPOINT=https://hf-mirror.com`。
镜像会 429 限流，所以有 4 次指数退避重试 + 24 小时本地缓存（`results/.hf-cache.json`）。

`--from-catalog` 会把 `models` 阶段换成这份 shortlist，不再用 `spec.py` 里写死的那四个。

---

## 播客素材

```bash
python3 -m cgbench shots                      # 打印分镜脚本
python3 -m cgbench.capture detect             # 这台机器能拍什么
python3 -m cgbench.capture run 03-smoke -- python3 -m cgbench phase smoke
```

三种素材，能力不同就如实降级：

| 素材 | 条件 | 产物 |
|---|---|---|
| 终端录像 | 有 `pty`（Linux 都有） | `results/captures/*.cast`，asciinema v2，可直接播放或 `agg` 转 GIF |
| 桌面截图 | 有 `DISPLAY` + gnome-screenshot/scrot | `results/captures/*.png` |
| 桌面录屏 | 有 `DISPLAY` + ffmpeg | `results/captures/desktop.mp4` |
| 成片九宫格 | 有 av/PIL | `results/captures/*-sheet.png`，可直接做封面 |

`shots` 给出的分镜包含**命令、建议时长、看点、解说提示**，按 O1–O7 的顺序排好，
是剪辑脚本而不是素材清单。

纯 SSH 下没有截图和桌面录屏——这是事实，不伪造。终端 cast 加关键帧九宫格已经够叙事；
真需要桌面就在灿果控制台开 VNC 再跑。

---

## 跑起来

```bash
# 0) 先在本机（或任何有 python 的机器）验证整套编排，不下载权重，几分钟
python3 -m cgbench all --dry-run
python3 -m cgbench report

# 1) 上机
bash bootstrap.sh
python3 -m cgbench env

# 2) 正式跑（自动进 tmux、自动续跑、跑完自动出报告并推送）
bash run_day.sh
```

SSH 断了不要紧：`tmux attach -t cgbench`。进程崩了重跑 `bash run_day.sh`，已完成的项不会重来
（`results/results.jsonl` 是唯一的真相来源，加 `--no-resume` 才会重测）。

---

## 事后分析

```bash
python3 -m cgbench report          # -> results/publish/report.md
```

产物三件：

- `report.md` — 按 O1..O7 的顺序给结论，不是罗列表格
- `metrics.csv` — 每条一行，直接丢进 pandas / Excel 画曲线
- `frontier.json` — 机器可读的边界值，以后换机器可以直接对比

CSV 里有 `s_per_it`、`s_per_mpframe`（每百万像素帧耗时）、`peak_gib`、`clip`、
`temporal_err`、`motion`、`flicker`——这几个字段就是全部的分析原料。

---

## 推送结果

```bash
export CGBENCH_RESULTS_REMOTE=https://github.com/<you>/canguo-results.git
export CGBENCH_GIT_TOKEN=<fine-grained PAT，只给这一个 repo 的 contents:write>
export CGBENCH_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
export CGBENCH_WEBHOOK_FORMAT=feishu      # feishu | wecom | slack

python3 tools/push.py --git --webhook
```

只推**小文件**：report.md、metrics.csv、frontier.json、results.jsonl、最多 6 个缩略视频。
原始 mp4 留在机器上——从计费实例往外推几个 GB 是最容易浪费钱的操作。

token 通过 `git -c http.extraheader=...` 一次性传入，**不落进 `.git/config`**。

---

## 库存与价格

```bash
python3 tools/canguo.py            # 当前 SKU、价格、现货
python3 tools/canguo.py --watch 60 # 有货就提示
```

注意：这个接口只读。下单还是在网页上做，钱的事不交给脚本。

---

## 几个会咬人的地方

- **fp8 很可能不可用**。GB10 是 sm_121，`torchao` 的 float8 路径在这个组合上未必成熟。
  代码里会如实记录 `fp8: unsupported (...)` 并回落到 bf16——**这本身就是 O5 的答案**，
  不是失败。看到这行不要去修它，去报告里写清楚。
- **不要用 flash-attn / xformers**。aarch64 上都没有轮子，源码编译要 1–2 小时且大概率失败。
  diffusers 默认走 SDPA，够用。
- **模型加载是真实成本**。一个 14B checkpoint 从 4TB NVMe 读起来要几十秒，
  所以引擎在阶段内复用，切模型才重新加载。报告里 `cold` 字段标记了哪次含加载。
- **帧数必须满足 4n+1**（Wan / Hunyuan 的时序 VAE 约束），代码里已经对齐。
- **分辨率必须是 16 或 32 的倍数**，否则 VAE 会报错或产出接缝。
- **生成时长不是线性增长**。attention 随时长平方增长，20s 那档很可能直接超时——
  这正好是 O2 想要的答案。

---

## 这套东西我没能替你验证的部分

我只能保证：代码通过编译、编排逻辑正确、`--dry-run` 能跑完整套并产出报告。
**真实模型的 pipeline 调用我无法在这里验证**——没有 GPU、没有权重。

所以设计上做了两层保护：

1. `--dry-run` 用合成引擎把编排、续跑、指标、报告、推送全部跑一遍；
2. 真实引擎按 `inspect.signature` 过滤参数，模型不认的关键字会被丢掉并记进
   `engine_info.dropped_kwargs`，而不是抛异常毁掉整天。

真机上第一次跑 `smoke` 阶段（20 分钟）就是为了暴露这类问题——**在花 24 小时之前**。
