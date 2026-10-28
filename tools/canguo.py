"""灿果Spark 库存/价格查询。只读，不下单。

  python tools/canguo.py
  python tools/canguo.py --watch 60

数据来自站点公开接口，字段含义与页面一致（金额单位见下方换算）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BASE = "https://canguoai.com/api/v1/market"
CENT = 1e6  # price_* 字段的最小单位是 1e-6 元


def get(path: str) -> dict:
    req = urllib.request.Request(BASE + path, headers={
        "User-Agent": "cgbench/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def show(sku_code: str = "dgx-spark-gb10-128g") -> dict:
    skus = {s["code"]: s for s in get("/skus").get("data", [])}
    demand = {d["sku_code"]: d for d in get("/demand").get("data", [])}
    s = skus.get(sku_code)
    if not s:
        print(f"SKU not found: {sku_code}")
        return {}

    d = demand.get(sku_code, {})
    print(f"{s['display_name']}  ({s['code']})")
    print(f"  GPU        {s['gpu_model']} x{s['gpu_count']}  {s['vram_gb']}GB "
          f"uniform={s['unified_mem']}  arch={s['arch']}")
    print(f"  整机        {s['vcpu']} vCPU / {s['ram_gb']}GB RAM / {s['disk_gb']}GB {s['disk_type']}")
    print(f"  网络/功耗   {s['bandwidth_mbps']} Mbps / {s['power_watts']}W / {s['os_name']}")
    print(f"  按量        ¥{s['price_yuan_per_hour']}/h   最小计费 {s['min_minutes']} 分钟")
    for p in s.get("plans", []):
        print(f"  {p['label']:<8}  ¥{p['price_yuan']:<7} ≈ ¥{p['equiv_yuan_per_hour']}/h")
    print(f"  存储        ¥{s['storage_price_per_gb_hour'] / CENT:.6f}/GB/小时"
          f"  (4TB 全算 ≈ ¥{4000 * s['storage_price_per_gb_hour'] / CENT:.2f}/h)")
    print(f"  现货        idle={s['idle_count']} total={s['total_count']} "
          f"queue={s['queue_length']} hosted={s['has_hosted_machines']}")
    if demand.get("reservation_only"):
        print(f"  预约制      {d.get('users', 0)} 用户在排队，{d.get('machines', 0)} 台在册")
    return {"sku": s, "demand": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", default="dgx-spark-gb10-128g")
    ap.add_argument("--watch", type=int, default=0, help="poll every N seconds")
    args = ap.parse_args()

    if not args.watch:
        show(args.sku)
        return 0

    last = None
    while True:
        try:
            info = show(args.sku)
            state = (info["sku"]["idle_count"], info["sku"]["total_count"])
            if state != last:
                print(f"[{time.strftime('%H:%M:%S')}] idle={state[0]} total={state[1]}")
                last = state
        except Exception as exc:  # noqa: BLE001
            print("poll error:", exc)
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
