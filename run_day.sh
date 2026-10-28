#!/usr/bin/env bash
# 24h orchestrator. Survives SSH drops (tmux), resumes after crashes, always
# produces a report, then pushes it.
#
#   bash run_day.sh                 # full 24h
#   bash run_day.sh --dry-run       # validate the plan in ~5 min, no weights
#   bash run_day.sh --phases smoke,resolution
set -uo pipefail
cd "$(dirname "$0")"

DRY=""
PHASES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run" ;;
    --phases)  PHASES="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# ---- run inside tmux so a dropped SSH session does not kill the day --------
if [ -z "${TMUX:-}" ] && [ "${CGBENCH_NO_TMUX:-0}" != "1" ]; then
  tmux kill-session -t cgbench 2>/dev/null || true
  tmux new-session -d -s cgbench "CGBENCH_NO_TMUX=1 bash '$0' $* 2>&1 | tee -a results/day.log"
  echo "started in tmux. follow with:  tmux attach -t cgbench   (or tail -f results/day.log)"
  exit 0
fi

mkdir -p results
echo "=== cgbench day start $(date -u) dry=${DRY:-no} ===" | tee -a results/day.log

# HuggingFace is not reachable from most CN hosts; the mirror serves the same API.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

bash bootstrap.sh 2>&1 | tee -a results/day.log
python3 -m cgbench env | tee results/env.json | head -30

# ---- screen the catalogue first: the day's model list comes from this -------
echo "--- catalog $(date -u)" | tee -a results/day.log
python3 -m cgbench catalog --top "${CGBENCH_TOP:-8}" 2>&1 | tee -a results/day.log

# ---- the actual day -------------------------------------------------------
if [ -n "$PHASES" ]; then
  IFS=',' read -ra list <<< "$PHASES"
  for p in "${list[@]}"; do
    echo "--- phase $p $(date -u)" | tee -a results/day.log
    python3 -m cgbench phase "$p" --from-catalog $DRY 2>&1 | tee -a results/day.log
  done
else
  python3 -m cgbench all --from-catalog $DRY 2>&1 | tee -a results/day.log
fi

# ---- always report, even if the run was interrupted -----------------------
echo "=== report $(date -u)" | tee -a results/day.log
python3 -m cgbench report 2>&1 | tee -a results/day.log

# ---- publish --------------------------------------------------------------
echo "=== push $(date -u)" | tee -a results/day.log
python3 tools/push.py --git --webhook 2>&1 | tee -a results/day.log

date -u > results/DONE
echo "=== done $(date -u) ===" | tee -a results/day.log
echo
echo "实例还在计费。确认结果已推送后，去控制台停止/释放机器。"
