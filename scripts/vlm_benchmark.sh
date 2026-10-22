#!/usr/bin/env bash
# Usage: scripts/vlm_benchmark.sh decomposition|verification|state_tracking|next_subtask --model MODEL [--out FILE] [...]
set -euo pipefail
[ $# -gt 0 ] || { sed -n "2s/^# //p" "$0"; exit 1; }

task=$1; shift
out=out/vlm_benchmark/$task.json args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out=$2; shift 2;;
    *) args+=("$1"); shift;;
  esac
done
out=$(realpath -m "$out")
mkdir -p "$(dirname "$out")"
source "$(dirname "$0")/lib/common.sh"
dataset=INSAIT-Institute/LADDER-Offline-VLM-Benchmark
if [ "$task" = decomposition ]; then
  exec bazel run //ladder/components/inference/models/vlams:score_decomposition -- \
    --dataset "$dataset" --out "$out" "${args[@]}"
fi
exec bazel run //ladder/components/inference/models/vlams:score_subtask_states -- \
  --dataset "$dataset" --task "$task" --out "$out" "${args[@]}"
