#!/usr/bin/env bash
# Usage: scripts/dashboard.sh [DIR ...] [--port PORT]
#        scripts/dashboard.sh fetch [--out DIR] [--suite SUITE ...] [--method METHOD ...] [--no-videos]
set -euo pipefail

if [ "${1:-}" = fetch ]; then
  shift
  out=$HOME/ladder_episodes args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --out) out=$2; shift 2;;
      *) args+=("$1"); shift;;
    esac
  done
  out=$(realpath -m "$out")
  source "$(dirname "$0")/lib/common.sh"
  exec bazel run //ladder/components/dashboard:fetch -- episodes \
    --repo INSAIT-Institute/LADDER-Evaluation-Episodes --out "$out" "${args[@]}"
fi

dirs=() args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --port|--host) args+=("$1" "$2"); shift 2;;
    *) dirs+=("$(realpath "$1")"); shift;;
  esac
done
source "$(dirname "$0")/lib/common.sh"
[ ${#dirs[@]} -gt 0 ] || dirs=("$PWD/out")
exec bazel run //ladder/components/dashboard:serve -- --results "${dirs[@]}" "${args[@]}"
