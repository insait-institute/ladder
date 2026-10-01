#!/usr/bin/env bash
# Evaluate a method on a LADDER suite.
#
#   scripts/eval.sh --suite SUITE --method METHOD [--task TASK ...] [--episodes N] [--output DIR]
#
#   --suite     simplerenv_long | robotarena_long | robolab_long
#   --method    low_level_policy | periodic_trigger | proprioceptive_trigger | proprioceptive_verifier |
#               volo | privileged_verifier | cap_x | tiptop | psp_agent | gpt_policy | scripted
#   --task      one or more tasks (TASKS.md lists them), or "all". Default: a 2-subgoal task.
#   --episodes  episodes per task (default 24 on simplerenv_long, 8 on the other two)
#   --output    results directory (default: out)
#
# EVALUATION.md lists the suites and methods. Results go to OUTPUT/<suite>/<low-level policy>/
# METHOD_c1/, which the dashboard reads. Methods that call a vision-language model read
# VLM_BASE_URL and VLM_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

suite= method= tasks=() episodes= output=out
while [ $# -gt 0 ]; do
  case "$1" in
    --suite) suite=$2; shift 2;;
    --method) method=$2; shift 2;;
    --task) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do tasks+=("$1"); shift; done;;
    --episodes) episodes=$2; shift 2;;
    --output) output=$2; shift 2;;
    -h|--help) usage 0;;
    *) echo "unknown argument: $1"; usage 1;;
  esac
done
[ -n "$suite" ] && [ -n "$method" ] || usage 1

M=ladder/components/inference/models
R=ladder/components/inference/robots
# the suite's name in the code (and in the output tree the dashboard reads)
name=$suite
case "$suite" in
  simplerenv_long) suite=simpler default=all2_plate;;
  robotarena_long) suite=robot_arena default=scene54_split_2;;
  robolab_long) suite=robolab default=comp_bin_2;;
  *) echo "unknown suite: $suite"; usage 1;;
esac
[ ${#tasks[@]} -gt 0 ] || tasks=("$default")
[ "${tasks[*]}" = all ] && mapfile -t tasks < "scripts/tasks/$name.txt"

# the method's config suffix (methods that drive a low-level policy) or its package
learned=1
case "$method" in
  low_level_policy) suffix=;;
  periodic_trigger) suffix=_plan_periodic;;
  proprioceptive_trigger) suffix=_plan_gripper;;
  proprioceptive_verifier) suffix=_plan_verifier;;
  volo) suffix=_volo;;
  privileged_verifier) suffix=_oracle;;
  cap_x) learned=0 pkg=capx cfg=capx_${suite}_composition_tiered.yaml;;
  tiptop) learned=0 pkg=tiptop cfg=vlam_${suite}_composition_tiptop.yaml;;
  psp_agent) learned=0 pkg=harness cfg=harness_${suite}_composition.yaml;;
  gpt_policy) learned=0 pkg=gpt_policy cfg=gpt_policy_${suite}_composition.yaml;;
  scripted) learned=0 pkg=scripted;;
  *) echo "unknown method: $method"; usage 1;;
esac
if [ "$method" != low_level_policy ] && [ "$method" != privileged_verifier ] && [ "$method" != scripted ] && { [ -z "${VLM_BASE_URL:-}" ] || [ -z "${VLM_API_KEY:-}" ]; }; then
  echo "$method calls a vision-language model: export VLM_BASE_URL and VLM_API_KEY first"; exit 1
fi

if [ $learned = 1 ] && [ "$suite" = robolab ]; then
  policy=pi05 config=$M/external/openpi/configs/robolab/composition/openpi_robolab_composition$suffix.yaml
  export RUN_MODEL=models/external/openpi:run_model INFERENCE_SERVER=0
elif [ $learned = 1 ]; then
  policy=bridge config=$M/vlams/configs/$suite/composition/vlam_${suite}_composition$suffix.yaml
  export SESSIONS_HOME=${SESSIONS_HOME:-$HOME/experiments/vlams/control}
  export SESSION_NAME=sess_bridge_policy CKPT_NAMES=model-checkpoint-ema-030000.pt
  if [ ! -f "$SESSIONS_HOME/$SESSION_NAME/checkpoints/$CKPT_NAMES" ]; then
    echo "Downloading the Bridge policy (11 GB) to $SESSIONS_HOME/$SESSION_NAME"
    bazel run //ladder/components/dashboard:fetch -- policy --repo INSAIT-Institute/LADDER-Bridge-Policy --out "$SESSIONS_HOME/$SESSION_NAME"
  fi
elif [ "${pkg:-}" = scripted ]; then
  # the launchers see the scripted policy in the config: nothing to download or serve
  policy=none config=$M/vlams/configs/vlam_simpler_scripted.yaml
  [ "$suite" = simpler ] || config=$M/vlams/configs/$suite/vlam_${suite}_scripted.yaml
  export SESSION_NAME=scripted
else
  policy=none config=$M/$pkg/configs/$suite/composition/$cfg
  export RUN_MODEL=models/$pkg:run_model PUBLISH_DEPTH=True SESSION_NAME=$method
  if [ "$suite" = robolab ]; then export INFERENCE_SERVER=0; else export NO_VLA=1; fi
fi
out=$(realpath -m "$output/$suite/$policy/${method}_c1")
echo "$method on $name (${tasks[*]}) -> $out"

case "$suite" in
  simpler)
    ENV_NAMES="${tasks[*]}" N_ENVS=${episodes:-24} VIDEO=True OUT_ROOT="$out" \
    SIM_CONFIG=$R/simpler/configs/widowx_composition.yaml SIMPLER_CONFIG=$config \
    bash $M/vlams/scripts/vlam_simpler_bridge.sh;;
  robot_arena)
    [ -z "$episodes" ] || echo "note: RobotArena-Long runs 8 episodes per task (--episodes is ignored)"
    SCENES="${tasks[*]}" CAMPAIGN_ROOT="$out" \
    RA_SERVER_CONFIG=$R/robot_arena/configs/widowx_composition_k8.yaml RA_CLIENT_CONFIG=$config \
    bash $M/vlams/scripts/vlam_robot_arena.sh;;
  robolab)
    TASKS="${tasks[*]}" NUM_ENVS=${episodes:-8} NUM_RUNS=1 CAMPAIGN_ROOT="$out" LADDER_CONFIG=$config \
    bash $R/robolab/scripts/vlam_robolab.sh;;
esac
# a launcher can finish while every episode of a task failed to run: check each task has a score
missing=()
for t in "${tasks[@]}"; do
  f=$(find "$out" -path "*/$t/results.yaml" 2>/dev/null | head -1)
  { [ -n "$f" ] && grep -q "success:" "$f"; } || missing+=("$t")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "No results for: ${missing[*]}. The logs are under $out."; exit 1
fi
echo "Done. Results: $out"
