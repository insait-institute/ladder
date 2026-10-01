#!/bin/bash

# RoboLab eval launcher (analogue of vlam_robot_arena.sh). Ladder models via the standard
# run_model/SimplerEpisodeRunner stack (env_server speaks the RobotServer ZMQ protocol); also emits
# RoboLab-native episode_results.jsonl so their dashboard can read the campaign.
# Run from the repo root:
#   SESSION_NAME=sess_... CKPT_NAME=model-checkpoint-ema-050000.pt \
#   TASKS="BananaOnPlateTask" NUM_ENVS=8 bash ladder/components/inference/robots/robolab/scripts/vlam_robolab.sh
# TASKS is split round-robin over LANES_PER_GPU parallel lanes on the one GPU (default: what the
# card fits). Lane outputs land under $CAMPAIGN_ROOT/lane<i>/output/<ckpt>/, and the per-task
# results under $CAMPAIGN_ROOT/<ckpt>/<task>/results.yaml.
#
# LADDER_CONFIG/TASKS also drive the long-horizon composition benchmark, whose tasks are listed in
# robots/robolab/composition/README.md; TASKS takes composition task NAMES (comp_bin_6, comp_split_plate_4)
# exactly as SCENES/ENV_NAMES do on the RobotArena and SimplerEnv launchers.

set -u


# EVAL_* are generic aliases of this launcher's own variables; a variable set directly wins.
: "${CKPT_NAME:=${EVAL_CKPT:-}}"
: "${TASKS:=${EVAL_UNITS:-}}"
: "${LADDER_CONFIG:=${EVAL_CONFIG:-}}"
: "${NUM_ENVS:=${EVAL_ENVS:-}}"
: "${VIDEO:=${EVAL_VIDEO:-}}"
: "${CAMPAIGN_ROOT:=${EVAL_OUT:-}}"
LADDER_CONFIG=${LADDER_CONFIG:-ladder/components/inference/models/vlams/configs/robolab/vlam_robolab.yaml}
TASKS=${TASKS:-"BananaOnPlateTask"}
NUM_ENVS=${NUM_ENVS:-1}
NUM_RUNS=${NUM_RUNS:-2}
# The default output directory; `set -u` would otherwise stop the run on an unbound CAMPAIGN_ROOT.
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-/home/$USER/experiments/vlams/robolab/run_$(date +%Y%m%d_%H%M%S)}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-default}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Scaled by lane count, now that every Isaac app boots at once rather than one at a time: a lone
# app takes ~10 min, and N booting together contend for the same cores.
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-$((1800 + 120 * ${LANES_PER_GPU:-1}))}


# The whole launcher-side interface for a policy that is not one of our checkpoints: which
# run_model target runs it (its own, when the model brings its own pip environment) and whether
# anything has to be served. A new model is then a config plus a benchmark entry, not an edit here.
RUN_MODEL=${RUN_MODEL:-models/vlams:run_model}
# The scripted completability policy carries no weights and serves nothing; it drives from the
# privileged state env_server publishes.
grep -q "scripted_policy.ScriptedPolicy" "$LADDER_CONFIG" 2>/dev/null && : "${INFERENCE_SERVER:=0}"
INFERENCE_SERVER=${INFERENCE_SERVER:-1}

# With nothing served there is no checkpoint of ours, so these two only label the output; name
# them after the config, or the results land under a name that reads as one of our checkpoints.
if [ "$INFERENCE_SERVER" = "0" ]; then
    SESSION_NAME=${SESSION_NAME:-$(basename "$LADDER_CONFIG" .yaml)}
    CKPT_NAME=${CKPT_NAME:-$(basename "$LADDER_CONFIG" .yaml)}
fi

SESSIONS_HOME=${SESSIONS_HOME:-/home/$USER/experiments/vlams/control}
SESSION_NAME=${SESSION_NAME:-}
CKPT_NAME=${CKPT_NAME:-model-checkpoint-ema-latest.pt}
BATCH_TIMEOUT_MS=${BATCH_TIMEOUT_MS:-100}
# Set for a client config whose VLM roles are vlm_served.yaml (plan_verifier_local): the shared
# server then hosts the VLM too, one copy for every lane instead of ~17 GB inside each run_model.
VLM_SERVER_CONFIG=${VLM_SERVER_CONFIG:-}
# Arbitrary config overrides, e.g. EXTRA_RUN_ARGS="--runner_config.hlc_config.mode=passthrough"
read -ra EXTRA_ARGS <<< "${EXTRA_RUN_ARGS:-}"

# One lane = env_server + run_model, and lanes share a single batch_inference_server, so a card
# holds (VRAM - 7 GB for the server) / 14 GB per lane.
# Measured: 4 lanes on a 98 GB RTX PRO 6000 give 2.1x one lane's throughput at 91-100% util;
# 2 lanes fill a 46 GB A6000, and a 3rd there OOMs PhysX into silently garbage physics.
#
# A lane is a SERIAL chain of sim steps, so packing lanes is the only thing that shortens a
# campaign -- nothing makes one unit finish sooner. Measured on an H200 running 3 lanes: 62% GPU,
# 51 of 143 GB, 1.2 of 16 cores. Five sixths of the card sat idle because the cap below was 4,
# which turned a 4-hour campaign into a 56-hour one.
#
# Host RAM kills an overpacked run LOUDLY (the batch server is OOM-killed and it surfaces as a ZMQ
# timeout), and each lane drives ~1.2 cores, so the count is bounded by the CPUs the job holds. VRAM
# kills it SILENTLY, which is worse, and is why env_server checks free VRAM on every reset: a
# full card makes the RTX renderer evict the streamed USD payload geometry instead of raising, so
# the scene disappears mid-episode while physics keeps stepping and the policy is served a picture
# of the empty room. Measured on an H200 (143771 MiB), same code and units: 6 lanes x 4 envs peaks
# at 99.5% of the card and blinds 30/48 episodes, while 6 x 1 (95%) and 3 x 8 (81%) are clean -- so
# the bound is the card's occupancy, not the lane count. Checking here does NOT work: lanes boot
# minutes apart, so the card is still half empty when the first ones report ready.
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
# LANES_PER_GPU matches the other two launchers and takes precedence over LANES.
# An explicit count is taken as given: the cap bounds only the auto-derived guess, and raising it is
# exactly what a lane sweep is for.
if [ -n "${LANES_PER_GPU:-}" ]; then
    LANES=$LANES_PER_GPU
else
    LANES=${LANES:-$(( (${GPU_MIB:-21000} - 7000) / 14000 ))}
    [ "$LANES" -lt 1 ] && LANES=1
    CPU_LANES=$(( ${SLURM_CPUS_PER_TASK:-16} / 2 ))
    [ "$LANES" -gt "$CPU_LANES" ] && LANES=$CPU_LANES
    [ "$LANES" -gt 8 ] && LANES=8
fi
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-$((LANES * NUM_ENVS))}

# Kit writes a timestamped log inside its own installed package, which sits in a bazel external
# repo, so a running sim invalidates the inputs of any build starting later on the same node.
KIT_LOG_DIR=${KIT_LOG_DIR:-${TMPDIR:-/tmp}/$USER/kit_logs}

find_free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'; }

# Block until a TCP port accepts a connection, or time out (dumping the log tail).
wait_for_port() {
    local port=$1 f=$2 t=${3:-900} pid=${4:-${WAIT_PID:-$$}} t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "$t" ]; do
        [ -n "${3:-}" ] && [ -n "$f" ] && ! kill -0 "$pid" 2>/dev/null && { echo "server pid died"; tail -40 "$f" 2>/dev/null; return 1; }
        (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && { exec 3>&- 3<&- 2>/dev/null; return 0; }
        sleep 2
    done
    echo "TIMEOUT after ${t}s waiting for port $port" >&2; tail -40 "$f" 2>/dev/null; return 1
}

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

# Group the per-lane episode logs and recordings into the per-unit layout every benchmark writes:
# <ckpt>/<unit>/{results.yaml, episode_log.yaml, episodes/}. Idempotent, so it is safe to call on
# a stop signal and again on normal completion.
split_results() {
    python3 - "$CAMPAIGN_ROOT" "$CKPT_NAME" $TASKS <<'SPLIT'
import os, pathlib, shutil, sys, yaml

# The metric layer is pure stdlib, so this runs it without bazel. Tallying by hand here instead
# gave RoboLab a SECOND definition of success, and the canonical per-unit file then carried only
# a success fraction while the per-lane one carried every metric.
sys.path.insert(0, os.getcwd())
from ladder.components.inference.api.client.episode import Episode
from ladder.components.inference.api.client.metrics import aggregate

root, ckpt, units = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
key = lambda s: "".join(c for c in s.lower() if c.isalnum())
# comp_transfer_2 -> comptransfer2 ; CompTransfer2Task -> comptransfer2
by_unit = {key(pathlib.Path(u).stem): pathlib.Path(u).stem for u in units}

grouped = {}
for log in root.glob("lane*/output/*/all/episode_log.yaml"):
    for entry in yaml.safe_load(log.read_text()) or []:
        name = key(entry.get("env_name", ""))
        unit = by_unit.get(name) or by_unit.get(name[:-4] if name.endswith("task") else name)
        if not unit:
            continue
        grouped.setdefault(unit, []).append(Episode.from_log_entry(entry))

# A lane holds one Isaac app across a SLICE of tasks, so its recordings all land in one directory
# and have to be split by the task each belongs to. episode_meta.yaml carries env_name, which is the
# same key the logs are grouped by -- the alternative, matching the Nth directory to the Nth log row,
# would silently mis-assign every video the moment a lane's episodes are written out of order.
recordings = {}
for meta in root.glob("lane*/output/*/all/episodes/*/episode_meta.yaml"):
    name = key((yaml.safe_load(meta.read_text()) or {}).get("env_name", ""))
    unit = by_unit.get(name) or by_unit.get(name[:-4] if name.endswith("task") else name)
    if unit:
        recordings.setdefault(unit, []).append(meta.parent)

# Any-step AND last-step success are both reported now, so choosing between them is a reader's
# decision and COMPOSITION_STRICT_SUCCESS no longer has to fix one at write time.
for unit, episodes in grouped.items():
    out = root / ckpt / unit
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    scores = aggregate(episodes)
    (out / "results.yaml").write_text(yaml.safe_dump({episodes[0].robot_name: {unit: scores}}))
    # The episode log is the metric INPUT, so the canonical cell has to carry it too; with only
    # results.yaml there, nothing downstream can recompute a metric or add one after the fact.
    (out / "episode_log.yaml").write_text(yaml.safe_dump([e.to_log_entry() for e in episodes]))
    # Moved, not copied: an episode belongs to exactly one unit, and these are videos. The lane's
    # episode_log.yaml and episode_results.jsonl stay put for the RoboLab dashboard, which reads
    # neither the directories nor the videos.
    for source in recordings.get(unit, []):
        target = out / "episodes" / source.name
        if not target.exists():
            shutil.move(str(source), str(target))
    print(f"    {unit}: {scores['success']}")
SPLIT
}

# Published for every run (the episode log records object state from it); EXPOSE_PRIVILEGED=0 disables.
EXPOSE_PRIVILEGED=${EXPOSE_PRIVILEGED:-1}
PRIVILEGED_ARGS=()
[ "$EXPOSE_PRIVILEGED" = "1" ] && PRIVILEGED_ARGS=(--expose-privileged)
# Depth and calibration for a camera-based policy. Off by default: an extra render target.
[ "${PUBLISH_DEPTH:-False}" = "True" ] && PRIVILEGED_ARGS+=(--publish-depth)

# No camera render, for a policy that reads no image (the scripted one).
[ "${RENDER_IMAGES:-True}" = "False" ] && PRIVILEGED_ARGS+=(--no-render)

# What the policy was trained on, not what the sim likes.
IMAGE_ARGS=()
[ -n "${PUBLISH_IMAGE_SIZE:-}" ] && IMAGE_ARGS=(--image-size $PUBLISH_IMAGE_SIZE)
[ -n "${PUBLISH_EXTERIOR_CAMERAS:-}" ] && IMAGE_ARGS+=(--exterior-cameras "$PUBLISH_EXTERIOR_CAMERAS")

if [ "$INFERENCE_SERVER" = "1" ]; then
    [ -n "$SESSION_NAME" ] || { echo "SESSION_NAME required"; exit 2; }
    CKPT_PATH="$SESSIONS_HOME/$SESSION_NAME/checkpoints/$CKPT_NAME"
    [ -e "$CKPT_PATH" ] || { echo "checkpoint missing: $CKPT_PATH"; exit 2; }
fi
echo "=== tasks=[$TASKS] num_envs=$NUM_ENVS num_runs=$NUM_RUNS ==="

# Build once and launch through generated scripts, as vlam_simpler_bridge.sh does: a lane then
# runs no bazel client at all, so it cannot lose an analysis race against a sibling -- which is
# what forced lanes to come up one at a time, and made bring-up cost O(lanes x Isaac boot).
LAUNCHERS=$(mktemp -d)
# Compose with the launcher's own cleanup rather than replacing it -- a bare `trap ... EXIT` here
# silently dropped the kill of every spawned lane.
trap 'cleanup; rm -rf "$LAUNCHERS"' EXIT
TARGETS=(robots/robolab:env_server robots/robolab:export_robolab_results "$RUN_MODEL")
[ "$INFERENCE_SERVER" = "1" ] && TARGETS+=(models/vlams:batch_inference_server)
for t in "${TARGETS[@]}"; do
    script="$LAUNCHERS/${t##*:}.sh"
    # Twice: sibling eval jobs on this node share one bazel server (one output_base), and it
    # lives in whichever job's cgroup started it -- when that job ends, ours dies with
    # "Server terminated abruptly ... Socket closed". Bazel starts a fresh one on the next call.
    for attempt in 1 2; do
        bazel run --script_path="$script" "//ladder/components/inference/$t" 2>&1 | tail -3
        [ -x "$script" ] && break
        echo "=== build attempt $attempt produced no launcher for $t; retrying"
    done
    # Test the artifact, not the pipeline: `| tail` makes $? tail's, so a failed build here
    # reported success and the lanes died later on a missing file with the cause tailed away.
    [ -x "$script" ] || { echo "BUILD_FAIL: no launcher generated for $t"; exit 2; }
done

# Lane co-location hooks: SHARED_VLA_PORT connects to an already-running server (no spawn);
# VLA_PORT_PIN spawns the server on a fixed port so sibling lanes can find it.
VLA_PORT=${SHARED_VLA_PORT:-${VLA_PORT_PIN:-$(find_free_port)}}; ROBOT_PORT=$(find_free_port)

# The port override run_model needs depends on which policy class the config names.
POLICY_PORT_FLAG=--policy_config.inference_config.client_config.port

if [ "$INFERENCE_SERVER" = "0" ]; then
    echo "=== policy holds its own model: no inference server ==="
elif [ -n "${SHARED_VLA_PORT:-}" ]; then
    echo "=== sharing batch_inference_server on :$VLA_PORT ==="
    WAIT_PID=$$
    wait_for_port "$VLA_PORT" "" "$SERVER_READY_TIMEOUT" || exit 4
else
    echo "=== batch_inference_server on :$VLA_PORT ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "$LAUNCHERS/batch_inference_server.sh" \
        ${VLM_SERVER_CONFIG:+--config "$VLM_SERVER_CONFIG"} \
        --policy_config.autoclass ladder.components.inference.models.vlams.vlam_inference.VLAMInference --policy_config.ckpt_path "$CKPT_PATH" --port "$VLA_PORT" \
        --max_batch_size "$MAX_BATCH_SIZE" --batch_timeout_ms "$BATCH_TIMEOUT_MS" \
        --seed "${VLA_SEED:-987654321}" \
        > "$CAMPAIGN_ROOT/batch_inference_server.log" 2>&1 &
    PIDS+=($!); WAIT_PID=$!
    wait_for_port "$VLA_PORT" "$CAMPAIGN_ROOT/batch_inference_server.log" "$SERVER_READY_TIMEOUT" || exit 4
fi


# Each lane takes a round-robin slice of TASKS and writes its own campaign subtree; one
# run_model per lane drains that slice's (task x run) batches, bucketed by env_name.
#
# A lane owns a SLICE rather than one task, unlike the RobotArena/SimplerEnv launchers' per-cell
# work queue: one env_server holds a single Isaac app for its whole slice, and booting that app
# costs minutes. Handing lanes one task at a time would pay that boot per task. The cost is that a
# lane dealt long tasks finishes last, so pass TASKS longest-budget-first when it matters.
read -ra ALL_TASKS <<< "$TASKS"
[ "$LANES" -gt "${#ALL_TASKS[@]}" ] && LANES=${#ALL_TASKS[@]}
mkdir -p "$KIT_LOG_DIR"

# VIDEO/OUT_ROOT overrides, matching vlam_simpler_bridge.sh; unset keeps the config's own values.
REC_ARGS=()
[ -n "${VIDEO:-}" ] && REC_ARGS+=(--runner_config.recorder_config.video_enabled "${VIDEO}")

RUN_PIDS=()
LANES_UP=()
LANES_FAIL=0
LANE_PORT=()
LANE_PID=()

# Phase 1 -- every Isaac app starts at once. Their boots then overlap instead of stacking, which
# is the whole cost of bring-up: one app takes ~10 min, and serially that was ~10 min PER LANE.
for ((LANE = 0; LANE < LANES; LANE++)); do
    LANE_TASKS=""
    for ((i = LANE; i < ${#ALL_TASKS[@]}; i += LANES)); do LANE_TASKS="$LANE_TASKS ${ALL_TASKS[$i]}"; done
    LANE_ROOT="$CAMPAIGN_ROOT/lane$LANE"
    mkdir -p "$LANE_ROOT/output/$CKPT_NAME/all"
    ROBOT_PORT=$(find_free_port)

    echo "=== lane $LANE: env_server on :$ROBOT_PORT tasks=[${LANE_TASKS# }] ==="
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "$LAUNCHERS/env_server.sh" \
        --port "$ROBOT_PORT" --task $LANE_TASKS --num-envs "$NUM_ENVS" --num-runs "$NUM_RUNS" --instruction-type "$INSTRUCTION_TYPE" --headless \
        "${PRIVILEGED_ARGS[@]}" \
        "${IMAGE_ARGS[@]}" \
        --kit_args="--/log/file=$KIT_LOG_DIR/kit_${ROBOT_PORT}_$$.log" \
        > "$LANE_ROOT/env_server.log" 2>&1 &
    PIDS+=($!)
    LANE_PORT[$LANE]=$ROBOT_PORT
    LANE_PID[$LANE]=$!
    # Warm the shared Kit extension cache with the FIRST app before releasing the others: they
    # all resolve extensions against the same /scratch cache, and starting cold together makes
    # all but one die in the dependency solver. Only this one boot is serialised.
    if [ "$LANE" = 0 ] && [ "$LANES" -gt 1 ]; then
        wait_for_port "$ROBOT_PORT" "$LANE_ROOT/env_server.log" "$SERVER_READY_TIMEOUT" "$!" \
            || echo "WARN lane 0 did not come up; starting siblings against a cold cache" >&2
    fi
done

# Phase 2 -- collect them IN READY ORDER, not lane order. Isaac boots vary by minutes, and
# waiting on lane N before looking at lane N+1 left ready lanes idle behind a slow one.
# A lane that never comes up costs that lane, not the campaign.
PENDING=($(seq 0 $((LANES - 1))))
DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT))
while [ ${#PENDING[@]} -gt 0 ]; do
    STILL=()
    for LANE in "${PENDING[@]}"; do
        LANE_ROOT="$CAMPAIGN_ROOT/lane$LANE"
        ROBOT_PORT=${LANE_PORT[$LANE]}
        if ! (exec 3<>"/dev/tcp/127.0.0.1/$ROBOT_PORT") 2>/dev/null; then
            if kill -0 "${LANE_PID[$LANE]}" 2>/dev/null && [ "$SECONDS" -lt "$DEADLINE" ]; then
                STILL+=("$LANE")
            else
                echo "FAIL lane $LANE: env_server never came up; continuing without it" >&2
                tail -20 "$LANE_ROOT/env_server.log" >&2 2>/dev/null
                LANES_FAIL=$((LANES_FAIL + 1))
            fi
            continue
        fi
        exec 3>&- 3<&- 2>/dev/null
        LANE_OUT="$LANE_ROOT/output/$CKPT_NAME/all"

        echo "=== lane $LANE: run_model -> :$ROBOT_PORT  (log: $LANE_ROOT/run_model.log) ==="
        # Overriding a client port a policy does not have would fail config resolution.
        POLICY_ARGS=()
        [ "$INFERENCE_SERVER" = "1" ] && POLICY_ARGS=("$POLICY_PORT_FLAG" "$VLA_PORT")
        # These assignments have to stay on the command: a comment between them and it would end
        # the line and leave them as plain shell variables that never reach the process.
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        SESSION_NAME=$SESSION_NAME CKPT_NAME=$CKPT_NAME ENV_NAME=all \
        "$LAUNCHERS/run_model.sh" \
            --config "$LADDER_CONFIG" \
            "${POLICY_ARGS[@]}" \
            --runner_config.env_config.port "$ROBOT_PORT" \
            --runner_config.recorder_config.output_path "$LANE_OUT/episodes" \
            "${REC_ARGS[@]}" \
            "${EXTRA_ARGS[@]}" \
            > "$LANE_ROOT/run_model.log" 2>&1 &
        PIDS+=($!); RUN_PIDS+=($!); LANES_UP+=("$LANE")
    done
    PENDING=("${STILL[@]}")
    [ ${#PENDING[@]} -gt 0 ] && sleep 2
done

# Flush whatever has finished if the job is told to stop (USR1/TERM, e.g. a time limit): the
# split reads the episode logs already on disk, so a cut-short run keeps the lanes that
# completed instead of throwing the entire job away.
trap 'echo "=== walltime warning: flushing completed lanes ==="; split_results; exit 2' USR1 TERM

EVAL_RC=0
for p in "${RUN_PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || EVAL_RC=$?; }; done
trap - USR1 TERM

# RoboLab-dashboard compatibility: map each lane that ran to episode_results.jsonl.
for LANE in "${LANES_UP[@]:-}"; do
    [ -n "$LANE" ] || continue
    LANE_OUT="$CAMPAIGN_ROOT/lane$LANE/output/$CKPT_NAME"
    [ "$EVAL_RC" -eq 0 ] && "$LAUNCHERS/export_robolab_results.sh" \
        --episode-log "$LANE_OUT/all/episode_log.yaml" --output-dir "$LANE_OUT" \
        >> "$CAMPAIGN_ROOT/lane$LANE/run_model.log" 2>&1
done

# Per-unit results.yaml, in the shape SimplerEnv and RobotArena write, so all three read the same
# way. The lane logs stay where they are for the RoboLab dashboard.
split_results

echo "=== eval rc=$EVAL_RC; ${#LANES_UP[@]} of $LANES lane(s) ran under $CAMPAIGN_ROOT ==="
[ "$EVAL_RC" -ne 0 ] && { echo "--- lane${LANES_UP[0]:-0} run_model.log tail ---"; tail -40 "$CAMPAIGN_ROOT/lane${LANES_UP[0]:-0}/run_model.log"; }
# A campaign that lost a lane exits non-zero even when the lanes that ran succeeded: the job
# otherwise reports success and only a missing episode_log.yaml records the gap.
[ "$EVAL_RC" -eq 0 ] && [ "$LANES_FAIL" -gt 0 ] && EVAL_RC=1
exit $EVAL_RC
