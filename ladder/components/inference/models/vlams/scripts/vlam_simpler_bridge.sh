#!/bin/bash

# Script that given a vlam SESSION_NAME, evaluates all session checkpoints on widowx SIMPLER envs

# Example usage:
#
# CKPT_NAMES="model-checkpoint-ema-030000.pt model-checkpoint-030000.pt" \
# SESSION_NAME=sess_<date>_<tag> \
# ENV_NAMES="widowx_put_carrot_on_plate widowx_put_spoon_on_table_cloth" \
# ./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh
#
# LANES_PER_GPU>1 runs that many envs at once on one GPU against the shared inference server, as in
# vlam_robot_arena.sh; 1 (the default) is one env at a time. SIM_CONFIG/ENV_NAMES also drive the
# long-horizon composition benchmark (see robots/simpler/README.md).
#
# Sessions are read from SESSIONS_HOME (default /home/$USER/experiments/vlams/control). OUT_ROOT, when
# set, is the output directory: <OUT_ROOT>/<ckpt>/<env>/ holds results.yaml and the recordings.

if [ -z "${SESSION_NAME}" ]; then
  echo "Error: SESSION_NAME not defined. Exiting"
  exit 1
fi

echo "SESSION_NAME: ${SESSION_NAME}"

CHUNK_SIZE=${CHUNK_SIZE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SIMPLER_CONFIG=${SIMPLER_CONFIG:-ladder/components/inference/models/vlams/configs/vlam_simpler.yaml}
# Arbitrary config overrides, as the RobotArena and RoboLab launchers take them.
read -ra EXTRA_ARGS <<< "${EXTRA_RUN_ARGS:-}"

# A model that brings its own pip environment provides its own run_model target.
RUN_MODEL=${RUN_MODEL:-models/vlams:run_model}
SESSIONS_HOME="${SESSIONS_HOME:-/home/$USER/experiments/vlams/control}"
LANES_PER_GPU=${LANES_PER_GPU:-1}              # concurrent envs per GPU, one shared inference server
# Set for a config whose VLM roles are vlm_served.yaml (plan_verifier_local): the shared server then
# hosts the VLM too, one copy for every lane instead of ~17 GB inside each run_model.
VLM_SERVER_CONFIG=${VLM_SERVER_CONFIG:-}
# EVAL_OUT is an alias of OUT_ROOT.
OUT_ROOT=${OUT_ROOT:-${EVAL_OUT:-}}
CELL_ROOT=${OUT_ROOT:-/tmp/vlam_simpler_$USER}  # per-cell logs, used only when LANES_PER_GPU > 1

# The scripted completability policy carries no weights, so a run has no checkpoint and no
# controller; SESSION_NAME then only labels the output. uses_remote_client already keeps the shared
# inference server out of the way, since the config names no remote client.
# NO_VLA=1 forces the no-checkpoint, nothing-served path for a policy the top-level config only
# reaches through an import chain, which a grep here cannot see (models/capx is one).
SCRIPTED=${NO_VLA:-$(grep -qc "scripted_policy.ScriptedPolicy" "$SIMPLER_CONFIG" 2>/dev/null && echo 1 || echo 0)}

# 24 = the full standard object-variation range in one batch. A completability gate needs one episode
# to finish and the scripted policy is deterministic, so extra episodes only re-measure the layout.
N_ENVS=${N_ENVS:-$([ "$SCRIPTED" = "1" ] && echo 1 || echo 24)}  # parallel ManiSkill3 envs per scene

if [ "$SCRIPTED" = "1" ]; then
    ALL_CKPT_NAMES=(scripted)
else
    ALL_CKPT_NAMES=($(find "$SESSIONS_HOME/$SESSION_NAME/checkpoints/" \
        \( -type f -o -type l \) \
        -name "model-checkpoint-*.pt" \
        | grep -E "model-checkpoint-([0-9]+|ema-[0-9]+)\.pt" \
        | xargs -n1 basename))
fi

ALL_ENV_NAMES=(
    widowx_put_carrot_on_plate
    widowx_put_spoon_on_table_cloth
    widowx_stack_green_cube_on_yellow_cube
    widowx_put_eggplant_in_basket
)

# Parse CKPT_NAMES
if [ -z "${CKPT_NAMES+x}" ]; then
    # CKPT_NAMES is not set, use all checkpoint names
    CKPT_NAMES=("${ALL_CKPT_NAMES[@]}")
elif [ -n "${CKPT_NAMES}" ] && [ -z "${CKPT_NAMES##* *}" ]; then
    # CKPT_NAMES is a string with spaces, convert to array
    read -ra CKPT_NAMES <<< "$CKPT_NAMES"
fi

# Parse ENV_NAMES
if [ -z "${ENV_NAMES+x}" ]; then
    # ENV_NAMES is not set, use all checkpoint names
    ENV_NAMES=("${ALL_ENV_NAMES[@]}")
elif [ -n "${ENV_NAMES}" ] && [ -z "${ENV_NAMES##* *}" ]; then
    # ENV_NAMES is a string with spaces, convert to array
    read -ra ENV_NAMES <<< "$ENV_NAMES"
fi

# Function to find a free port
find_free_port() {
    local port
    while true; do
        port=$((RANDOM + 10000))
        if ! nc -z localhost $port 2>/dev/null; then
            echo $port
            break
        fi
    done
}

# Build once, and launch through generated scripts rather than `bazel run`, so no lane runs a bazel
# client at all: a lane cannot then lose an analysis/build race against a sibling, and the run
# holds one bazel invocation instead of 3*LANES. `--script_path` writes the launch command out instead
# of running it, ending the script in "$@", so per-lane arguments still work:
# https://bazel.build/reference/command-line-reference#param-script-path
LAUNCHERS=$(mktemp -d)
trap 'rm -rf "$LAUNCHERS"' EXIT
bazel run --script_path="$LAUNCHERS/simpler_server.sh" \
    //ladder/components/inference/robots/simpler:simpler_server
bazel run --script_path="$LAUNCHERS/run_model.sh" \
    //ladder/components/inference/$RUN_MODEL
bazel run --script_path="$LAUNCHERS/batch_inference_server.sh" \
    //ladder/components/inference/models/vlams:batch_inference_server

# Does this config, or anything it imports, select the batched REMOTE inference client? That client
# needs a shared batch policy server to serve the checkpoint over ZMQ; the in-process default
# (vlam_simpler.yaml) loads the ckpt itself and needs no server. Every composition config reaches the
# remote client through a chain of imports, so matching only the top-level file misses it and the run
# hangs against a server that was never started. Imports resolve relative to the file that CONTAINS
# them, hence the per-file dirname.
uses_remote_client() {
    local config=$1 dir imported
    [ -f "$config" ] || return 1
    grep -q "RemoteInferenceClient" "$config" && return 0
    dir=$(dirname "$config")
    while read -r imported; do
        uses_remote_client "$dir/${imported%\'}" && return 0
    done < <(sed -n "s/^[[:space:]]*import:[[:space:]]*'\{0,1\}//p" "$config")
    return 1
}

NEEDS_VLA_SERVER=0
uses_remote_client "${SIMPLER_CONFIG}" && NEEDS_VLA_SERVER=1
# A policy that runs no VLA still imports the remote base for its runner config, so
# uses_remote_client sees it; NO_VLA=1 says there is nothing to serve.
[ "${NO_VLA:-0}" = "1" ] && NEEDS_VLA_SERVER=0

# Block until log file $1 contains literal $2, or $3 seconds elapse.
# Ready when the marker appears, failed as soon as the process is gone. Both servers log their marker
# AFTER binding the socket, so the marker proves that THIS process owns the port -- a plain port probe
# also goes true when the sibling that stole the port bound it. Passing the pid matters because a
# server that dies on bind would otherwise cost the whole timeout before the retry.
wait_for_log() {
    local f=$1 pat=$2 timeout_s=$3 pid=${4:-} t=0
    while [ "$t" -lt "$timeout_s" ]; do
        [ -f "$f" ] && grep -qF "$pat" "$f" && return 0
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "pid $pid exited after ${t}s without logging '$pat'" >&2
            tail -30 "$f" >&2 2>/dev/null || true
            return 1
        fi
        sleep 1; t=$((t + 1))
    done
    echo "TIMEOUT after ${t}s waiting for '$pat' in $f" >&2
    tail -30 "$f" >&2 2>/dev/null || true
    return 1
}

# One (checkpoint, env) cell. Reads CKPT_NAME/ENV_NAME/VLA_PORT from the loop below: each call is
# backgrounded, so the subshell snapshots them. With LANES_PER_GPU>1 the cell logs to its own file,
# since concurrent consoles interleave.
run_cell() {
    local PORT LOG=/dev/stdout SERVER_PID CLIENT_PID attempt
    PORT=$(find_free_port)
    if [ "$LANES_PER_GPU" -gt 1 ]; then
        LOG=$CELL_ROOT/$CKPT_NAME/$ENV_NAME/cell.log; mkdir -p "$(dirname "$LOG")"
    else
        echo "Using port: $PORT"
    fi

    # Sim config: by default the per-scene config/<ENV_NAME>.yaml (the 4 original bridge scenes,
    # unchanged). When SIM_CONFIG is set (e.g. the shared composition config
    # widowx_composition.yaml) it is used instead and ENV_NAME is passed as the scene id. This is
    # how composition scenes run through the SAME launcher.
    if [ -n "${SIM_CONFIG:-}" ]; then
        SIM_ARGS=(--config "${SIM_CONFIG}" --env_config.env_name "${ENV_NAME}-v0")
    else
        SIM_ARGS=(--config "ladder/components/inference/robots/simpler/configs/${ENV_NAME}.yaml")
    fi
    # Always published: the scripted policy solves from it and the episode log records object state from it.
    SIM_ARGS+=(--env_config.expose_privileged True)
    # Depth and calibration for a camera-based policy. Off by default: an extra render target.
    SIM_ARGS+=(--env_config.publish_depth "${PUBLISH_DEPTH:-False}")

    # A port that probes free can be taken between the probe and the bind -- by a sibling lane, or by
    # another job sharing the node, both routine now that lanes really run concurrently. The loser
    # dies with "Address already in use" and its scene vanishes silently: the cell returns, the job
    # still exits 0, and only a missing results.yaml records that the scene ever existed. So wait for
    # the readiness marker (always, not just for concurrent lanes) and retry on a fresh port.
    for attempt in 1 2 3; do
        [ "$LANES_PER_GPU" -gt 1 ] || set -x
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        "$LAUNCHERS/simpler_server.sh" \
        "${SIM_ARGS[@]}" \
        --env_config.n_envs ${N_ENVS} \
        --request_listener_config.req_rep_server_config.port ${PORT} \
        > "$LOG" 2>&1 &
        set +x
        SERVER_PID=$!
        wait_for_log "$LOG" "Starting ZMQ request reply server" "${SIM_READY_TIMEOUT:-600}" "$SERVER_PID" && break
        kill "$SERVER_PID" 2>/dev/null
        if [ "$attempt" = 3 ]; then
            echo "FAIL: sim server never came up for $ENV_NAME after $attempt attempts" >&2
            return 1
        fi
        echo "sim server failed on port $PORT (attempt $attempt); retrying on a fresh port" >&2
        PORT=$(find_free_port)
    done

    # Optional recorder overrides. Unset -> use SIMPLER_CONFIG defaults, so the 4 original scenes
    # are unchanged. VIDEO=True records a video per episode; OUT_ROOT sets a per-scene output dir.
    REC_ARGS=()
    [ -n "${VIDEO:-}" ] && REC_ARGS+=(--runner_config.recorder_config.video_enabled "${VIDEO}")
    [ -n "${OUT_ROOT:-}" ] && REC_ARGS+=(--runner_config.recorder_config.output_path "${OUT_ROOT}/${CKPT_NAME}/${ENV_NAME}/rec")

    # Point the remote client at this checkpoint's shared batch server (no-op for in-process).
    VLA_ARGS=()
    [ -n "$VLA_PORT" ] && VLA_ARGS+=(--policy_config.inference_config.client_config.port "${VLA_PORT}")

    POLICY_ARGS=(
        --policy_config.env_name bridge_orig
        --policy_config.controller_config.chunk_size "${CHUNK_SIZE}"
    )
    [ "$SCRIPTED" = "1" ] && POLICY_ARGS=()

    # Run model client
    [ "$LANES_PER_GPU" -gt 1 ] || set -x
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    CKPT_NAME=$CKPT_NAME \
    SESSION_NAME=$SESSION_NAME \
    ENV_NAME=$ENV_NAME \
    "$LAUNCHERS/run_model.sh" \
    --config "${SIMPLER_CONFIG}" \
    --runner_config.env_config.port ${PORT} \
    "${POLICY_ARGS[@]}" \
    "${VLA_ARGS[@]}" \
    "${REC_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG" 2>&1 &
    set +x
    CLIENT_PID=$!

    # Wait for both processes to complete
    wait $SERVER_PID $CLIENT_PID

    # Clean up any remaining processes on the port
    fuser -k ${PORT}/tcp 2>/dev/null || true
}

SKIPPED_CKPTS=0
for CKPT_NAME in "${CKPT_NAMES[@]}";
do
    # Shared batch inference server for this checkpoint (remote/batched configs only). One GPU model
    # copy serves every scene's run_model client sequentially over ZMQ; the in-process path skips this.
    VLA_PORT=""
    VLA_SERVER_PID=""
    if [ "${NEEDS_VLA_SERVER}" = "1" ]; then
        VLA_PORT=$(find_free_port)
        CKPT_PATH="$SESSIONS_HOME/$SESSION_NAME/checkpoints/$CKPT_NAME"
        VLA_LOG="${OUT_ROOT:-/tmp}/${CKPT_NAME}/batch_inference_server.log"
        mkdir -p "$(dirname "$VLA_LOG")"
        echo "=== checkpoint=$CKPT_NAME  spawning shared inference server on port $VLA_PORT ==="
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        "$LAUNCHERS/batch_inference_server.sh" \
            ${VLM_SERVER_CONFIG:+--config "$VLM_SERVER_CONFIG"} \
            --policy_config.autoclass ladder.components.inference.models.vlams.vlam_inference.VLAMInference --policy_config.ckpt_path "$CKPT_PATH" \
            --port "$VLA_PORT" \
            --seed "${VLA_SEED:-987654321}" \
            > "$VLA_LOG" 2>&1 &
        VLA_SERVER_PID=$!
        # Several jobs starting together each read a large checkpoint off shared storage, which can
        # take well over ten minutes. A checkpoint skipped here costs the whole job, so the timeout is
        # generous and the skip is recorded.
        if ! wait_for_log "$VLA_LOG" "Batch inference server running" "${INFERENCE_SERVER_READY_TIMEOUT:-2400}" \
                "$VLA_SERVER_PID"; then
            kill "$VLA_SERVER_PID" 2>/dev/null
            fuser -k "$VLA_PORT/tcp" 2>/dev/null || true
            echo "FAIL: inference server did not come up for $CKPT_NAME; skipping checkpoint" >&2
            SKIPPED_CKPTS=$((SKIPPED_CKPTS + 1))
            continue
        fi
        echo "  inference server ready (pid=$VLA_SERVER_PID port=$VLA_PORT)"
    fi

    # Keep LANES_PER_GPU cells in flight; at 1 lane this starts and waits for one at a time.
    CELL_PIDS=()
    for ENV_NAME in "${ENV_NAMES[@]}"; do
        run_cell &
        CELL_PIDS+=($!)
        # Wait for whichever cell finishes first, not the oldest: composition step budgets span 10x
        # (100 per goal), so waiting in order would idle lanes behind one long env.
        if [ "${#CELL_PIDS[@]}" -ge "$LANES_PER_GPU" ]; then
            wait -n "${CELL_PIDS[@]}"
            CELL_PIDS=($(for p in "${CELL_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && printf '%s ' "$p"; done))
        fi
    done
    [ "${#CELL_PIDS[@]}" -gt 0 ] && wait "${CELL_PIDS[@]}"

    # Tear down this checkpoint's shared batch server.
    if [ -n "$VLA_SERVER_PID" ]; then
        kill "$VLA_SERVER_PID" 2>/dev/null || true
        [ -n "$VLA_PORT" ] && fuser -k "$VLA_PORT/tcp" 2>/dev/null || true
    fi
done

# Exiting 0 after skipping every checkpoint would report success with no results, so fail instead.
if [ "$SKIPPED_CKPTS" -gt 0 ]; then
    echo "FAIL: $SKIPPED_CKPTS of ${#CKPT_NAMES[@]} checkpoint(s) never started an inference server" >&2
    exit 1
fi

# Cells run backgrounded and are reaped with `wait -n`, so no per-cell status survives to here.
# Check instead that every (checkpoint, env) pair left a non-empty results.yaml: that is the run's
# whole output, and a missing one catches every way a cell can vanish -- a crashed client, a sim server
# that never bound, an env name with no registered scene.
if [ -n "${OUT_ROOT:-}" ]; then
    MISSING=()
    for CKPT_NAME in "${CKPT_NAMES[@]}"; do
        for ENV_NAME in "${ENV_NAMES[@]}"; do
            [ -s "$OUT_ROOT/$CKPT_NAME/$ENV_NAME/results.yaml" ] || MISSING+=("$CKPT_NAME/$ENV_NAME")
        done
    done
    if [ "${#MISSING[@]}" -gt 0 ]; then
        echo "FAIL: ${#MISSING[@]} cell(s) produced no results: ${MISSING[*]}" >&2
        exit 1
    fi
fi
