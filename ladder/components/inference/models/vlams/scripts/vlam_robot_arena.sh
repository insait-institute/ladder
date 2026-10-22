#!/bin/bash

# Given a vlam SESSION_NAME, evaluates all session checkpoints on RobotArena scenes.
#
# Architecture: for each checkpoint we spawn one shared BatchPolicyServer
# (one model copy in GPU memory) and then run a number of "lanes" in parallel
# that all connect to it via ZMQ. A lane is one (checkpoint, scene) cell: a
# robot_arena_server (Genesis sim) plus a run_model client. The inference server
# batches requests across all lanes so each step is one forward() at
# batch = LANES_PER_GPU * envs_per_scene (137 in the unified config).
#
# LANES_PER_GPU defaults to 2 (fits a 48 GB GPU). A ~140 GB GPU fits LANES_PER_GPU=6.

# Example usage:
#
# CKPT_NAMES="model-checkpoint-ema-030000.pt model-checkpoint-030000.pt" \
# SESSION_NAME=sess_<date>_<tag> \
# SCENES="default1 scene2 scene4" \
# LANES_PER_GPU=2 \
# ./ladder/components/inference/models/vlams/scripts/vlam_robot_arena.sh
#
# RA_SERVER_CONFIG/RA_CLIENT_CONFIG/SCENES also drive the long-horizon composition benchmark, whose
# tasks are listed in robots/robot_arena/README.md; RUN_LABEL names the output subdirectory.

set -u

if [ -z "${SESSION_NAME}" ]; then
  echo "Error: SESSION_NAME not defined. Exiting"
  exit 1
fi

echo "SESSION_NAME: ${SESSION_NAME}"

LANES_PER_GPU=${LANES_PER_GPU:-2}
CHUNK_SIZE=${CHUNK_SIZE:-4}
BATCH_TIMEOUT_MS=${BATCH_TIMEOUT_MS:-20}
INFERENCE_SERVER_READY_TIMEOUT=${INFERENCE_SERVER_READY_TIMEOUT:-600}
RA_SERVER_READY_TIMEOUT=${RA_SERVER_READY_TIMEOUT:-300}
# A model that brings its own pip environment provides its own run_model target.
RUN_MODEL=${RUN_MODEL:-models/vlams:run_model}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
POLICY_ENV_NAME=${POLICY_ENV_NAME:-bridge_orig}

SESSIONS_HOME=${SESSIONS_HOME:-/home/$USER/experiments/vlams/control}
# Results land under CAMPAIGN_ROOT (default: the session directory). EVAL_OUT is an alias of it.
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-${EVAL_OUT:-$SESSIONS_HOME/$SESSION_NAME}}
RA_SERVER_CONFIG=${RA_SERVER_CONFIG:-ladder/components/inference/robots/robot_arena/configs/widowx_batched_unified.yaml}
# Set for a client config whose VLM roles are vlm_served.yaml (plan_verifier_local): the shared
# server then hosts the VLM too, one copy for every lane instead of ~17 GB inside each run_model.
VLM_SERVER_CONFIG=${VLM_SERVER_CONFIG:-}
RA_CLIENT_CONFIG=${RA_CLIENT_CONFIG:-ladder/components/inference/models/vlams/configs/robot_arena/language/vlam_robot_arena_unified.yaml}
# With EVAL_OUT set there is no RUN_LABEL level: results go to <CAMPAIGN_ROOT>/<ckpt>/<scene>.
if [ -n "${EVAL_OUT:-}" ]; then RUN_LABEL=${RUN_LABEL:-}; else RUN_LABEL=${RUN_LABEL:-robot_arena}; fi

# The scripted completability policy carries no weights, so a run has no checkpoint and needs no
# inference server; SESSION_NAME then only labels the campaign directory.
# NO_VLA=1 forces the no-checkpoint, nothing-served path for a policy the top-level config only
# reaches through an import chain, which a grep here cannot see (models/capx is one).
SCRIPTED=${NO_VLA:-$(grep -qc "scripted_policy.ScriptedPolicy" "$RA_CLIENT_CONFIG" 2>/dev/null && echo 1 || echo 0)}

if [ "$SCRIPTED" = "1" ]; then
    ALL_CKPT_NAMES=(scripted)
else
    ALL_CKPT_NAMES=($(find "$SESSIONS_HOME/$SESSION_NAME/checkpoints/" \
        \( -type f -o -type l \) \
        -name "model-checkpoint-*.pt" \
        | grep -E "model-checkpoint-([0-9]+|ema-[0-9]+)\.pt" \
        | xargs -n1 basename))
fi

# scene98 is excluded: its thin "green blob" distractor NaN-crashes the Genesis
# rigid solver (thin-object / interpenetration contact blowup) regardless of
# welding state. scene12/41/93 (present only in the wider 74-scene archive, not
# in this set) NaN-crash the same way and are likewise omitted.
ALL_SCENES=(
    default1 default2 default3 default4
    scene2 scene4 scene7 scene8 scene15 scene17 scene19 scene22 scene24
    scene27 scene28 scene31 scene32 scene34 scene38 scene39 scene40
    scene47 scene53 scene54 scene60 scene70 scene71 scene75 scene76
    scene77 scene80 scene82 scene97
)

if [ -z "${CKPT_NAMES+x}" ]; then
    CKPT_NAMES=("${ALL_CKPT_NAMES[@]}")
elif [ -n "${CKPT_NAMES}" ] && [ -z "${CKPT_NAMES##* *}" ]; then
    read -ra CKPT_NAMES <<< "$CKPT_NAMES"
fi

if [ -z "${SCENES+x}" ]; then
    SCENES=("${ALL_SCENES[@]}")
elif [ -n "${SCENES}" ] && [ -z "${SCENES##* *}" ]; then
    read -ra SCENES <<< "$SCENES"
fi

# Auto-detect ENVS_PER_SCENE from the server YAML: it counts the object_variation_offsets_m
# entries (137 in the unified 16+121 config).
if [ -z "${ENVS_PER_SCENE:-}" ]; then
    ENVS_PER_SCENE=$(grep -cE '^\s+- \[' "$RA_SERVER_CONFIG" 2>/dev/null || echo 0)
    [ "$ENVS_PER_SCENE" -gt 0 ] || { echo "FATAL: ENVS_PER_SCENE could not be detected from $RA_SERVER_CONFIG; set it explicitly" >&2; exit 2; }
fi
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-$((LANES_PER_GPU * ENVS_PER_SCENE))}

mkdir -p "$CAMPAIGN_ROOT" || { echo "FATAL: cannot create CAMPAIGN_ROOT=$CAMPAIGN_ROOT" >&2; exit 2; }

# Function to find a free port
find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'
}

# Block until a substring appears in a log file, or time out and dump the tail.
wait_for_log() {
    local f=$1 p=$2 t=${3:-300} t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "$t" ]; do
        grep -qF "$p" "$f" 2>/dev/null && return 0
        sleep 1
    done
    echo "TIMEOUT after ${t}s waiting for '$p' in $f" >&2
    tail -30 "$f" >&2 2>/dev/null || true
    return 1
}

# Run one (ckpt, scene) cell: spawn a per-scene robot_arena_server, then the run_model client
# that connects to it AND to the shared batch_inference_server on $vla_port.
run_cell() {
    local ckpt=$1 scene=$2 vla_port=$3
    local out="$CAMPAIGN_ROOT${RUN_LABEL:+/$RUN_LABEL}/$ckpt/$scene"
    mkdir -p "$out"
    local ra_port; ra_port=$(find_free_port)

    # --test_config.scene_names expects a Python list literal
    local scene_arg="[\"$scene\"]"

    # Always published: the scripted policy solves from it and the episode log records object state from it.
    local privileged_args=(--env_config.expose_privileged True)
    # Depth and calibration for a camera-based policy. Off by default: an extra render target.
    privileged_args+=(--env_config.publish_depth "${PUBLISH_DEPTH:-False}")
    # False renders no image, for a policy that reads none (the scripted one).
    privileged_args+=(--env_config.render_images "${RENDER_IMAGES:-True}")

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    bazel run //ladder/components/inference/robots/robot_arena:robot_arena_server -- \
        --config "$RA_SERVER_CONFIG" \
        "${privileged_args[@]}" \
        --test_config.scene_names "$scene_arg" \
        --test_config.batch_size_per_scene "$ENVS_PER_SCENE" \
        --request_listener_config.req_rep_server_config.port "$ra_port" \
        > "$out/ra_server.log" 2>&1 &
    local ra_pid=$!
    if ! wait_for_log "$out/ra_server.log" "Starting ZMQ request reply server" "$RA_SERVER_READY_TIMEOUT"; then
        kill "$ra_pid" 2>/dev/null
        echo "FAIL ra_server: $ckpt / $scene" >&2
        return 1
    fi

    # Arbitrary config overrides, e.g. EXTRA_RUN_ARGS="--runner_config.hlc_config.mode=passthrough"
    read -ra extra_args <<< "${EXTRA_RUN_ARGS:-}"


    local policy_args=(
        --policy_config.env_name "$POLICY_ENV_NAME"
        --policy_config.controller_config.chunk_size "$CHUNK_SIZE"
        --policy_config.inference_config.client_config.port "$vla_port"
    )
    [ "$SCRIPTED" = "1" ] && policy_args=()

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    SESSION_NAME=$SESSION_NAME CKPT_NAME=$ckpt ENV_NAME="$scene" \
    bazel run //ladder/components/inference/$RUN_MODEL -- \
        --config "$RA_CLIENT_CONFIG" \
        "${policy_args[@]}" \
        --runner_config.env_config.port "$ra_port" \
        --runner_config.recorder_config.output_path "$out/episodes" \
        "${extra_args[@]}" \
        > "$out/client.log" 2>&1
    local rc=$?
    kill "$ra_pid" 2>/dev/null
    fuser -k "$ra_port/tcp" 2>/dev/null
    return "$rc"
}

# Build once
bazel build \
    //ladder/components/inference/models/vlams:batch_inference_server \
    //ladder/components/inference/robots/robot_arena:robot_arena_server \
    //ladder/components/inference/$RUN_MODEL 2>&1 | tail -5

CELLS_OK=0
CELLS_FAIL=0

for CKPT in "${CKPT_NAMES[@]}"; do
    OUT_CKPT="$CAMPAIGN_ROOT${RUN_LABEL:+/$RUN_LABEL}/$CKPT"
    mkdir -p "$OUT_CKPT"
    VLA_PORT=0
    SERVER_PID=""
    if [ "$SCRIPTED" = "0" ]; then
        CKPT_PATH="$SESSIONS_HOME/$SESSION_NAME/checkpoints/$CKPT"
        [ -e "$CKPT_PATH" ] || { echo "WARN: checkpoint $CKPT_PATH missing; skipping" >&2; continue; }

        VLA_PORT=$(find_free_port)
        VLA_LOG="$OUT_CKPT/batch_inference_server.log"
        echo "=== checkpoint=$CKPT  spawning shared inference server on port $VLA_PORT ==="

        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        bazel run //ladder/components/inference/models/vlams:batch_inference_server -- \
            ${VLM_SERVER_CONFIG:+--config "$VLM_SERVER_CONFIG"} \
            --policy_config.autoclass ladder.components.inference.models.vlams.vlam_inference.VLAMInference --policy_config.ckpt_path "$CKPT_PATH" \
            --port "$VLA_PORT" \
            --max_batch_size "$MAX_BATCH_SIZE" \
            --batch_timeout_ms "$BATCH_TIMEOUT_MS" \
            --seed "${VLA_SEED:-987654321}" \
            > "$VLA_LOG" 2>&1 &
        SERVER_PID=$!

        if ! wait_for_log "$VLA_LOG" "Batch inference server running" "$INFERENCE_SERVER_READY_TIMEOUT"; then
            kill "$SERVER_PID" 2>/dev/null
            fuser -k "$VLA_PORT/tcp" 2>/dev/null
            echo "FAIL: inference server did not come up for $CKPT; skipping" >&2
            CELLS_FAIL=$((CELLS_FAIL + ${#SCENES[@]}))
            continue
        fi
        echo "  inference server ready (pid=$SERVER_PID port=$VLA_PORT)"
    fi

    CELL_PIDS=()
    for SCENE in "${SCENES[@]}"; do
        run_cell "$CKPT" "$SCENE" "$VLA_PORT" &
        CELL_PIDS+=($!)
        if [ "${#CELL_PIDS[@]}" -ge "$LANES_PER_GPU" ]; then
            # Reap whichever cell finished first: waiting on CELL_PIDS[0] idles lanes behind one long
            # task, and composition step budgets span 4x (100 per goal).
            if wait -n "${CELL_PIDS[@]}"; then CELLS_OK=$((CELLS_OK + 1)); else CELLS_FAIL=$((CELLS_FAIL + 1)); fi
            for i in "${!CELL_PIDS[@]}"; do
                kill -0 "${CELL_PIDS[i]}" 2>/dev/null || { unset 'CELL_PIDS[i]'; CELL_PIDS=("${CELL_PIDS[@]}"); break; }
            done
        fi
    done
    for p in "${CELL_PIDS[@]}"; do
        if wait "$p"; then CELLS_OK=$((CELLS_OK + 1)); else CELLS_FAIL=$((CELLS_FAIL + 1)); fi
    done

    # graceful shutdown: SIGTERM lets the server exit its poll loop, flush its final
    # batch-count log and close the ZMQ socket cleanly; escalate to SIGKILL only if it
    # is still alive after a 5 s grace
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$SERVER_PID" 2>/dev/null || break   # exited cleanly - stop waiting
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true        # still alive after grace - force it
        fuser -k "$VLA_PORT/tcp" 2>/dev/null || true
    fi
    echo "=== checkpoint=$CKPT done ==="
done

echo "=== all checkpoints done: ok=$CELLS_OK fail=$CELLS_FAIL ==="
[ "$CELLS_FAIL" -gt 0 ] && exit 1
exit 0
