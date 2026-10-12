#!/bin/bash

# Script that given a vlam SESSION_NAME, evaluates all session checkpoints on all SIMPLER envs

if [ -z "${SESSION_NAME}" ]; then
  echo "Error: SESSION_NAME not defined. Exiting"
  exit 1
fi

SESSIONS_HOME="${SESSIONS_HOME:-/home/$USER/experiments/vlams/control}"
SESSION_PATH=$SESSIONS_HOME/$SESSION_NAME

echo "SESSION_NAME: ${SESSION_NAME}"
echo "SESSION_PATH: ${SESSION_PATH}"

REPO_ROOT="$(realpath "$(dirname "$0")/../../../../../../")"
cd $REPO_ROOT

ENV_VARS=(SESSION_NAME="$SESSION_NAME")
# Pass CKPT_NAMES only if they are defined
if [ -n "${CKPT_NAMES:-}" ]; then
    ENV_VARS+=(CKPT_NAMES="$CKPT_NAMES")
fi

# Maybe run bridge eval
if grep -q 'bridge' "$SESSION_PATH/full_config.yaml"; then

    env "${ENV_VARS[@]}" \
        ENV_NAMES="widowx_put_carrot_on_plate widowx_put_spoon_on_table_cloth" \
        ./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh &

    env "${ENV_VARS[@]}" \
        ENV_NAMES="widowx_stack_green_cube_on_yellow_cube widowx_put_eggplant_in_basket" \
        ./ladder/components/inference/models/vlams/scripts/vlam_simpler_bridge.sh &

    # Wait for background jobs to complete
    wait
fi

# TODO: Maybe run fractal eval
# if grep -q 'fractal' "$SESSION_PATH/full_config.yaml"; then
# fi
