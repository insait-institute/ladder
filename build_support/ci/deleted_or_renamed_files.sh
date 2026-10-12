#!/bin/bash

if [ "$#" -eq 0 ]; then
  PULL_BASE_SHA=origin/main
else
  PULL_BASE_SHA=$1
fi

COMMIT_RANGE=${COMMIT_RANGE:-$(git merge-base "${PULL_BASE_SHA}" HEAD)}
DELETED_OR_RENAMED_FILES=$(git diff --diff-filter=DR --name-status "${COMMIT_RANGE}" | cut -f 2)

echo "${DELETED_OR_RENAMED_FILES}"
