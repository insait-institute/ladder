#!/bin/bash

if [ "$#" -eq 0 ]; then
  PULL_BASE_SHA=origin/main
else
  PULL_BASE_SHA=$1
fi

GIT_DIFF_FILTER=$2

COMMIT_RANGE=${COMMIT_RANGE:-$(git merge-base "${PULL_BASE_SHA}" HEAD)}
CHANGED_FILES=$(git diff --diff-filter="${GIT_DIFF_FILTER}" --name-only "${COMMIT_RANGE}")

echo "${CHANGED_FILES}"
