# Submitting to the leaderboard

The [leaderboard](https://ladderrobotics.ai/#leaderboard) takes any policy, orchestrator or
agent. An entry is a pull request that adds one directory under `submissions/`. CI checks it and
recomputes every number from the episode logs. When it is merged, the page updates by itself.

## 1. Evaluate every task

Plug your system in ([`examples/README.md`](../examples/README.md)) and run each suite with
`--task all`. Keep the default number of episodes:

```bash
for suite in simplerenv_long robotarena_long robolab_long; do
  scripts/eval.sh --suite $suite --task all --policy-server localhost:8000 --name my_policy
done
```

An entry may cover one, two or all three suites. It is ranked overall only with all three. On
another suite it shows "&ndash;".

## 2. Package the results

```bash
scripts/leaderboard.py package --name my_policy --kind policy   # policy | orchestrator | agent
```

This writes `leaderboard/submissions/my_policy/` with two files:

- `episodes.jsonl`: one line per episode (suite, task, success, subgoals completed, steps). It is
  read from your evaluation's episode logs, never typed in.
- `submission.yaml`: who you are and what the system is. Fill in every field. The command prints
  what is still missing, and the scores the leaderboard will show.

Upload the whole results directory (`out/`, with the episode logs and videos) somewhere public,
such as a Hugging Face dataset, and put its link in `results_url`. Anyone can then replay your
episodes in the dashboard and check them.

## 3. Open a pull request

Check the entry locally, then open a pull request that adds only your directory:

```bash
scripts/leaderboard.py check leaderboard/submissions/my_policy
```

CI runs the same check. Once the pull request is merged, the entry appears on the leaderboard as
**self-reported**. The maintainers mark it **verified** once they have re-run the system with the
same code and seen the same scores within their intervals. That needs a public implementation
(`code_url`) or access to it.

## Rules

- **Every episode counts.** Report all episodes of every task of a suite, with the default episode
  count. Do not re-run failures or pick the best of several runs. `check` rejects a suite with a
  missing task or a wrong episode count.
- **No privileged information.** A plugged-in system only sees what a real robot would see
  ([`examples/README.md`](../examples/README.md#what-every-system-sees)). A system that reads the
  simulator's state must set `privileged: true`. It is then listed as a reference, like the paper's
  privileged verifier, and is not ranked.
- **Say what was tuned on LADDER.** Set `tuned_on_ladder: true` if any part of the system (weights,
  prompts, thresholds) was trained or tuned on LADDER's tasks or scenes, and describe it in
  `description`. Per-task prompts or settings are not allowed.
- **Pin the version.** `ladder_commit` is the commit you evaluated with. The scores of different
  benchmark versions (`CHANGELOG.md`) are not comparable.
- **One entry per system.** Several checkpoints or prompt variants of one system count as one entry.
  Submit the one you would report.
- **Keep secrets out.** Episode logs record the instructions a system issued. Remove API keys and
  anything private before uploading.

## How the numbers are computed

Exactly as in the paper (`scripts/leaderboard.py`, function `score`), so a submitted system and a
paper baseline are compared on equal terms:

- **Task success.** The share of episodes in which every subgoal was completed. Each suite pools
  its episodes. The ± is the larger side of the 95% Wilson interval.
- **All suites.** The mean of the three suite rates, so each suite weighs the same.
- **Subgoal progress.** The share of subgoals completed at the end of the episodes. A later leg of
  the same object (Relay, Revisit) counts only if it was completed after the earlier legs.
- **By horizon and by task category.** Task success grouped by the subgoal count and the category in
  [`tasks.yaml`](tasks.yaml), the paper's task catalog.

`scripts/leaderboard.py build` writes the page's data (`docs/static/data/submissions.json`) from
every submission. The Pages workflow runs it on every merge, so do not edit that file by hand.
