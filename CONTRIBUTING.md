# Contributing

Thanks for your interest in LADDER. Bug reports, questions and pull requests are welcome.

## Reporting a problem

Open an [issue](https://github.com/insait-institute/ladder/issues) with the command you ran, the suite and system, and the end of the
log. The launchers print the path of every log they write. For a wrong score, attach the task's
`results.yaml` and, if you can, one episode video.

## Pull requests

- Keep a pull request to one change, and describe what it fixes or adds.
- Run the checks CI runs before you push:
  ```bash
  bazel test //ladder/... --test_tag_filters=-needs-gpu
  ```
- Do not change the benchmark itself (tasks, scene layouts, success predicates, step budgets) in a
  pull request that does something else. Such a change needs its own pull request, and it bumps the
  major version (see `CHANGELOG.md`).

## Adding a system

To evaluate your own policy, orchestrator or agent, start from [`examples/`](examples/README.md). It
needs no change to this repository. To put it on the leaderboard, follow
[`leaderboard/README.md`](leaderboard/README.md): a submission is a pull request that adds one
directory under `leaderboard/submissions/` and nothing else.

To add a system to the code itself, add a config file and, if it brings its own code, a package under
`ladder/components/inference/models/`. [EVALUATION.md](EVALUATION.md#launchers-and-configs) shows how
the existing systems are wired, and `ladder/components/inference/README.md` describes the client and
server interfaces.

## Releasing (maintainers)

1. Add the new version's section at the top of `CHANGELOG.md`, and set the same `version` and
   `date-released` in `CITATION.cff`.
2. Merge that to `main`, then tag it and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "LADDER vX.Y.Z" && git push origin vX.Y.Z
   ```
3. The `release` workflow checks that the tag matches `CITATION.cff` and `CHANGELOG.md`, then
   publishes the GitHub Release with that changelog section and the source archive (plus its
   sha256).

Files that the build downloads (simulation assets) are attached to their own release (`assets-v1`)
and pinned by checksum. They are never replaced in place: a changed file gets a new tag, and the
pins in `http-assets.MODULE.bazel` move to it in a pull request.
