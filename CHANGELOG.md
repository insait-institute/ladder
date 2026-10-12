# Changelog

Every release of this repository is listed here, newest first. Releases follow
[semantic versioning](https://semver.org): a change to the benchmark itself (task set, scene
layouts, success predicates, step budgets) bumps the major version, so that scores reported against
different major versions are never compared.

## [1.0.0] - 2026-10-22

First public release.

- The three benchmark suites: RoboLab-Long (44 tasks), SimplerEnv-Long (54 tasks) and
  RobotArena-&infin;-Long (47 tasks), with their programmatic per-subgoal success predicates.
- Ten baseline methods: the orchestrator and its configurations, PSP-Agent, and the
  adapters for VoLo, CaP-X, TiPToP and GPT-Policy.
- The low-level policies: the Bridge policy ([`INSAIT-Institute/LADDER-Bridge-Policy`](https://huggingface.co/INSAIT-Institute/LADDER-Bridge-Policy))
  and pi0.5 (DROID, from openpi).
- The baseline rollouts: every episode's log and video in
  [`INSAIT-Institute/LADDER-Evaluation-Episodes`](https://huggingface.co/datasets/INSAIT-Institute/LADDER-Evaluation-Episodes), browsable with the eval dashboard.
- Your own system: a policy served from any Python environment, an orchestrator or an agent plugs
  into `scripts/eval.sh` with one short file (`examples/`).
- Leaderboard submissions by pull request (`leaderboard/`), scored from the episode logs exactly as
  the paper's systems are (`scripts/leaderboard.py`).
- The files the build downloads (simulation assets, model weights and third-party sources) are
  attached to this release, each archive with its original license. The two weight files are
  covered by their projects' licenses: `m2t2.pth` by the NVIDIA License (non-commercial, in
  `m2t2_src.tar.gz`) and `sam2.1_hiera_large.pt` by Apache-2.0 (in `sam2_src.tar.gz`).

[1.0.0]: https://github.com/insait-institute/ladder/releases/tag/v1.0.0
