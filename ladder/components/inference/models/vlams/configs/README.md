# VLAM eval configs

Every file here is a runnable `RunnerConfig`. `test_configs_parse` parses the whole tree and fails on
anything that is not one, so a shared fragment does not belong here. The hierarchical systems live in
`api/client/hlc/configs/` for that reason. A relative `import:` resolves against the file it appears
in, so that same test fails an import that is off by a directory.

## Layout

One directory per benchmark, then one per benchmark variant (`language/`, `composition/`) within it:

```
vlam_simpler.yaml            the two shared sim clients: in-process inference, and a remote
vlam_simpler_remote.yaml     client on the shared batch server. The `simpler` is the runner they
                             configure, `SimplerEpisodeRunner`, which every sim benchmark drives:
                             RobotArena's batched_envs imports the first, RoboLab's base and the
                             composition benchmark the second. Shared, hence here and not simpler/.

hlc/                         the per-benchmark VoLo configs
<benchmark>/                 simpler/, robot_arena/ and robolab/
  <benchmark base configs>
  language/                  instruction_override probes
  composition/               the long-horizon composition benchmark
    <one config per hierarchical system, named after it>
    language/                the same probe, for composition
```

A config's name says which system it is: `_oracle`, `_plan_gripper`, `_plan_verifier`,
`_plan_verifier_interval`, `_stepwise_gripper`, `_stepwise_interval` match the file of that name in
`api/client/hlc/configs/`. `_plan_verifier_live` is the one exception and is not a system. It is
`_plan_verifier` with `cache_path` left unset, so the VLM decomposes the reset frame.

The three composition benchmarks ship the SAME set of configs, file for file: one per hierarchical
system, plus `_plan_verifier_live` and the `language/` instruction probe, so a system's number
compares across all three. Each benchmark's plan cache lives with its task package, at
`robots/<benchmark>/composition/decomp_cache.json`.
