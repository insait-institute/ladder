# ladder/core

Minimal code used throughout the entire `ladder` codebase: no ML training or deployment logic, just
the shared primitives that `ladder/components` is built on.

## Subpackages

- **`databib`** - the config/dataclass library used everywhere else in the repo for typed,
  runtime-checked configuration and data containers. See
  [databib/README.md](/ladder/core/databib/README.md).
- **`common`** - environment and platform utilities not specific to any ML workload: `git_utils`,
  `np_utils`, `path_utils`, `platform`.
- **`compute`** - numeric primitives: `checksum`
  (deterministic checksums for regression tests), `collate`, `seeder` (RNG seeding), `tensor_utils`.

## Dependency rule

`ladder/core` can depend only on external libraries, never on `ladder/components`.
Within `ladder/core` itself, code that does not use `torch` directly must not depend on code that
does (e.g. `common/path_utils.py` cannot depend on `compute/tensor_utils.py`), so that downstream
code which doesn't need `torch` isn't forced to pull it in.
