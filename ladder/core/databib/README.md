Databib is a Python library that provides the following high-level features:
- `Dataclass` functionality with type-checks, class member functions for automated recursive manipulation, etc. Built on top of native python `dataclasses`.
- `Config` and `Configurable` classes with built-in type-checks, file parsing, autoconfiguration, etc
- `AutoConfig` and `AutoConfigurable` classes for autoconfiguration at runtime

It is vendored into this monorepo at `//ladder/core/databib` and is imported as `ladder.core.databib`:

```python
from ladder.core.databib.config import Config, Configurable
from ladder.core.databib.dataclasses import Dataclass, dataclass
```
