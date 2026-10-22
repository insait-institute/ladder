import functools
import inspect
import pprint
from typing import Any, Type, TypeVar

from ladder.core.databib.dataclasses.dataclass_utils import parametrize

from .config import (
    AutoRecordsManager,
    Config,
    ConfigAttributeContext,
    ConfigMeta,
    ConfigT,
    MissingConfigAttributeError,
)


class AutoConfigMeta(ConfigMeta):
    @functools.lru_cache(maxsize=None, typed=True)  # noqa: B019
    def __getitem__(cls, config_type) -> type:  # noqa: N805
        """AutoConfig[ModelConfig]: an AutoConfig that only creates subclasses of ModelConfig."""
        if cls is not AutoConfig:
            raise TypeError(f"Only AutoConfig can be subscripted, not {cls.__name__}")
        # Prevent things like AutoConfig[TypeVar] or AutoConfig[Any]
        if not inspect.isclass(config_type):
            raise TypeError(f"{cls.__name__} expects a config class, but got {config_type}")
        return parametrize(cls, (config_type,), ConfigT=config_type)

    def __instancecheck__(cls, maybe_config):  # noqa: N805
        """
        Override instance checks so that AutoConfig type annotations in classes work properly
            isinstance(ConfigA(), AutoConfig[ConfigA]) -> True
            isinstance(ConfigB(), AutoConfig[ConfigA]) -> False
            isinstance(ConfigA(), AutoConfig) -> True
        """
        if not isinstance(cls.ConfigT, TypeVar):
            return isinstance(maybe_config, cls.ConfigT)
        if isinstance(maybe_config, Config):
            return True
        return super().__instancecheck__(maybe_config)


class AutoConfig(Config, metaclass=AutoConfigMeta):
    """
    A generic config class that will be instantiated as a specific config class returned by __new__.
    Selecting the correct config class requires setting the correct class path to a corresponding
    Configurable class in the autoclass attribute, e.g. databib.config.config.Configurable. The returned
    config class is the one to which `ConfigT` attribute points to. Note that only Configurable classes
    whose config class is concrete (not a TypeVar) can be specified.

    Example:
        class ModelConfig(Config):
            ...

        class Model(Configurable[ModelConfig]):
            ...

        model_config = AutoConfig(autoclass="full.path.to.module.Model")
        assert isinstance(model_config, ModelConfig)  # True
        assert model_config.autoclass == "full.path.to.module.Model"  # True
        model = AutoConfiguralbe(model_config)
        assert isinstance(model, Model)  # True
    """

    ConfigT = ConfigT  # a TypeVar until the class is subscripted, e.g. AutoConfig[ModelConfig]

    def __new__(cls, *args, **kwargs):
        ConfigType = cls._get_actual_config_class(kwargs)
        # Make sure the AutoConfig[...] argument is respected
        if not isinstance(cls.ConfigT, TypeVar):
            if not issubclass(ConfigType, cls.ConfigT):
                raise TypeError(
                    f"Class {cls.__name__} can only create config instances, which are "
                    f"subclasses of {cls.ConfigT}, but instance of {ConfigType} requested"
                )
        return ConfigType(*args, **kwargs)

    @classmethod
    def _get_actual_config_class(cls, data: dict) -> Type[Config]:
        key = "autoclass"

        with ConfigAttributeContext(cls, key):
            if key not in data:
                raise MissingConfigAttributeError(f"Field '{key}' of {cls.__name__} was not provided")

            class_path = data[key]
            if class_path == "":
                raise ValueError(
                    f"Attribute {key} missing for {cls.__name__}. Availble data:\n{pprint.pformat(data, indent=2)}"
                )

            ConfigType = AutoRecordsManager.config_cls_from_configurable_path(class_path)

        return ConfigType  # type: ignore[return-value]

    @classmethod
    def _ignore_redundant_keys(cls, data: dict) -> dict:
        # Make sure from_json works with AutoConfig
        ConfigType = cls._get_actual_config_class(data)
        return ConfigType._ignore_redundant_keys(data)  # pylint: disable=protected-access  # noqa: SLF001


class AutoConfigurable:
    """
    A generic configurable class that will be instantiated as a specific instance of Configurable returned by __new__.
    Selecting the correct config class requires passing a config instance with autoclass attribute set to
    a full class path, e.g. databib.config.config.Configurable. Note that only Configurable classes whose config
    class is concrete (not a TypeVar) can be specified. Note that only Configurable classes will be created

    Example:
        class ModelConfig(Config):
            ...

        class Model(Configurable[ModelConfig]):
            ...

        model_config = AutoConfig(autoclass="full.path.to.module.Model")
        assert isinstance(model_config, ModelConfig)  # True
        assert model_config.autoclass == "full.path.to.module.Model"  # True
        model = AutoConfiguralbe(model_config)
        assert isinstance(model, Model)  # True
    """

    def __new__(cls, config, **kwargs):
        ConfigurableT = AutoConfigurable.target_class_from_config(config)
        return ConfigurableT(config=config, **kwargs)

    @staticmethod
    def target_class_from_config(config: Config) -> Type:
        if not isinstance(config, Config):
            raise TypeError(
                f"{AutoConfigurable.__name__} expects an instance of Config, but got {type(config)}"
            )

        return AutoRecordsManager.configurable_cls_from_config(config)  # type: ignore[arg-type]

    def __getattr__(self, _name: str) -> Any:
        """Define to help pyright"""

    def __call__(self, *_args, **_kwargs) -> Any:
        """Define to help pyright"""
        return None
