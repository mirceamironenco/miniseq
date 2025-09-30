from dataclasses import Field, asdict, dataclass, is_dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from typing_extensions import override


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def config_as_dict(config: DataclassInstance, /, **kw_overrides: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise ValueError(f"Expected config to be dataclass, got {type(config)}.")

    fields = asdict(config)

    if "_target" in fields:
        fields.pop("_target")

    for key, value in kw_overrides.items():
        if key not in fields:
            raise KeyError(f"Unexpected override key {key} not a valid field.")

        fields[key] = value

    # Eliminate None-valued items to allow class constructor defaults.
    fields = {k: v for (k, v) in fields.items() if v is not None}

    return fields


OutT_cov = TypeVar("OutT_cov", covariant=True)


@runtime_checkable
class BuilderProtocol(Protocol[OutT_cov]):
    def build(self, **kwd_override: Any) -> OutT_cov: ...


OutT = TypeVar("OutT")


class BuilderConfig(Generic[OutT]):
    """Config class used to instantiate objects of type `_target`."""

    _target: type[OutT]

    def build(self, **kwd_overrides: Any) -> OutT:
        """Override if provided valid options. Custom constructors should override this."""

        if kwd_overrides:
            for key, value in kwd_overrides.items():
                if value is None:
                    continue

                if hasattr(self, key):
                    setattr(self, key, value)

        # Pass the entire config.
        return self._target(self)  # type: ignore[call-arg]


@dataclass
class DataclassBuilderConfig(BuilderConfig[OutT]):
    _target: type[OutT]

    def build_with_kwds(self, **kwd_overrides: Any) -> OutT:
        fields = self.get_fields(**kwd_overrides)

        return self._target(**fields)

    @override
    def build(self, **kwd_overrides: Any) -> OutT:
        return self.build_with_kwds(**kwd_overrides)

    def get_fields(self, **kwd_overrides: Any) -> dict[str, Any]:
        return config_as_dict(self, **kwd_overrides)
