from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
    overload,
)

import tyro
import tyro.conf as conf
from tyro.constructors import ConstructorRegistry

OutT = TypeVar("OutT")
CallableType = TypeVar("CallableType", bound=Callable)
constructor_registry = ConstructorRegistry()
union_struct_registry: dict[str, tuple[list, Any]] = {}


def registry(name: str):
    if name not in union_struct_registry:
        raise ValueError(f"No union struct with name {name} exists.")

    callable_or_type = union_struct_registry[name][1]

    return callable_or_type.__tyro__markers__


def make_union_registry(
    name: str, choices: dict[str, CallableType] | None = None
) -> Callable[[CallableType], CallableType]:
    if name in union_struct_registry:
        raise ValueError(f"struct registry {name} alredy exists.")

    def _inner(callable: CallableType) -> CallableType:
        union_struct_registry[name] = ([], callable)

        if choices is not None:
            for command, calltype in choices.items():
                calltype = conf.configure(conf.subcommand(name=command))(calltype)

                union_struct_registry[name][0].append(calltype)

        callable.__tyro_markers__ = (  # type: ignore
            conf.arg(
                constructor_factory=lambda: Union[*union_struct_registry[name][0]]  # type: ignore
            ),
        )

        return callable

    return _inner


def union_struct_choice(
    registry: str, *, enforce_type: bool = True, command: str | None = None
) -> Callable[[CallableType], CallableType]:
    if registry not in union_struct_registry:
        raise ValueError(
            f"registry '{registry}' does not exist. Use cli.union_registry to create it."
        )

    def _inner(callable: CallableType) -> CallableType:
        _registry = union_struct_registry[registry]
        base_type = _registry[1]

        # if `base_type` is a Protocol, we skip the check due to limited issubclass support
        # e.g. protocols with non-method members do not support issubclass()
        is_protocol = getattr(base_type, "_is_protocol", False)
        check_type = enforce_type and not is_protocol

        if check_type and not issubclass(callable, base_type):
            raise ValueError(
                f"type {callable} cannot be registered to struct with base type {base_type}"
            )

        if command is not None:
            callable = conf.configure(conf.subcommand(name=command))(callable)

        _registry[0].insert(0, callable)

        return callable

    return _inner


@overload
def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    prog: str | None = None,
    args: None | Sequence[str] = None,
) -> OutT: ...


@overload
def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    return_unknown_args: Literal[True],
    prog: str | None = None,
    args: None | Sequence[str] = None,
) -> tuple[OutT, list[str]]: ...


def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    return_unknown_args: bool = False,
    prog: str | None = None,
    args: None | Sequence[str] = None,
) -> OutT | tuple[OutT, list[str]]:
    return tyro.cli(
        f,
        prog=prog,
        args=args,
        use_underscores=True,
        console_outputs=console_outputs,
        compact_help=False,
        config=(
            conf.CascadeSubcommandArgs,
            conf.FlagConversionOff,
            conf.SuppressFixed,
        ),
        return_unknown_args=return_unknown_args,
        registry=constructor_registry,
    )
