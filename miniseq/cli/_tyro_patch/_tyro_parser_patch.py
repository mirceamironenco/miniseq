from __future__ import annotations

import dataclasses
import pathlib
import sys
import warnings
from collections import deque
from collections.abc import Sequence
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import shtab
import tyro.conf as conf
from tyro import _argparse as argparse
from tyro import (
    _argparse_formatter,
    _arguments,
    _calling,
    _docstrings,
    _fields,
    _resolver,
    _singleton,
    _strings,
    _unsafe_cache,
)
from tyro._parsers import (
    ParserSpecification,
    SubparsersSpecification,
    add_subparsers_to_leaves,
    handle_field,
)
from tyro._typing import TypeForm
from tyro.conf import _markers
from tyro.constructors import ConstructorRegistry
from tyro.constructors._primitive_spec import UnsupportedTypeAnnotationError
from tyro.constructors._struct_spec import UnsupportedStructTypeMessage

from miniseq.cli._tyro_patch._help_formatter import TyroFlatSubcommandHelpFormatter

# TODO: Document that the patch only works if all subcomands actually have a default!


class DSU(_singleton.Singleton):
    _parent: dict[str, str]
    _size: dict[str, int]

    def init(self, *args, **kwds) -> None:
        self._parent = {}
        self._size = {}
        self._components = 0

    def is_node(self, node: str) -> bool:
        return node in self._parent

    def make(self, node: str) -> bool:
        if not self.is_node(node):
            self._parent[node] = node
            self._size[node] = 1
            self._components += 1

            return True

        return False

    def find(self, node: str) -> str:
        if not self.is_node(node):
            self.make(node)
            return node

        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]

        return self._parent[node]

    def union(self, node_x: str, node_y: str) -> bool:
        x, y = self.find(node_x), self.find(node_y)

        if x == y:
            return False

        if self._size[x] < self._size[y]:
            x, y = y, x

        self._parent[y] = x
        self._size[x] += self._size[y]
        self._components -= 1
        return True

    def connected(self, node_x: str, node_y: str) -> bool:
        return self.find(node_x) == self.find(node_y)

    def num_components(self) -> int:
        return self._components

    def component_size(self, node_x: str) -> int:
        return self._size[self.find(node_x)]


T = TypeVar("T")


class _Node:
    DEFAULT_PARSER_NAME = "<#_#>"
    default_name: str
    edges: list[_Node]
    is_subparser: bool

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        is_subparser: bool = False,
    ) -> None:
        if is_subparser:
            if name is None:
                raise ValueError("Subparsers must have default name provided.")
        self.default_name = name if name else _Node.DEFAULT_PARSER_NAME
        self.edges = []
        self.is_subparser = is_subparser

    def add_edge(self, other: _Node) -> None:
        self.edges.append(other)


@dataclasses.dataclass(frozen=True)
class _ParserSpecification(ParserSpecification):
    _node: _Node


# Patch ParserSpecification.from_callable to retain default subcommand names
def from_callable_or_type(
    f: Callable[..., T],
    markers: Set[_markers._Marker],
    description: str | None,
    parent_classes: Set[Type[Any]],
    default_instance: Union[
        T, _singleton.PropagatingMissingType, _singleton.NonpropagatingMissingType
    ],
    intern_prefix: str,
    extern_prefix: str,
    add_help: bool,
    subcommand_prefix: str = "",
    support_single_arg_types: bool = False,
) -> ParserSpecification:
    """Create a parser definition from a callable or type."""

    # Consolidate subcommand types.
    markers = markers | set(_resolver.unwrap_annotated(f, _markers._Marker)[1])
    consolidate_subcommand_args = _markers.ConsolidateSubcommandArgs in markers

    # Cycle detection.
    #
    # - 'parent' here refers to in the nesting hierarchy, not the superclass.
    # - We threshold by `max_nesting_depth` to suppress false positives,
    #  for example from custom constructors that behave differently
    #  depending the default value. (example: ml_collections.ConfigDict)
    max_nesting_depth = 128
    if (
        f in parent_classes
        and f is not dict
        and intern_prefix.count(".") > max_nesting_depth
    ):
        raise UnsupportedTypeAnnotationError(
            f"Found a cyclic dependency with type {f}."
        )

    # TODO: we are abusing the (minor) distinctions between types, classes, and
    # callables throughout the code. This is mostly for legacy reasons, could be
    # cleaned up.
    parent_classes = parent_classes | {cast(Type, f)}

    # Resolve the type of `f`, generate a field list.
    with _fields.FieldDefinition.marker_context(tuple(markers)):
        out = _fields.field_list_from_type_or_callable(
            f=f,
            default_instance=default_instance,
            support_single_arg_types=support_single_arg_types,
        )
        assert not isinstance(out, UnsupportedStructTypeMessage), out
        f, field_list = out

    has_required_args = False
    args = []
    helptext_from_intern_prefixed_field_name: Dict[str, str | None] = {}

    child_from_prefix: Dict[str, ParserSpecification] = {}

    subparsers = None
    subparsers_from_prefix = {}

    ### Patch
    node = _Node()
    subcommand_dsu = DSU()
    ### Patch

    for field in field_list:
        field_out = handle_field(
            field,
            parent_classes=parent_classes,
            intern_prefix=intern_prefix,
            extern_prefix=extern_prefix,
            subcommand_prefix=subcommand_prefix,
            add_help=add_help,
        )
        if isinstance(field_out, _arguments.ArgumentDefinition):
            # Handle single arguments.
            args.append(field_out)
            if field_out.lowered.required:
                has_required_args = True
            continue

        # From https://github.com/brentyi/tyro/pull/227
        # TODO: Decide if we keep and implement. Doesn't work at the moment?
        # class_field_name = _strings.make_field_name([intern_prefix, field.intern_name])
        # if field.helptext is not None:
        #     helptext_from_intern_prefixed_field_name[class_field_name] = field.helptext

        if isinstance(field_out, SubparsersSpecification):
            # Handle subparsers.
            subparsers_from_prefix[field_out.intern_prefix] = field_out
            subparsers = add_subparsers_to_leaves(subparsers, field_out)

            ### Patch
            # TODO: This is something to test out optional subcommands
            # which don't have = None default. If the = None default exists,
            # then the current code will already add 'command:None' to defaults.
            # if field_out.default_name is None:
            #     assert field_out.default_parser is None

            #     # TODO: Unify this and below into a single function call!
            #     # TODO: Propose field_out_hooks?
            #     none_default = None
            #     for _pname, _parser in field_out.parser_from_name.items():
            #         if _parser.f is none_proxy:
            #             subcommand_dsu.make(_pname)
            #             if _pname not in subcommand_default_names:
            #                 subcommand_default_names[_pname] = field_out
            #                 none_default = _pname
            #                 break

            #     if none_default is not None:
            #         for choice in field_out.parser_from_name.keys():
            #             subcommand_dsu.union(none_default, choice)

            #     continue

            if field_out.default_name is not None:
                assert field_out.default_name in field_out.parser_from_name

                subcommand_dsu.make(field_out.default_name)

                subparser_node = _Node(name=field_out.default_name, is_subparser=True)
                node.add_edge(subparser_node)

                for choice, choice_parser in field_out.parser_from_name.items():
                    subcommand_dsu.union(field_out.default_name, choice)
                    if isinstance(choice_parser, _ParserSpecification):
                        choice_parser._node.default_name = choice
                        subparser_node.add_edge(choice_parser._node)
            ### Patch

        elif isinstance(field_out, ParserSpecification):
            # Handle nested parsers.
            nested_parser = field_out
            child_from_prefix[field_out.intern_prefix] = nested_parser

            if nested_parser.has_required_args:
                has_required_args = True

            # Include nested subparsers.
            if nested_parser.subparsers is not None:
                subparsers_from_prefix.update(
                    nested_parser.subparsers_from_intern_prefix
                )
                subparsers = add_subparsers_to_leaves(
                    subparsers, nested_parser.subparsers
                )

            # Helptext for this field; used as description for grouping arguments.
            class_field_name = _strings.make_field_name(
                [intern_prefix, field.intern_name]
            )
            if field.helptext is not None:
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    field.helptext
                )
            else:
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    _docstrings.get_callable_description(nested_parser.f)
                )

            # If arguments are in an optional group, it indicates that the default_instance
            # will be used if none of the arguments are passed in.
            if (
                len(nested_parser.args) >= 1
                and _markers._OPTIONAL_GROUP in nested_parser.args[0].field.markers
            ):
                current_helptext = helptext_from_intern_prefixed_field_name[
                    class_field_name
                ]
                helptext_from_intern_prefixed_field_name[class_field_name] = (
                    ("" if current_helptext is None else current_helptext + "\n\n")
                    + "Default: "
                    + str(field.default)
                )

            ### Patch
            if isinstance(field_out, _ParserSpecification):
                node.add_edge(field_out._node)
            ### Patch

    return _ParserSpecification(
        f=f,
        markers=markers,
        description=_strings.remove_single_line_breaks(
            description
            if description is not None
            else _docstrings.get_callable_description(f)
        ),
        args=args,
        field_list=field_list,
        child_from_prefix=child_from_prefix,
        helptext_from_intern_prefixed_field_name=helptext_from_intern_prefixed_field_name,
        subparsers=subparsers,
        subparsers_from_intern_prefix=subparsers_from_prefix,
        intern_prefix=intern_prefix,
        extern_prefix=extern_prefix,
        has_required_args=has_required_args,
        consolidate_subcommand_args=consolidate_subcommand_args,
        add_help=add_help,
        _node=node,
    )


# TODO: Possible move run_default_cli/custom calls and below registries.
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


### CUSTOM COMMANDS USING BELOW CLI ###


@overload
def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    args: None | Sequence[str] = None,
) -> OutT: ...


@overload
def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    return_unknown_args: Literal[True],
    args: None | Sequence[str] = None,
) -> tuple[OutT, list[str]]: ...


def run_default_cli(
    f: type[OutT] | Callable[..., OutT],
    *,
    console_outputs: bool = True,
    return_unknown_args: bool = False,
    args: None | Sequence[str] = None,
) -> OutT | tuple[OutT, list[str]]:
    return cli(
        f,
        args=args,
        use_underscores=True,
        console_outputs=console_outputs,
        config=(
            conf.ConsolidateSubcommandArgs,
            conf.FlagConversionOff,
            conf.SuppressFixed,
        ),
        return_unknown_args=return_unknown_args,
        registry=constructor_registry,
        apply_subcomm_defaults=True,
    )


### CUSTOM COMMANDS USING BELOW CLI ###


@overload
def subcommand_cli_from_dict(
    subcommands: Dict[str, Callable[..., T]],
    *,
    prog: str | None = None,
    description: str | None = None,
    args: Sequence[str] | None = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: Sequence[conf._markers.Marker] | None = None,
    sort_subcommands: bool = False,
    registry: ConstructorRegistry | None = None,
    apply_subcomm_defaults: bool = False,
) -> T: ...


# TODO: hack. We prefer the above signature, which Pyright understands, but as of 1.6.1
# mypy doesn't reason about the generics properly.
@overload
def subcommand_cli_from_dict(
    subcommands: Dict[str, Callable[..., Any]],
    *,
    prog: str | None = None,
    description: str | None = None,
    args: Sequence[str] | None = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: Sequence[conf._markers.Marker] | None = None,
    sort_subcommands: bool = False,
    registry: ConstructorRegistry | None = None,
    apply_subcomm_defaults: bool = False,
) -> Any: ...


def subcommand_cli_from_dict(
    subcommands: Dict[str, Callable[..., Any]],
    *,
    prog: str | None = None,
    description: str | None = None,
    args: Sequence[str] | None = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: Sequence[conf._markers.Marker] | None = None,
    sort_subcommands: bool = False,
    registry: ConstructorRegistry | None = None,
    apply_subcomm_defaults: bool = False,
) -> Any:
    """Generate a subcommand CLI from a dictionary of functions.

    For an input like:

    .. code-block:: python

        tyro.extras.subcommand_cli_from_dict(
            {
                "checkout": checkout,
                "commit": commit,
            }
        )

    This is internally accomplished by generating and calling:

    .. code-block:: python

        from typing import Annotated, Any, Union
        import tyro

        tyro.cli(
            Union[
                Annotated[
                    Any,
                    tyro.conf.subcommand(name="checkout", constructor=checkout),
                ],
                Annotated[
                    Any,
                    tyro.conf.subcommand(name="commit", constructor=commit),
                ],
            ]
        )

    Args:
        subcommands: Dictionary that maps the subcommand name to function to call.
        prog: The name of the program printed in helptext. Mirrors argument from
            :py:class:`argparse.ArgumentParser`.
        description: Description text for the parser, displayed when the --help flag is
            passed in. If not specified, `f`'s docstring is used. Mirrors argument from
            :py:class:`argparse.ArgumentParser`.
        args: If set, parse arguments from a sequence of strings instead of the
            commandline. Mirrors argument from :py:meth:`argparse.ArgumentParser.parse_args()`.
        use_underscores: If True, use underscores as a word delimeter instead of hyphens.
            This primarily impacts helptext; underscores and hyphens are treated equivalently
            when parsing happens. We default helptext to hyphens to follow the GNU style guide.
            https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
        console_outputs: If set to ``False``, parsing errors and help messages will be
            supressed. This can be useful for distributed settings, where :func:`tyro.cli()`
            is called from multiple workers but we only want console outputs from the
            main one.
        config: Sequence of config marker objects, from :mod:`tyro.conf`.
        registry: A :class:`tyro.constructors.ConstructorRegistry` instance containing custom
            constructor rules.
    """

    keys = list(subcommands.keys())
    if sort_subcommands:
        keys = sorted(keys)

    # We need to form a union type, which requires at least two elements.
    return cli(
        Union[  # type: ignore
            tuple(
                [
                    Annotated[
                        # The constructor function can return any object.
                        Any,
                        # We'll instantiate this object by invoking a subcommand with
                        # name k, via a constructor.
                        conf.subcommand(name=k, constructor=subcommands[k]),
                    ]
                    for k in keys
                ]
                # Union types need at least two types. To support the case
                # where we only pass one subcommand in, we'll pad with `None`
                # but suppress it.
                + [Annotated[None, conf._markers.Suppress]]
            )
        ],
        prog=prog,
        description=description,
        args=args,
        use_underscores=use_underscores,
        console_outputs=console_outputs,
        config=config,
        registry=registry,
        apply_subcomm_defaults=apply_subcomm_defaults,
    )


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
) -> OutT: ...


@overload
def cli(
    f: TypeForm[OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
) -> tuple[OutT, list[str]]: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: None = None,
    return_unknown_args: Literal[False] = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
) -> OutT: ...


@overload
def cli(
    f: Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    # Passing a default makes sense for things like dataclasses, but are not
    # supported for general callables. These can, however, be specified in the
    # signature of the callable itself.
    default: None = None,
    return_unknown_args: Literal[True],
    use_underscores: bool = False,
    console_outputs: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
) -> tuple[OutT, list[str]]: ...


def cli(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str = None,
    args: None | Sequence[str] = None,
    default: None | OutT = None,
    return_unknown_args: bool = False,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
    **deprecated_kwargs,
) -> OutT | tuple[OutT, list[str]]:
    """Generate a command-line interface from type annotations and populate the target with arguments.

    :func:`cli()` is the core function of tyro. It takes a type-annotated function or class
    and automatically generates a command-line interface to populate it from user arguments.

    Two main usage patterns are supported:

    1. With a function (CLI arguments become function parameters):

       .. code-block:: python

          import tyro

          def main(a: str, b: str) -> None:
              print(a, b)

          if __name__ == "__main__":
              tyro.cli(main)  # Parses CLI args, calls main() with them

    2. With a class (CLI arguments become object attributes):

       .. code-block:: python

          from dataclasses import dataclass
          from pathlib import Path

          import tyro

          @dataclass
          class Config:
              a: str
              b: str

          if __name__ == "__main__":
              config = tyro.cli(Config)  # Parses CLI args, returns populated AppConfig
              print(f"Config: {config}")

    Args:
        f: The function or type to populate from command-line arguments. This must have
            type-annotated inputs for tyro to work correctly.
        prog: The name of the program to display in the help text. If not specified, the
            script filename is used. This mirrors the argument from
            :py:class:`argparse.ArgumentParser()`.
        description: The description text shown at the top of the help output. If not
            specified, the docstring of `f` is used. This mirrors the argument from
            :py:class:`argparse.ArgumentParser()`.
        args: If provided, parse arguments from this sequence of strings instead of
            the command line. This is useful for testing or programmatic usage. This mirrors
            the argument from :py:meth:`argparse.ArgumentParser.parse_args()`.
        default: An instance to use for default values. This is only supported if ``f`` is a
            type like a dataclass or dictionary, not if ``f`` is a general callable like a
            function. This is useful for merging CLI arguments with values loaded from
            elsewhere, such as a config file.
        return_unknown_args: If True, returns a tuple of the output and a list of unknown
            arguments that weren't consumed by the parser. This mirrors the behavior of
            :py:meth:`argparse.ArgumentParser.parse_known_args()`.
        use_underscores: If True, uses underscores as word delimiters in the help text
            instead of hyphens. This only affects the displayed help; both underscores and
            hyphens are treated equivalently during parsing. The default (False) follows the
            GNU style guide for command-line options.
            https://www.gnu.org/software/libc/manual/html_node/Argument-Syntax.html
        console_outputs: If set to False, suppresses parsing errors and help messages.
            This is useful in distributed settings where tyro.cli() is called from multiple
            workers but console output is only desired from the main process.
        config: A sequence of configuration marker objects from :mod:`tyro.conf`. This
            allows applying markers globally instead of annotating individual fields.
            For example: ``tyro.cli(Config, config=(tyro.conf.PositionalRequiredArgs,))``
        registry: A :class:`tyro.constructors.ConstructorRegistry` instance containing custom
            constructor rules.

    Returns:
        If ``f`` is a type (like a dataclass), returns an instance of that type populated
        with values from the command line. If ``f`` is a function, calls the function with
        arguments from the command line and returns its result. If ``return_unknown_args``
        is True, returns a tuple of the result and a list of unused command-line arguments.
    """

    # Make sure we start on a clean slate. Some tests may fail without this due to
    # memory address conflicts.
    _unsafe_cache.clear_cache()

    with _strings.delimeter_context("_" if use_underscores else "-"):
        output = _cli_impl(
            f,
            prog=prog,
            description=description,
            args=args,
            default=default,
            return_parser=False,
            return_unknown_args=return_unknown_args,
            use_underscores=use_underscores,
            console_outputs=console_outputs,
            add_help=add_help,
            config=config,
            registry=registry,
            apply_subcomm_defaults=apply_subcomm_defaults,
            **deprecated_kwargs,
        )

    # Prevent unnecessary memory usage.
    _unsafe_cache.clear_cache()

    if return_unknown_args:
        assert isinstance(output, tuple)
        run_with_args_from_cli = output[0]
        return run_with_args_from_cli(), output[1]
    else:
        run_with_args_from_cli = cast(Callable[[], OutT], output)
        return run_with_args_from_cli()


def get_parser(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    # We have no `args` argument, since this is only used when
    # parser.parse_args() is called.
    prog: None | str = None,
    description: None | str = None,
    default: None | OutT = None,
    use_underscores: bool = False,
    console_outputs: bool = True,
    add_help: bool = True,
    config: None | Sequence[conf._markers.Marker] = None,
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
) -> argparse.ArgumentParser:
    """Get the ``argparse.ArgumentParser`` object generated under-the-hood by
    :func:`tyro.cli()`. Useful for tools like ``sphinx-argparse``, ``argcomplete``, etc.

    For tab completion, we recommend using :func:`tyro.cli()`'s built-in
    ``--tyro-write-completion`` flag."""
    with _strings.delimeter_context("_" if use_underscores else "-"):
        return cast(
            argparse.ArgumentParser,
            _cli_impl(
                f,
                prog=prog,
                description=description,
                args=None,
                default=default,
                return_parser=True,
                return_unknown_args=False,
                use_underscores=use_underscores,
                console_outputs=console_outputs,
                add_help=add_help,
                config=config,
                registry=registry,
                apply_subcomm_defaults=apply_subcomm_defaults,
            ),
        )


def _cli_impl(
    f: TypeForm[OutT] | Callable[..., OutT],
    *,
    prog: None | str = None,
    description: None | str,
    args: None | Sequence[str],
    default: None | OutT,
    return_parser: bool,
    return_unknown_args: bool,
    console_outputs: bool,
    add_help: bool,
    config: None | Sequence[conf._markers.Marker],
    registry: None | ConstructorRegistry = None,
    apply_subcomm_defaults: bool = False,
    **deprecated_kwargs,
) -> (
    OutT
    | argparse.ArgumentParser
    | tuple[
        Callable[[], OutT],
        list[str],
    ]
):
    """Helper for stitching the `tyro` pipeline together."""

    # assert args is None
    # args = list(sys.argv[1:])
    args = list(sys.argv[1:]) if args is None else list(args)

    # TODO: not sure if this still works well in all cases.
    # TODO: Remove this if you figure out how to show choices box.
    # Note: If the user passes --choices and apply_subcomm_defaults is False
    # this is will error out since it's treated as if --choices was a valid cli arg.
    if apply_subcomm_defaults and "--choices" in args:
        # if len(args) > 1:
        #     raise ValueError("For displaying choices, only --choices is allowed.")

        args = ["--help"]
        apply_subcomm_defaults = False

        if config is not None:
            config = tuple(
                _mark for _mark in config if _mark != conf.ConsolidateSubcommandArgs
            )

            if not config:
                config = None

    if config is not None:
        f = Annotated[(f, *config)]  # type: ignore

    if "default_instance" in deprecated_kwargs:
        warnings.warn(
            "`default_instance=` is deprecated! use `default=` instead.", stacklevel=2
        )
        default = deprecated_kwargs["default_instance"]
    if deprecated_kwargs.get("avoid_subparsers", False):
        f = conf.AvoidSubcommands[f]  # type: ignore
        warnings.warn(
            "`avoid_subparsers=` is deprecated! use `tyro.conf.AvoidSubcommands[]`"
            " instead.",
            stacklevel=2,
        )

    # Internally, we distinguish between two concepts:
    # - "default", which is used for individual arguments.
    # - "default_instance", which is used for _fields_ (which may be broken down into
    #   one or many arguments, depending on various factors).
    #
    # This could be revisited.
    default_instance_internal: _singleton.NonpropagatingMissingType | OutT = (
        _singleton.MISSING_NONPROP if default is None else default
    )

    # We wrap our type with a dummy dataclass if it can't be treated as a nested type.
    # For example: passing in f=int will result in a dataclass with a single field
    # typed as int.
    #
    # Why don't we always use a dummy dataclass?
    # => Docstrings for inner structs are currently lost when we nest struct types.
    f = _resolver.TypeParamResolver.resolve_params_and_aliases(f)
    if not _fields.is_struct_type(cast(type, f), default_instance_internal):
        dummy_field = cast(
            dataclasses.Field,
            dataclasses.field(),
        )
        f = dataclasses.make_dataclass(
            cls_name="dummy",
            fields=[(_strings.dummy_field_name, cast(type, f), dummy_field)],
            frozen=True,
        )
        default_instance_internal = f(default_instance_internal)  # type: ignore
        dummy_wrapped = True
    else:
        dummy_wrapped = False

    # Read and fix arguments. If the user passes in --field_name instead of
    # --field-name, correct for them.
    args = list(sys.argv[1:]) if args is None else list(args)

    # Fix arguments. This will modify all option-style arguments replacing
    # underscores with hyphens, or vice versa if use_underscores=True.
    # If two options are ambiguous, e.g., --a_b and --a-b, raise a runtime error.
    modified_args: dict[str, str] = {}
    for index, arg in enumerate(args):
        if not arg.startswith("--"):
            continue

        if "=" in arg:
            arg, _, val = arg.partition("=")
            fixed = "--" + _strings.swap_delimeters(arg[2:]) + "=" + val
        else:
            fixed = "--" + _strings.swap_delimeters(arg[2:])
        if (
            return_unknown_args
            and fixed in modified_args
            and modified_args[fixed] != arg
        ):
            raise RuntimeError(
                "Ambiguous arguments: " + modified_args[fixed] + " and " + arg
            )
        modified_args[fixed] = arg
        args[index] = fixed

    # If we pass in the --tyro-print-completion or --tyro-write-completion flags: turn
    # formatting tags, and get the shell we want to generate a completion script for
    # (bash/zsh/tcsh).
    #
    # shtab also offers an add_argument_to() functions that fulfills a similar goal, but
    # manual parsing of argv is convenient for turning off formatting.
    #
    # Note: --tyro-print-completion is deprecated! --tyro-write-completion is less prone
    # to errors from accidental logging, print statements, etc.
    print_completion = False
    write_completion = False
    if len(args) >= 2:
        # We replace underscores with hyphens to accomodate for `use_undercores`.
        print_completion = args[0].replace("_", "-") == "--tyro-print-completion"
        write_completion = (
            len(args) >= 3 and args[0].replace("_", "-") == "--tyro-write-completion"
        )

    # Note: setting USE_RICH must happen before the parser specification is generated.
    # TODO: revisit this. Ideally we should be able to eliminate the global state
    # changes.
    completion_shell = None
    completion_target_path = None
    if print_completion or write_completion:
        completion_shell = args[1]
    if write_completion:
        completion_target_path = pathlib.Path(args[2])
    if print_completion or write_completion or return_parser:
        _arguments.USE_RICH = False
    else:
        _arguments.USE_RICH = True

    # Patch the parser cosntructor here!
    # TODO: Unpatch for repeated use?
    ParserSpecification.from_callable_or_type = staticmethod(from_callable_or_type)

    if registry is not None:
        with registry:
            parser_spec = ParserSpecification.from_callable_or_type(
                f,
                markers=set(),
                description=description,
                parent_classes=set(),  # Used for recursive calls.
                default_instance=default_instance_internal,  # Overrides for default values.
                intern_prefix="",  # Used for recursive calls.
                extern_prefix="",  # Used for recursive calls.
                add_help=add_help,
            )
    else:
        # Map a callable to the relevant CLI arguments + subparsers.
        parser_spec = ParserSpecification.from_callable_or_type(
            f,
            markers=set(),
            description=description,
            parent_classes=set(),  # Used for recursive calls.
            default_instance=default_instance_internal,  # Overrides for default values.
            intern_prefix="",  # Used for recursive calls.
            extern_prefix="",  # Used for recursive calls.
            add_help=add_help,
        )

    # consolidate_subcomm = parser_spec.consolidate_subcommand_args

    consolidate_subcomm = False

    if config is not None:
        consolidate_subcomm = _markers.ConsolidateSubcommandArgs in config

    if apply_subcomm_defaults:
        # TODO: This might be a false positive for subcommand_from_dict
        # annotated with consolidate, it seems it's replacted with dummy.
        # To solve this, check for consolidatesubcommand yourself before parser_spec
        # (and before dummy wrap).
        if not consolidate_subcomm:
            # TODO: Better error name.
            raise ValueError(
                "Defaults instantiation enabled only for tyro.conf.ConsolidateSubcommandArgs"
            )

        if config is not None and conf.OmitSubcommandPrefixes in tuple(config):
            raise ValueError(
                "OmitSubcommandPrefix not compatible with apply_subcomm_defaults."
            )

        # TODO: Does it work with OmitArgPrefixes?

    if consolidate_subcomm and apply_subcomm_defaults:
        assert isinstance(parser_spec, _ParserSpecification)

        root_to_user_option = {}
        dsu = DSU()

        first_non_choice = None
        for index, arg_item in enumerate(args):
            if dsu.is_node(arg_item):
                root = dsu.find(arg_item)

                if root in root_to_user_option:
                    raise ValueError(
                        "Cannot specify multiple choices for the same subcommand,"
                        f"got {arg_item} and {root_to_user_option[root]}"
                    )

                root_to_user_option[root] = arg_item
            else:
                first_non_choice = index
                break

        # found_defaults = False
        new_args = []
        queue: deque[_Node] = deque([parser_spec._node])
        while queue:
            node = queue.pop()
            if node.is_subparser:
                choice = node.default_name

                if (choice_root := dsu.find(choice)) in root_to_user_option:
                    choice = root_to_user_option[choice_root]

                for nei in node.edges:
                    if nei.default_name == choice:
                        queue.append(nei)
                        break
            else:
                if node.default_name != _Node.DEFAULT_PARSER_NAME:
                    new_args.append(node.default_name)
                    # found_defaults = True
                    queue.extend(node.edges)
                else:
                    for nei in node.edges:
                        queue.appendleft(nei)

        if first_non_choice is not None:
            new_args.extend(args[first_non_choice:])

        args = new_args

        # if not found_defaults:
        #     warnings.warn("apply_subcomm_defaults set to True but no defaults found.")

    # Generate parser!
    formatter_class = TyroFlatSubcommandHelpFormatter  # (see start of this file)
    with _argparse_formatter.ansi_context():
        parser = _argparse_formatter.TyroArgumentParser(
            prog=prog,
            # formatter_class=_argparse_formatter.TyroArgparseHelpFormatter,
            formatter_class=formatter_class,
            allow_abbrev=False,
        )
        parser._parser_specification = parser_spec
        parser._parsing_known_args = return_unknown_args
        parser._console_outputs = console_outputs
        parser._args = args
        parser_spec.apply(parser, force_required_subparsers=False)

        # Print help message when no arguments are passed in. (but arguments are
        # expected)
        # if len(args) == 0 and parser_spec.has_required_args:
        #     args = ["--help"]

        if return_parser:
            _arguments.USE_RICH = True
            return parser

        if print_completion or write_completion:
            _arguments.USE_RICH = True
            assert completion_shell in (
                "bash",
                "zsh",
                "tcsh",
            ), (
                "Shell should be one `bash`, `zsh`, or `tcsh`, but got"
                f" {completion_shell}"
            )

            if write_completion and completion_target_path != pathlib.Path("-"):
                assert completion_target_path is not None
                completion_target_path.write_text(
                    shtab.complete(
                        parser=parser,
                        shell=completion_shell,
                        root_prefix=f"tyro_{parser.prog}",
                    )
                )
            else:
                print(
                    shtab.complete(
                        parser=parser,
                        shell=completion_shell,
                        root_prefix=f"tyro_{parser.prog}",
                    )
                )
            sys.exit()

        if return_unknown_args:
            namespace, unknown_args = parser.parse_known_args(args=args)
        else:
            unknown_args = None
            namespace = parser.parse_args(args=args)
        value_from_prefixed_field_name = vars(namespace)

    if dummy_wrapped:
        value_from_prefixed_field_name = {
            k.replace(_strings.dummy_field_name, ""): v
            for k, v in value_from_prefixed_field_name.items()
        }

    try:
        # Attempt to call `f` using whatever was passed in.
        get_out, consumed_keywords = _calling.callable_with_args(
            f,
            parser_spec,
            default_instance_internal,
            value_from_prefixed_field_name,
            field_name_prefix="",
        )
    except _calling.InstantiationError as e:
        # Print prettier errors.
        # This doesn't catch errors raised directly by get_out(), since that's
        # called later! This is intentional, because we do less error handling
        # for the root callable. Relevant: the `field_name_prefix == ""`
        # condition in `callable_with_args()`!

        # Emulate argparse's error behavior when invalid arguments are passed in.
        from rich.console import Console, Group
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.style import Style
        from tyro._argparse_formatter import THEME

        if console_outputs:
            console = Console(theme=THEME.as_rich_theme(), stderr=True)
            console.print(
                Panel(
                    Group(
                        "[bright_red][bold]Error parsing"
                        f" {'/'.join(e.arg.lowered.name_or_flags) if isinstance(e.arg, _arguments.ArgumentDefinition) else e.arg}[/bold]:[/bright_red] {e.message}",
                        *cast(  # Cast to appease mypy...
                            list,
                            (
                                []
                                if not isinstance(e.arg, _arguments.ArgumentDefinition)
                                or e.arg.lowered.help is None
                                else [
                                    Rule(style=Style(color="red")),
                                    "Argument helptext:",
                                    Padding(
                                        Group(
                                            f"{'/'.join(e.arg.lowered.name_or_flags)} [bold]{e.arg.lowered.metavar}[/bold]",
                                            e.arg.lowered.help,
                                        ),
                                        pad=(0, 0, 0, 4),
                                    ),
                                    # Rule(style=Style(color="red")),
                                    # f"For full helptext, see [bold]{parser.prog} --help[/bold]",
                                    *(
                                        [
                                            Rule(style=Style(color="red")),
                                            f"For full helptext, see [bold]{parser.prog} --help[/bold]",
                                        ]
                                        if parser.add_help
                                        else []
                                    ),
                                ]
                            ),
                        ),
                    ),
                    title="[bold]Value error[/bold]",
                    title_align="left",
                    border_style=Style(color="red"),
                )
            )
        sys.exit(2)

    assert len(value_from_prefixed_field_name.keys() - consumed_keywords) == 0, (
        f"Parsed {value_from_prefixed_field_name.keys()}, but only consumed"
        f" {consumed_keywords}"
    )

    if dummy_wrapped:
        get_wrapped_out = get_out
        get_out = lambda: getattr(get_wrapped_out(), _strings.dummy_field_name)  # noqa

    if return_unknown_args:
        assert unknown_args is not None, "Should have parsed with `parse_known_args()`"
        # If we're parsed unknown args, we should return the original args, not
        # the fixed ones.
        unknown_args = [modified_args.get(arg, arg) for arg in unknown_args]
        return get_out, unknown_args  # type: ignore
    else:
        assert unknown_args is None, "Should have parsed with `parse_args()`"
        return get_out  # type: ignore
