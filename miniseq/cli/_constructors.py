from __future__ import annotations

import dataclasses
from typing import Annotated

import tyro

from miniseq.cli import constructor_registry


@constructor_registry.primitive_rule
def _torch_dtype_rule(
    type_info: tyro.constructors.PrimitiveTypeInfo,
) -> tyro.constructors.PrimitiveConstructorSpec | None:
    # Potentially expensive imports done lazily to not clog CLI
    import torch

    from miniseq.utils import make_dtype

    if type_info.type != torch.dtype:
        return None

    return tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="{bfloat16,float32,float64}",
        instance_from_str=lambda args: make_dtype(args[0]),  # type: ignore
        is_instance=lambda instance: isinstance(instance, torch.dtype),
        str_from_instance=lambda instance: [str(instance).split(".")[-1]],
        choices=("bfloat16", "float32", "float64"),
    )


# https://github.com/brentyi/tyro/issues/308#issuecomment-2920006916
# Note: This doesn't do validation on choices at the CLI level.
CLIModelName = Annotated[
    str,
    tyro.conf.arg(
        help_behavior_hint=lambda df: f"(default: {df}, run entry.py model_registry)",
        # metavar="{" + ",".join(all_registered_models()[:3]) + ",...}",
        metavar="{qwen2.5-3b-instruct,qwen2.5-0.5b,qwen3-1.7b,...}",
        constructor_factory=lambda: Annotated[  # type: ignore
            str,
            tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="",
                instance_from_str=lambda args: args[0],
                is_instance=lambda instance: isinstance(instance, str),
                str_from_instance=lambda instance: [instance],
                # Does not work with constructor_factory returning PrimitiveSpec
                # choices=tuple(all_registered_models()),
            ),
        ],
    ),
]


@constructor_registry.struct_rule
def _model_options_rule(
    type_info: tyro.constructors.StructTypeInfo,
) -> tyro.constructors.StructConstructorSpec | None:
    # Potentially expensive imports done lazily to not clog CLI
    import torch

    from miniseq.configs import PretrainedModelConfig

    if type_info.type != PretrainedModelConfig:
        return None

    if not isinstance(type_info.default, PretrainedModelConfig):
        assert type_info.default in (
            tyro.constructors.MISSING,
            tyro.constructors.MISSING_NONPROP,
        )
        type_info = dataclasses.replace(type_info, default=PretrainedModelConfig())

    default = (
        type_info.default.name,
        type_info.default.dtype,
        type_info.default.flex_attention,
        type_info.default.flash_attention2,
        type_info.default.reduce_flex,
        type_info.default.use_cce,
        type_info.default.load_on_cpu,
        type_info.default.finetune_repo_id,
    )

    return tyro.constructors.StructConstructorSpec(
        instantiate=PretrainedModelConfig,
        fields=(
            tyro.constructors.StructFieldSpec(
                name="name",
                type=CLIModelName,  # type: ignore
                default=default[0],
            ),
            tyro.constructors.StructFieldSpec(
                name="dtype", type=torch.dtype, default=default[1]
            ),
            tyro.constructors.StructFieldSpec(
                name="flex_attention", type=bool, default=default[2]
            ),
            tyro.constructors.StructFieldSpec(
                name="flash_attention2", type=bool, default=default[3]
            ),
            tyro.constructors.StructFieldSpec(
                name="reduce_flex", type=bool, default=default[4]
            ),
            tyro.constructors.StructFieldSpec(
                name="use_cce", type=bool, default=default[5]
            ),
            tyro.constructors.StructFieldSpec(
                name="load_on_cpu", type=bool, default=default[6]
            ),
            tyro.constructors.StructFieldSpec(
                name="finetune_repo_id",
                type=str | None,  # type: ignore
                default=default[7],
            ),
        ),
    )
