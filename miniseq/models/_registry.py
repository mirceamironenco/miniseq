from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from huggingface_hub import ModelCard

from miniseq.data import SequenceBatch
from miniseq.transformer import (
    AttentionConfig,
    FFNConfig,
    FlexSDPA,
    PositionalEncoderConfig,
    TransformerNormOrder,
)


@dataclass(kw_only=True)
class ModelConfig:
    attn_config: AttentionConfig

    ffn_config: FFNConfig

    pos_encoder_config: PositionalEncoderConfig

    norm_order: TransformerNormOrder = TransformerNormOrder.PRE

    norm_eps: float = 1e-6

    tie_weights: bool = False

    _model_dim: int = field(init=False)
    """Embedding dimension of the model. It should be set after config is instantiated,
    using the `model_dim` property. `model_dim` property is mutable and automatically
    changes model_dim of the underlying attn_cfg and ffn_cfg.

    Example:
        config = ModelConfig(**kwargs)
        config.model_dim = 4096 
        assert config.attn_cfg.model_dim == config.ffn_cfg == 4096
    """

    _name: str = field(init=False)
    """Registry name. Set if the model is a registered architecture."""

    vocab_size: int

    pad_idx: int | None = None

    eos_idx: int | None = None

    bos_idx: int | None = None

    unk_idx: int | None = None

    num_layers: int = 32

    max_seq_len: int = 8192

    cut_cross_entropy: bool = False

    finetune_repo_id: str | None = None

    @property
    def base_model_repo_id(self) -> str:
        return get_model_repo_id(self.model)

    @property
    def repo_id(self) -> str:
        if self.finetune_repo_id is not None:
            return self.finetune_repo_id

        return self.base_model_repo_id

    @property
    def model(self) -> str:
        return self._name

    @property
    def model_dim(self) -> int:
        return self._model_dim

    @model_dim.setter
    def model_dim(self, model_dim: int) -> None:
        self._model_dim = model_dim

        self.attn_config.model_dim = model_dim

        self.ffn_config.model_dim = model_dim

    @classmethod
    def from_model_name(
        cls,
        name: str,
        *,
        flex_attention: bool | None = None,
        cut_cross_entropy: bool | None = None,
        finetune_repo_id: str | None = None,
        autotune_flex: bool = True,
        reduce_flex_stages: bool = False,
        flash_attention2: bool = False,
    ) -> ModelConfig:
        model_config = MODEL_REGISTRY[name]()
        model_config._name = name

        model_config.finetune_repo_id = finetune_repo_id

        if finetune_repo_id is not None:
            verify_base_models_match(name, finetune_repo_id)

        if flex_attention is not None and flex_attention:
            assert not flash_attention2
            # Note: If not using document packing, flex attention is not recommended.

            SequenceBatch.using_flex_attention = flex_attention

            model_config.attn_config.force_flex = flex_attention

            if autotune_flex:
                FlexSDPA.compile_flex_attention(mode="max-autotune", fullgraph=True)
            else:
                FlexSDPA.compile_flex_attention()

        if cut_cross_entropy is not None:
            model_config.cut_cross_entropy = cut_cross_entropy

        if reduce_flex_stages:
            model_config.attn_config.reduce_flex_stages = reduce_flex_stages

        if flash_attention2:
            assert not flex_attention

            model_config.attn_config.force_flash2 = flash_attention2

        return model_config


class CheckpointConverter(Protocol):
    def __call__(
        self, checkpoint: dict[str, Any], config: ModelConfig
    ) -> dict[str, Any]: ...


MODEL_REGISTRY: dict[str, Callable[..., ModelConfig]] = {}

MODEL_FAMILY_REGISTRY: dict[str, str] = {}

FAMILY_MODEL_REGISTRY: dict[str, set[str]] = defaultdict(set)

FAMILY_CONFIG_KLS_REGISTRY: dict[str, type[ModelConfig]] = defaultdict(
    lambda: ModelConfig
)

FAMILY_CKPT_TO_HF: dict[str, CheckpointConverter] = {}

FAMILY_CKPT_FROM_HF: dict[str, CheckpointConverter] = {}

ModelConfigT_cov = TypeVar("ModelConfigT_cov", bound=ModelConfig, covariant=True)
ModelConfigT = TypeVar("ModelConfigT", bound=ModelConfig)


class ModelConfigBuilder(Protocol[ModelConfigT_cov]):
    def __call__(self) -> ModelConfigT_cov: ...


class FamilyModelDecorator(Protocol[ModelConfigT]):
    def __call__(
        self, builder: ModelConfigBuilder[ModelConfigT], /
    ) -> ModelConfigBuilder[ModelConfigT]: ...


def get_family_decorator(
    family: str, config_kls: type[ModelConfigT] = ModelConfig
) -> Callable[[str], FamilyModelDecorator[ModelConfigT]]:
    def family_decorator(
        name: str,
    ) -> FamilyModelDecorator[ModelConfigT]:
        def _register_arch(
            fn: ModelConfigBuilder[ModelConfigT_cov],
        ) -> ModelConfigBuilder[ModelConfigT_cov]:
            if name in MODEL_REGISTRY:
                raise ValueError(f"Model {name} already registered.")

            if name in FAMILY_MODEL_REGISTRY[family]:
                raise ValueError(f"Model {name} already registered to family {family}.")

            MODEL_REGISTRY[name] = fn
            MODEL_FAMILY_REGISTRY[name] = family
            FAMILY_MODEL_REGISTRY[family].add(name)
            FAMILY_CONFIG_KLS_REGISTRY[family] = config_kls

            return fn

        return _register_arch

    return family_decorator


def register_family_checkpoint_converter(
    family: str,
    *,
    from_hf_converter: CheckpointConverter,
    to_hf_converter: CheckpointConverter,
) -> None:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(
            f"No registered families named {family}, register a model first."
        )

    FAMILY_CKPT_FROM_HF[family] = from_hf_converter

    FAMILY_CKPT_TO_HF[family] = to_hf_converter


def get_to_hf_checkpoint_converter(family: str) -> CheckpointConverter:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(f"No registered families named {family}.")

    if family not in FAMILY_CKPT_TO_HF:
        raise ValueError(
            f"No converter to hugging face format registered for model family {family}."
        )

    return FAMILY_CKPT_TO_HF[family]


def get_from_hf_ckpt_converter(family: str) -> CheckpointConverter:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(f"No registered families named {family}.")

    if family not in FAMILY_CKPT_FROM_HF:
        raise ValueError(
            f"No converter from hugging face format registered for model family {family}."
        )

    return FAMILY_CKPT_FROM_HF[family]


def get_model_family(model: str) -> str:
    """Returns the family of `model`."""

    return MODEL_FAMILY_REGISTRY[model]


def get_model_repo_id(model: str) -> str:
    """Returns the HuggingFace Hub model repo ID of `model`."""

    if model not in MODEL_FAMILY_REGISTRY:
        raise ValueError(f"No registered family for model {model}.")

    family = MODEL_FAMILY_REGISTRY[model]

    return f"{family}/{model}"


def all_model_families() -> list[str]:
    """Returns all registered model families."""

    return list(FAMILY_MODEL_REGISTRY.keys())


def get_models_from_family(family: str, newest_first: bool = True) -> list[str]:
    """Returns all models belonging to `family`."""

    models = list(FAMILY_MODEL_REGISTRY[family])

    if newest_first:
        return models[::-1]

    return models


def all_registered_models(newest_first: bool = True) -> list[str]:
    """Returns all registered models."""

    models = list(MODEL_REGISTRY.keys())

    if newest_first:
        return models[::-1]

    return models


def model_is_registered(model_name: str) -> bool:
    return model_name in MODEL_REGISTRY


def verify_base_models_match(base_model: str, repo_id: str) -> None:
    card = ModelCard.load(repo_id)

    card_data = card.data.to_dict()

    base_repo_id = get_model_repo_id(base_model)

    if card_base_model := card_data["base_model"][0].lower() != base_repo_id.lower():
        raise ValueError(
            f"Repo id {repo_id} base model expected {base_repo_id}, found {card_base_model}."
        )
