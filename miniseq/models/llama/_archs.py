from dataclasses import dataclass, field
from functools import partial
from typing import Final

from miniseq.models import ModelConfig, get_family_decorator
from miniseq.transformer import (
    AttentionConfig,
    FFNConfig,
    LLaMARoPEScaleConfig,
    RotaryEncoderConfig,
    TransformerNormOrder,
    llama_scale_freqs,
)


@dataclass(kw_only=True)
class LlamaModelConfig(ModelConfig):
    pos_encoder_config: RotaryEncoderConfig = field(
        default_factory=lambda: RotaryEncoderConfig(rope_theta=10_000.0)
    )


META_ORG: Final = "meta-llama"


def register_llama_models() -> None:
    arch = get_family_decorator(family=META_ORG, config_kls=LlamaModelConfig)

    @arch("llama-3-8b")
    def _llama3_8b() -> LlamaModelConfig:
        # https://huggingface.co/meta-llama/Meta-Llama-3-8B
        attn_cfg = AttentionConfig(
            num_heads=32, num_kv_heads=8, head_dim=None, bias=False, output_bias=False
        )

        ffn_cfg = FFNConfig(
            inner_dim=int(4096 * 4 * 1.3),
            multiple_of=1024,
            ffn_dim_multiplier=2 / 3,
            bias=False,
        )

        config = LlamaModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_size=128_256,
            bos_idx=128_000,
            eos_idx=128_001,
            pad_idx=None,
            num_layers=32,
            max_seq_len=8192,
            norm_order=TransformerNormOrder.PRE,
            norm_eps=1e-5,
            tie_weights=False,
        )

        config.pos_encoder_config.rope_theta = 500_000.0

        config.model_dim = 4096

        return config

    @arch("llama-3.1-8b")
    def _llama3_1_8b() -> LlamaModelConfig:
        config = _llama3_8b()

        config.max_seq_len = 131_072

        config.pos_encoder_config.freqs_init_fn = partial(
            llama_scale_freqs, rope_scale=LLaMARoPEScaleConfig()
        )

        return config

    @arch("llama-3.1-8b-instruct")
    def _llama3_1_8b_instruct() -> LlamaModelConfig:
        return _llama3_1_8b()

    @arch("llama-3.2-3b")
    def _llama3_2_3b() -> LlamaModelConfig:
        config = _llama3_1_8b()

        config.model_dim = 3072
        config.num_layers = 28
        config.tie_weights = True
        config.ffn_config.inner_dim = int(3072 * 4 * 1.0)
        config.ffn_config.multiple_of = 256
        config.attn_config.num_heads = 24
        config.attn_config.num_kv_heads = 8

        config.pos_encoder_config.freqs_init_fn = partial(
            llama_scale_freqs, rope_scale=LLaMARoPEScaleConfig(factor=32.0)
        )

        return config

    @arch("llama-3.2-3b-instruct")
    def _llama_3_2_3b_instruct() -> LlamaModelConfig:
        return _llama3_2_3b()

    @arch("llama-3.2-1b")
    def _llama3_2_1b() -> LlamaModelConfig:
        config = _llama3_1_8b()

        config.model_dim = 2048
        config.num_layers = 16
        config.tie_weights = True
        config.ffn_config.inner_dim = int(2048 * 4 * 1.5)
        config.ffn_config.multiple_of = 256
        config.attn_config.num_heads = 32
        config.attn_config.num_kv_heads = 8

        config.pos_encoder_config.freqs_init_fn = partial(
            llama_scale_freqs, rope_scale=LLaMARoPEScaleConfig(factor=32.0)
        )

        return config

    @arch("llama-3.2-1b-instruct")
    def _llama_3_2_1b_instruct() -> LlamaModelConfig:
        return _llama3_2_1b()
