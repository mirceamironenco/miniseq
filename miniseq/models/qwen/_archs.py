from dataclasses import dataclass, field
from typing import Final

from miniseq.models import ModelConfig, get_family_decorator
from miniseq.transformer import (
    AttentionConfig,
    FFNConfig,
    RotatedRotaryEncoderConfig,
    TransformerNormOrder,
)


@dataclass(kw_only=True)
class QwenModelConfig(ModelConfig):
    pos_encoder_config: RotatedRotaryEncoderConfig = field(
        default_factory=lambda: RotatedRotaryEncoderConfig(rope_theta=10_000.0)
    )


QWEN_ORG: Final = "qwen"

# For r1 distilled qwen models.
DEEPSEEK_ORG: Final = "deepseek-ai"


def register_qwen_models() -> None:
    arch = get_family_decorator(family=QWEN_ORG, config_kls=QwenModelConfig)

    @arch("qwen2.5-3b")
    def _qwen25_3_b() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            num_heads=16, num_kv_heads=2, head_dim=None, bias=True, output_bias=False
        )

        ffn_cfg = FFNConfig(
            inner_dim=11008,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            bias=False,
        )

        config = QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_size=151_936,
            bos_idx=151_643,
            eos_idx=151_643,
            pad_idx=None,
            num_layers=36,
            max_seq_len=32768,
            norm_order=TransformerNormOrder.PRE,
            norm_eps=1e-6,
            tie_weights=True,
        )

        config.pos_encoder_config.rope_theta = 1000000.0

        config.model_dim = 2048

        return config

    @arch("qwen2.5-1.5b")
    def _qwen25_1_5_b() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            num_heads=12, num_kv_heads=2, head_dim=None, bias=True, output_bias=False
        )

        ffn_cfg = FFNConfig(
            inner_dim=8960,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            bias=False,
        )

        config = QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_size=151_936,
            bos_idx=151_643,
            eos_idx=151_643,
            pad_idx=None,
            num_layers=28,
            max_seq_len=131072,
            norm_order=TransformerNormOrder.PRE,
            norm_eps=1e-6,
            tie_weights=True,
        )

        config.pos_encoder_config.rope_theta = 1000000.0

        config.model_dim = 1536

        return config

    @arch("qwen2.5-1.5b-instruct")
    def _qwen25_1_5_b_instruct() -> QwenModelConfig:
        config = _qwen25_1_5_b()

        config.eos_idx = 151_645

        config.max_seq_len = 32768

        return config

    @arch("qwen2.5-0.5b-instruct")
    def _qwen25_0_5_b_instruct() -> QwenModelConfig:
        config = _qwen25_1_5_b_instruct()
        config.num_layers = 24
        config.model_dim = 896
        config.attn_config.num_heads = 14
        config.attn_config.num_kv_heads = 2
        config.ffn_config.inner_dim = 4864

        return config

    @arch("qwen2.5-math-1.5b")
    def _qwen25_math_1_5_b() -> QwenModelConfig:
        config = _qwen25_1_5_b()

        config.pos_encoder_config.rope_theta = 10000.0
        config.max_seq_len = 4096

        return config

    @arch("qwen2.5-math-1.5b-instruct")
    def _qwen25_math_1_5_b_instruct() -> QwenModelConfig:
        config = _qwen25_math_1_5_b()
        config.eos_idx = 151_645

        return config

    @arch("qwen2.5-7b-instruct")
    def _qwen25_7_b_instruct() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            num_heads=28, num_kv_heads=4, head_dim=None, bias=True, output_bias=False
        )

        ffn_cfg = FFNConfig(
            inner_dim=18944,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            bias=False,
        )

        config = QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_size=152_064,
            bos_idx=151_643,
            eos_idx=151_645,
            pad_idx=None,
            num_layers=28,
            max_seq_len=32768,
            norm_order=TransformerNormOrder.PRE,
            norm_eps=1e-6,
            tie_weights=False,
        )

        config.pos_encoder_config.rope_theta = 1000000.0

        config.model_dim = 3584

        return config

    # Additionally register deepseek-r1 distills.
    deepseek_arch = get_family_decorator(
        family=DEEPSEEK_ORG, config_kls=QwenModelConfig
    )

    @deepseek_arch("deepseek-r1-distill-qwen-1.5b")
    def _distill_qwen_1_5b() -> QwenModelConfig:
        config = _qwen25_math_1_5_b()

        config.max_seq_len = 131_072

        return config

    @arch("qwen3-1.7b")
    def _qwen3_1_7b() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            qk_norm=True,
            qk_norm_eps=1e-6,
            bias=False,
            output_bias=False,
        )

        ffn_cfg = FFNConfig(
            inner_dim=6144,
            multiple_of=1,
            ffn_dim_multiplier=1.0,
            bias=False,
        )

        config = QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_size=151_936,
            bos_idx=151_643,
            eos_idx=151_645,
            num_layers=28,
            max_seq_len=40960,
            norm_order=TransformerNormOrder.PRE,
            norm_eps=1e-6,
            tie_weights=True,
        )
        config.pos_encoder_config.rope_theta = 1000000.0

        config.model_dim = 2048

        return config

    @arch("qwen2.5-0.5b")
    def _qwen25_0_5_b() -> QwenModelConfig:
        config = _qwen25_0_5_b_instruct()
        config.eos_idx = 151_643

        return config

    @arch("qwen2.5-3b-instruct")
    def _qwen25_3_b_instruct() -> QwenModelConfig:
        config = _qwen25_3_b()
        config.eos_idx = 151_645
        return config
