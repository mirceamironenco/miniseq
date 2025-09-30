from typing import Any

import torch

from miniseq.models import ModelConfig, convert_model_state_dict

# fmt: off
_key_map_hf_to_mini = {
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.o_proj.",
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_norm.",
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"decoder.layers.\1.ffn.w1.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"decoder.layers.\1.ffn.w2.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"decoder.layers.\1.ffn.w3.",
    r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"decoder.layers.\1.attn_norm.",
    r"^model\.norm\.":                                       r"decoder.norm.",
    r"^model\.embed_tokens\.":                               r"decoder_frontend.",
    r"^lm_head\.":                                           r"final_proj.",
}
# fmt: on

# fmt: off
_key_map_llama_to_mini = {
    r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
    r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
    r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
    r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.o_proj.",
    r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.attn_norm.",
    r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.w1.",
    r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.w2.",
    r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.w3.",
    r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_norm.",
    r"^norm\.":                               r"decoder.norm.",
    r"^tok_embeddings\.":                     r"decoder_frontend.",
    r"^output\.":                             r"final_proj.",
}
# fmt: on

# fmt: off
_key_map_mini_to_hf = {
    r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"model.layers.\1.self_attn.q_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"model.layers.\1.self_attn.k_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"model.layers.\1.self_attn.v_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"model.layers.\1.self_attn.o_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn_norm\.":                 r"model.layers.\1.post_attention_layernorm.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w1\.":                  r"model.layers.\1.mlp.gate_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w2\.":                  r"model.layers.\1.mlp.down_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w3\.":                  r"model.layers.\1.mlp.up_proj.",
    r"^decoder\.layers\.([0-9]+)\.attn_norm\.":                r"model.layers.\1.input_layernorm.",
    r"^decoder\.norm\.":                                       r"model.norm.",
    r"^decoder_frontend\.":                                    r"model.embed_tokens.",
    r"^final_proj\.":                                          r"lm_head.",
}
# fmt: on


def convert_hf_llama_ckpt_to_mini(
    checkpoint: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    # Note: This function will modify 'checkpoint' in place.

    if config.tie_weights:
        assert "lm_head.weight" not in checkpoint

        checkpoint["lm_head.weight"] = checkpoint["model.embed_tokens.weight"]

    assert "lm_head.weight" in checkpoint

    head_dim = config.model_dim // config.attn_config.num_heads
    attn_heads, kv_heads = config.attn_config.num_heads, config.attn_config.num_kv_heads

    # https://github.com/huggingface/transformers/issues/25199
    def permute_rotary(w: torch.Tensor, num_heads: int) -> torch.Tensor:
        # (H, M) -> (H_d, 2, D / 2, M)
        w = w.view(num_heads, 2, head_dim // 2, config.model_dim)

        # (H_d, 2, D / 2, M) -> (H_d, D / 2, 2, M)
        w = w.transpose(1, 2)

        # (H_d, D / 2, 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    for index in range(config.num_layers):
        q_key = f"model.layers.{index}.self_attn.q_proj.weight"
        k_key = f"model.layers.{index}.self_attn.k_proj.weight"

        q_proj = checkpoint[q_key]
        k_proj = checkpoint[k_key]

        q_proj = permute_rotary(q_proj, attn_heads)
        k_proj = permute_rotary(k_proj, kv_heads)

        checkpoint[q_key] = q_proj
        checkpoint[k_key] = k_proj

    if not config.tie_weights:
        return convert_model_state_dict(checkpoint, _key_map_hf_to_mini)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in _key_map_hf_to_mini.items()}

    local_key_map[r"^lm_head\."] = r"final_proj.embed."

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    return new_ckpt


def convert_original_llama_ckpt_to_mini(
    checkpoint: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    checkpoint = {k: v for (k, v) in checkpoint.items() if "rope.freqs" not in k}

    if not config.tie_weights:
        return convert_model_state_dict(checkpoint, _key_map_llama_to_mini)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in _key_map_llama_to_mini.items()}

    local_key_map[r"^output\."] = r"final_proj.embed."

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    return new_ckpt


def convert_llama_mini_ckpt_to_hf(
    checkpoint: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    local_key_map = {k: v for k, v in _key_map_mini_to_hf.items()}

    if config.tie_weights:
        local_key_map[r"^final_proj\.embed\."] = local_key_map.pop(r"^final_proj\.")

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    head_dim = config.model_dim // config.attn_config.num_heads
    attn_heads, kv_heads = config.attn_config.num_heads, config.attn_config.num_kv_heads

    # https://github.com/huggingface/transformers/issues/25199
    def unpermute_rotary(w: torch.Tensor, num_heads: int) -> torch.Tensor:
        # (H, M) -> (H_d, D / 2, 2, M)
        w = w.view(num_heads, head_dim // 2, 2, config.model_dim)

        # (H_d, D / 2, 2, M) -> (H_d, 2, D / 2, M)
        w = w.transpose(1, 2)

        # (H_d, 2, D / 2, M) -> (H, M)
        return w.reshape(-1, config.model_dim)

    for index in range(config.num_layers):
        q_key = f"model.layers.{index}.self_attn.q_proj.weight"
        k_key = f"model.layers.{index}.self_attn.k_proj.weight"

        q_proj = new_ckpt[q_key]
        k_proj = new_ckpt[k_key]

        q_proj = unpermute_rotary(q_proj, attn_heads)
        k_proj = unpermute_rotary(k_proj, kv_heads)

        new_ckpt[q_key] = q_proj
        new_ckpt[k_key] = k_proj

    if config.tie_weights:
        if "lm_head.weight" in new_ckpt:
            del new_ckpt["lm_head.weight"]

    return new_ckpt
