from typing import Any

from miniseq.models import ModelConfig, convert_model_state_dict

# fmt: off
key_map = {
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.o_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":        r"decoder.layers.\1.self_attn.q_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":        r"decoder.layers.\1.self_attn.k_norm.",
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
reverse_key_map = {
    r"^decoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":          r"model.layers.\1.self_attn.q_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":          r"model.layers.\1.self_attn.k_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":          r"model.layers.\1.self_attn.v_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.o_proj\.":          r"model.layers.\1.self_attn.o_proj.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.q_norm\.":          r"model.layers.\1.self_attn.q_norm.",
    r"^decoder\.layers\.([0-9]+)\.self_attn\.k_norm\.":          r"model.layers.\1.self_attn.k_norm.",
    r"^decoder\.layers\.([0-9]+)\.ffn_norm\.":                   r"model.layers.\1.post_attention_layernorm.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w1\.":                    r"model.layers.\1.mlp.gate_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w2\.":                    r"model.layers.\1.mlp.down_proj.",
    r"^decoder\.layers\.([0-9]+)\.ffn\.w3\.":                    r"model.layers.\1.mlp.up_proj.",
    r"^decoder\.layers\.([0-9]+)\.attn_norm\.":                  r"model.layers.\1.input_layernorm.",
    r"^decoder\.norm\.":                                         r"model.norm.",
    r"^decoder_frontend\.":                                      r"model.embed_tokens.",
    r"^final_proj\.":                                            r"lm_head.",
}
# fmt: on


def convert_qwen_hf_checkpoint_to_mini(
    checkpoint: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    if not config.tie_weights:
        return convert_model_state_dict(checkpoint, key_map)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in key_map.items()}

    local_key_map[r"^lm_head\."] = r"final_proj.embed."

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    new_ckpt["final_proj.embed.weight"] = new_ckpt["decoder_frontend.weight"]

    return new_ckpt


def convert_qwen_mini_to_hf_checkpoint(
    checkpoint: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    if not config.tie_weights:
        return convert_model_state_dict(checkpoint, reverse_key_map)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in reverse_key_map.items()}

    local_key_map[r"^final_proj\.embed\."] = local_key_map.pop(r"^final_proj\.")

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    if "lm_head.weight" in new_ckpt:
        del new_ckpt["lm_head.weight"]

    return new_ckpt
