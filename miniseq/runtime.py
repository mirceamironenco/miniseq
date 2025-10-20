_models_registered: bool = False


def register_models() -> None:
    global _models_registered

    if _models_registered:
        return

    from miniseq.models import register_family_checkpoint_converter
    from miniseq.models.llama import (
        META_ORG,
        convert_hf_llama_ckpt_to_mini,
        convert_llama_mini_ckpt_to_hf,
        register_llama_models,
    )

    register_llama_models()

    register_family_checkpoint_converter(
        META_ORG,
        from_hf_converter=convert_hf_llama_ckpt_to_mini,
        to_hf_converter=convert_llama_mini_ckpt_to_hf,
    )

    from miniseq.models.qwen import (
        DEEPSEEK_ORG,
        QWEN_ORG,
        convert_qwen_hf_checkpoint_to_mini,
        convert_qwen_mini_to_hf_checkpoint,
        register_qwen_models,
    )

    register_qwen_models()

    register_family_checkpoint_converter(
        QWEN_ORG,
        from_hf_converter=convert_qwen_hf_checkpoint_to_mini,
        to_hf_converter=convert_qwen_mini_to_hf_checkpoint,
    )

    register_family_checkpoint_converter(
        DEEPSEEK_ORG,
        from_hf_converter=convert_qwen_hf_checkpoint_to_mini,
        to_hf_converter=convert_qwen_mini_to_hf_checkpoint,
    )

    _models_registered = True
