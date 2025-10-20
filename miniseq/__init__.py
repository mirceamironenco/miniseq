import os


def setup_vllm() -> None:
    vllm_logging = os.environ.get("VLLM_CONFIGURE_LOGGING", None)

    if vllm_logging is None:
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


setup_vllm()
