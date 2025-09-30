import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Literal, Protocol, TypedDict, overload

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore
from transformers.tokenization_utils_base import (
    AddedToken,  # type: ignore
    BatchEncoding,
    TruncationStrategy,
)
from transformers.utils.generic import PaddingStrategy, TensorType


class Message(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str


class Conversation(Protocol):
    messages: list[Message]


def make_chat_prefix(
    *,
    user_message: str,
    system_message: str | None = None,
    assistant_message: str | None = None,
) -> list[Message]:
    prefix: list[Message] = []

    # Note: Some tokenizer templates will add a system prompt irrespective of
    # one being provided, e.g. qwen-2.5-1.5b-math-instruct.
    if system_message is not None:
        prefix.append({"role": "system", "content": system_message})

    if not user_message:
        raise ValueError("`user_message` cannot be empty string.")

    prefix.append({"role": "user", "content": user_message})

    if assistant_message is not None:
        prefix.append({"role": "assistant", "content": assistant_message})

    return prefix


SPECIAL_TOKEN = Literal[
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]


class PretrainedHFTokenizer(Protocol):
    """A tokenizer as returned by AutoTokenizer.from_pretrained(...)."""

    bos_token_id: int | None

    # Note: Changed from optional to int.
    eos_token_id: int

    unk_token_id: int | None

    sep_token_id: int | None

    pad_token_id: int | None

    bos_token: str | None

    eos_token: str | None

    unk_token: str | None

    sep_token: str | None

    pad_token: str | None

    chat_template: str | dict[str, str] | None

    @overload
    def apply_chat_template(
        self,
        conversation: list[Message],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[False],
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: Literal[False] = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> str: ...

    @overload
    def apply_chat_template(
        self,
        conversation: list[list[Message]] | list[Conversation],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[False],
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: Literal[False] = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[str]: ...

    @overload
    def apply_chat_template(
        self,
        conversation: list[Message],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[True] = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: Literal[False] = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[int]: ...

    @overload
    def apply_chat_template(
        self,
        conversation: list[list[Message]] | list[Conversation],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[True] = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: Literal[False] = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[list[int]]: ...

    @overload
    def apply_chat_template(
        self,
        conversation: list[Message] | list[list[Message]] | list[Conversation],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: Literal[True] = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: Literal[True],
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BatchEncoding: ...

    def apply_chat_template(
        self,
        conversation: list[Message] | list[list[Message]] | list[Conversation],
        *,
        tools: list[dict | Callable] | None = None,
        documents: list[dict[str, str]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of
        token ids. This method is intended for use with chat models, and will read the
        tokenizer's chat_template attribute to determine the format and control tokens
        to use when converting.

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (`List[Dict]`, *optional*):
                A list of tools (callable functions) that will be accessible to the
                model. If the template does not support function calling, this argument
                will have no effect. Each tool should be passed as a JSON Schema, giving
                the name, description and argument types for the tool. See our [chat
                templating
                guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
                for more information.
            documents (`List[Dict[str, str]]`, *optional*):
                A list of dicts representing documents that will be accessible to the
                model if it is performing RAG (retrieval-augmented generation). If the
                template does not support RAG, this argument will have no effect. We
                recommend that each document should be a dict containing "title" and
                "text" keys. Please see the RAG section of the [chat templating
                guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
                for examples of passing documents with chat templates.
            chat_template (`str`, *optional*):
                A Jinja template to use for this conversion. It is usually not necessary
                to pass anything to this argument, as the model's template will be used
                by default.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate the start of an
                assistant message will be appended to the formatted output. This is
                useful when you want to generate a response from the model.  Note that
                this argument will be passed to the chat template, and so it must be
                supported in the template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final message in
                the chat is open-ended, without any EOS tokens. The model will continue
                this message rather than starting a new one. This allows you to
                "prefill" part of the model's response for it. Cannot be used at the
                same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                 Select a strategy to pad the returned sequences (according to the
                 model's padding side and padding index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no
                padding if only a single sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument
                `max_length` or to the maximum acceptable input length for the model if
                that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a
                batch with sequences of different lengths).
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if
                tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no
                effect if tokenize is `False`. If not specified, the tokenizer's
                `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if
                tokenize is `False`. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if
                tokenize is `False`.
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens
                generated by the assistant, the mask will contain 1. For user and system
                tokens, the mask will contain 0.  This functionality is only available
                for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be
            *accessible by the chat template.

        Returns:
            `Union[List[int], Dict]`: A list of token ids representing the tokenized
            chat so far, including control tokens. This output is ready to pass to the
            model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """
        ...

    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """
        ...

    def truncate_sequences(
        self,
        ids: list[int],
        pair_ids: list[int] | None = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str | TruncationStrategy = "longest_first",
        stride: int = 0,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (`int`, *optional*, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (`str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `'longest_first'`):
                The strategy to follow for truncation. Can be:

                - `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will truncate
                  token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
                  batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater
                  than the model maximum admissible input size).
            stride (`int`, *optional*, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            `Tuple[List[int], List[int], List[int]]`: The truncated `ids`, the truncated `pair_ids` and the list of
            overflowing tokens. Note: The *longest_first* strategy returns empty list of overflowing tokens if a pair
            of sequences (or a batch of pairs) is provided.
        """
        ...

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | torch.Tensor | np.ndarray,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        ...

    def decode(
        self,
        token_ids: int | list[int] | torch.Tensor | np.ndarray,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with
        options to remove special tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        ...

    def __call__(
        self,
        text: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        text_pair: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        text_target: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        text_pair_target: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Mapping:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        ...

    @overload
    def convert_ids_to_tokens(
        self, ids: int, skip_special_tokens: bool = False
    ) -> str: ...

    @overload
    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]: ...

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]: ...

    def get_chat_template(
        self, chat_template: str | None = None, tools: list[dict] | None = None
    ) -> str: ...

    def tokenize(
        self,
        text: str,
        pair: str | None = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]: ...

    @overload
    def encode(
        self,
        text: str | list[str] | list[int],
        *,
        text_pair: str | list[str] | list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: Literal["pt"],
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    def encode(
        self,
        text: str | list[str] | list[int],
        text_pair: str | list[str] | list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: NoneType = None,
        **kwargs,
    ) -> list[int]: ...

    def encode(
        self,
        text: str | list[str] | list[int],
        text_pair: str | list[str] | list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: Literal["pt", "tf", "np"] | TensorType | None = None,
        **kwargs,
    ) -> list[int] | torch.Tensor:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        ...

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        # Changed Iterable[str] to list, see https://github.com/python/typing/issues/256
        ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a Iterable of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `list[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        ...

    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        ...

    def get_vocab(self) -> dict[str, int]: ...

    def add_special_tokens(
        self,
        special_tokens_dict: dict[
            SPECIAL_TOKEN, str | AddedToken | Sequence[str | AddedToken]
        ],
        replace_additional_special_tokens=True,
    ) -> int: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


def load_hf_pretrained_tokenzier(
    model_repo_id: str,
    *,
    cache_dir: Path,
    force_download: bool = False,
    use_fast: bool = True,
    set_none_pad_token_to_eos: bool = True,
) -> PretrainedHFTokenizer:
    """Simplified equivalent to AutoTokenizer.from_pretrained(...) with type hints."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_repo_id,
        cache_dir=cache_dir,
        force_download=force_download,
        use_fast=use_fast,
        trust_remote_code=True,
    )

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer without eos_token_id not supported.")

    if tokenizer.pad_token_id is None and set_none_pad_token_to_eos:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def contains_bos_token(sequence: list[int], tokenizer: PretrainedHFTokenizer) -> bool:
    if tokenizer.bos_token_id is None:
        return False

    return tokenizer.bos_token_id in sequence


def contains_eos_token(sequence: list[int], tokenizer: PretrainedHFTokenizer) -> bool:
    if tokenizer.eos_token_id is None:
        return False

    return tokenizer.eos_token_id in sequence


def log_tokenizer(log: logging.Logger, tokenizer: PretrainedHFTokenizer) -> None:
    eos_token, bos_token, pad_token = "", "", ""

    if tokenizer.eos_token is not None:
        eos_token = f"({tokenizer.eos_token}) "

    if tokenizer.bos_token is not None:
        bos_token = f"({tokenizer.bos_token}) "

    if tokenizer.pad_token is not None:
        pad_token = f"({tokenizer.pad_token}) "

    s = (
        f"{tokenizer.__class__.__name__} | "
        f"Size: {len(tokenizer)} | "
        f"UNK: {tokenizer.unk_token_id} | "
        f"BOS: {tokenizer.bos_token_id} {bos_token}| "
        f"EOS: {tokenizer.eos_token_id} {eos_token}| "
        f"PAD: {tokenizer.pad_token_id} {pad_token}| "
        f"NUM_SPECIAL_TOKENS: {tokenizer.num_special_tokens_to_add()} | "
    )

    log.info(f"Tokenizer - {s}")
