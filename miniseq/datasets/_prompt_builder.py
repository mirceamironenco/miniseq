from __future__ import annotations

from collections.abc import Mapping
from typing import Callable, Protocol, TypeVar

from typing_extensions import override

from miniseq.data import PretrainedHFTokenizer, make_chat_prefix

HFItemT = TypeVar("HFItemT", bound=Mapping)


class PromptBuilder(Protocol[HFItemT]):
    def __call__(
        self, document: HFItemT, *, tokenizer: PretrainedHFTokenizer | None
    ) -> tuple[HFItemT, str]: ...


def maybe_map_to_callable(key: str | Callable) -> Callable:
    if not isinstance(key, str):
        return key

    def _select_key(item: Mapping) -> str:
        return item[key]

    return _select_key


class ConcatenatePromptBuilder(PromptBuilder[HFItemT]):
    _prompt_key: Callable[[HFItemT], str]
    _system_message: str | None
    _assistant_message: str | None
    _prompt_transform: Callable[[str], str] | None

    def __init__(
        self,
        *,
        prompt_keymap: str | Callable[[HFItemT], str],
        system_message: str | None = None,
        assistant_message: str | None = None,
        prompt_transform: Callable[[str], str] | None = None,
    ) -> None:
        self._prompt_key = maybe_map_to_callable(prompt_keymap)
        self._system_message = system_message
        self._assistant_message = assistant_message
        self._prompt_transform = prompt_transform

    @override
    def __call__(
        self, document: HFItemT, *, tokenizer: PretrainedHFTokenizer | None
    ) -> tuple[HFItemT, str]:
        user_prompt = self._prompt_key(document)

        if self._prompt_transform is not None:
            user_prompt = self._prompt_transform(user_prompt)

        prompt = []

        if self._system_message is not None:
            prompt.append(self._system_message)

        prompt.append(user_prompt)

        if self._assistant_message is not None:
            prompt.append(self._assistant_message)

        # User should set separators directly if desired.
        # e.g. prompt can be "prompt_message\n" for '\n'
        prompt = "".join(prompt).strip()

        return document, prompt


class ChatTemplatePromptBuilder(PromptBuilder[HFItemT]):
    _prompt_key: Callable[[HFItemT], str]
    _system_message: str | None
    _assistant_message: str | None
    _prompt_transform: Callable[[str], str] | None

    def __init__(
        self,
        *,
        prompt_keymap: str | Callable[[HFItemT], str],
        system_message: str | None = None,
        assistant_message: str | None = None,
        prompt_transform: Callable[[str], str] | None = None,
    ) -> None:
        if system_message is not None:
            if not system_message:
                raise ValueError(
                    "Cannot have empty system_message, the template would add it anyway."
                )

        if assistant_message is not None:
            if not assistant_message:
                raise ValueError(
                    "Cannot have empty assistant_messaage, the template would add it anyway."
                )

        self._prompt_key = maybe_map_to_callable(prompt_keymap)
        self._system_message = system_message
        self._assistant_message = assistant_message
        self._prompt_transform = prompt_transform

    @override
    def __call__(
        self, document: HFItemT, *, tokenizer: PretrainedHFTokenizer | None
    ) -> tuple[HFItemT, str]:
        if tokenizer is None:
            raise ValueError(
                "ChatTemplatePromptBuilder requries tokenizer to be passed in."
            )

        user_prompt = self._prompt_key(document)

        if self._prompt_transform is not None:
            user_prompt = self._prompt_transform(user_prompt)

        prefix = make_chat_prefix(
            user_message=user_prompt,
            system_message=self._system_message,
            assistant_message=self._assistant_message,
        )

        has_assistant_start = self._assistant_message is not None

        prompt = tokenizer.apply_chat_template(
            prefix,
            tokenize=False,
            add_generation_prompt=not has_assistant_start,
            continue_final_message=has_assistant_start,
        )

        return document, prompt
