from unittest.mock import Mock

import pytest

from miniseq.datasets import ChatTemplatePromptBuilder, ConcatenatePromptBuilder


def test_concatenate_prompt_builder():
    builder = ConcatenatePromptBuilder(prompt_keymap="text")
    doc = {"text": "Hello world"}
    _, prompt = builder(doc, tokenizer=None)
    assert prompt == "Hello world"


def test_concatenate_prompt_builder_with_system_message():
    builder = ConcatenatePromptBuilder(prompt_keymap="text", system_message="System: ")
    doc = {"text": "Hello world"}
    _, prompt = builder(doc, tokenizer=None)
    assert prompt == "System: Hello world"


def test_concatenate_prompt_builder_with_assistant_message():
    builder = ConcatenatePromptBuilder(
        prompt_keymap="text", assistant_message="Assistant: "
    )
    doc = {"text": "User: Hello"}
    _, prompt = builder(doc, tokenizer=None)
    assert prompt == "User: HelloAssistant:"


def test_concatenate_prompt_builder_with_transform():
    builder = ConcatenatePromptBuilder(
        prompt_keymap="text", prompt_transform=lambda x: x.upper()
    )
    doc = {"text": "Hello world"}
    _, prompt = builder(doc, tokenizer=None)
    assert prompt == "HELLO WORLD"


def test_chat_template_prompt_builder_requires_tokenizer():
    builder = ChatTemplatePromptBuilder(prompt_keymap="text")
    doc = {"text": "Hello world"}
    with pytest.raises(
        ValueError, match="ChatTemplatePromptBuilder requries tokenizer"
    ):
        builder(doc, tokenizer=None)


def test_chat_template_prompt_builder():
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<|im_start|>user\nHello world<|im_end|>"
    )

    builder = ChatTemplatePromptBuilder(prompt_keymap="text")
    doc = {"text": "Hello world"}
    _, prompt = builder(doc, tokenizer=mock_tokenizer)

    assert prompt == "<|im_start|>user\nHello world<|im_end|>"
    mock_tokenizer.apply_chat_template.assert_called_once()
