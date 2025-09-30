from miniseq.data import make_chat_prefix


def test_make_chat_prefix() -> None:
    # Test with a user message only.
    prefix = make_chat_prefix(user_message="Hello")
    assert prefix == [{"role": "user", "content": "Hello"}]

    # Test with a user message and a system message.
    prefix = make_chat_prefix(
        user_message="Hello", system_message="You are a helpful assistant."
    )
    assert prefix == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    # Test with a user message, system message, and assistant message.
    prefix = make_chat_prefix(
        user_message="Hello",
        system_message="You are a helpful assistant.",
        assistant_message="Hi there!",
    )
    assert prefix == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
