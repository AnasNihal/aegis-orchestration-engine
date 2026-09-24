import json

import pytest

from aegis_engine.models import ChatMessage
from aegis_engine.web import INDEX_HTML, parse_history


def test_web_history_is_validated_and_converted() -> None:
    history = parse_history(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert history == (
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="hi"),
    )


@pytest.mark.parametrize(
    "value",
    [
        {"role": "user", "content": "not a list"},
        [{"role": "system", "content": "not accepted"}],
        [{"role": "user", "content": "x" * 32_001}],
    ],
)
def test_web_history_rejects_unsafe_shapes(value) -> None:
    with pytest.raises(ValueError):
        parse_history(value)


def test_web_page_contains_model_selector_and_chat_endpoint() -> None:
    assert 'id="model"' in INDEX_HTML
    assert "'/api/models'" in INDEX_HTML
    assert "'/api/chat'" in INDEX_HTML
