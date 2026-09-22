import pytest

from mmo_engine.config import ConfigurationError, Settings


def test_settings_defaults_are_local_and_explicit() -> None:
    configured = Settings()

    assert configured.ollama_base_url == "http://127.0.0.1:11434"
    assert configured.default_model == "qwen2.5:7b"
    assert configured.local_only is True


def test_settings_can_be_loaded_from_environment_mapping() -> None:
    configured = Settings.from_env(
        {
            "MMO_OLLAMA_BASE_URL": "http://localhost:11434/",
            "MMO_DEFAULT_MODEL": "deepseek-r1:8b",
            "MMO_REQUEST_TIMEOUT_SECONDS": "12.5",
            "MMO_LOCAL_ONLY": "yes",
        }
    )

    assert configured.ollama_base_url == "http://localhost:11434"
    assert configured.default_model == "deepseek-r1:8b"
    assert configured.request_timeout_seconds == 12.5
    assert configured.local_only is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("MMO_OLLAMA_BASE_URL", "not-a-url"),
        ("MMO_REQUEST_TIMEOUT_SECONDS", "0"),
        ("MMO_LOCAL_ONLY", "sometimes"),
    ],
)
def test_invalid_environment_values_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({field: value})

