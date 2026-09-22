import pytest

from aegis_engine.config import ConfigurationError, Settings


def test_settings_defaults_are_local_and_explicit() -> None:
    configured = Settings()

    assert configured.ollama_base_url == "http://127.0.0.1:11434"
    assert configured.default_model == "qwen2.5:7b"
    assert configured.local_only is True


def test_settings_can_be_loaded_from_environment_mapping() -> None:
    configured = Settings.from_env(
        {
            "AEGIS_OLLAMA_BASE_URL": "http://localhost:11434/",
            "AEGIS_DEFAULT_MODEL": "deepseek-r1:8b",
            "AEGIS_REQUEST_TIMEOUT_SECONDS": "12.5",
            "AEGIS_LOCAL_ONLY": "yes",
        }
    )

    assert configured.ollama_base_url == "http://localhost:11434"
    assert configured.default_model == "deepseek-r1:8b"
    assert configured.request_timeout_seconds == 12.5
    assert configured.local_only is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("AEGIS_OLLAMA_BASE_URL", "not-a-url"),
        ("AEGIS_REQUEST_TIMEOUT_SECONDS", "0"),
        ("AEGIS_LOCAL_ONLY", "sometimes"),
    ],
)
def test_invalid_environment_values_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({field: value})

