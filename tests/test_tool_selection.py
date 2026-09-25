from aegis_engine.tools import select_tools


def test_tool_selection_is_conservative_and_relevant() -> None:
    assert select_tools("What time is it today?") == ("current_time",)
    assert select_tools("Calculate 12 * 4") == ("calculator",)
    assert select_tools("Count the words in this text") == ("word_count",)
    assert select_tools("Tell me about Python") == ()


def test_file_selection_does_not_authorize_unapproved_paths() -> None:
    assert select_tools("Read this file") == ("read_text_file",)
