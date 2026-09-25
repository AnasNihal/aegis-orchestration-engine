"""Conservative deterministic selection of relevant safe tools."""

from __future__ import annotations

import re


_CALCULATOR = re.compile(r"\b(calculate|compute|work out|what is)\b|\d\s*[+\-*/%]\s*\d", re.IGNORECASE)
_TIME = re.compile(r"\b(time|date|today|tomorrow|yesterday|day of the week)\b", re.IGNORECASE)
_WORD_COUNT = re.compile(r"\b(count|number of)\b.*\b(words?|characters?|lines?)\b", re.IGNORECASE)
_READ_FILE = re.compile(r"\b(read|open|inspect|summarize)\b.*\b(file|document|text)\b", re.IGNORECASE)


def select_tools(request: str) -> tuple[str, ...]:
    """Select only obvious tools from the request.

    This is intentionally conservative. A false negative produces a normal
    model answer; a false positive should never grant a permission the caller
    did not configure.
    """

    if not request.strip():
        return ()
    selected: list[str] = []
    if _CALCULATOR.search(request):
        selected.append("calculator")
    if _TIME.search(request):
        selected.append("current_time")
    if _WORD_COUNT.search(request):
        selected.append("word_count")
    if _READ_FILE.search(request):
        selected.append("read_text_file")
    return tuple(selected)
