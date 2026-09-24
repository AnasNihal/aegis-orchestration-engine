"""Persistent local storage for Aegis execution state."""

from .sqlite import SQLiteTaskStateStore

__all__ = ["SQLiteTaskStateStore"]
