"""Small SQLite store for explicitly saved notes.

Notes are never created implicitly by the orchestrator. They are separate from
task state and can be searched or deleted explicitly by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryNote:
    note_id: str
    title: str
    content: str
    created_at: str
    updated_at: str


class SQLiteMemoryStore:
    """Explicit CRUD and simple text search for local notes."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute(
                """CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.commit()
            if self._connection is None:
                connection.close()

    def save(self, title: str, content: str, *, note_id: str | None = None) -> MemoryNote:
        if not title.strip() or not content.strip():
            raise ValueError("note title and content must not be empty")
        now = _now()
        note = MemoryNote(note_id or str(uuid4()), title.strip(), content, now, now)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    "INSERT INTO notes(note_id, title, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (note.note_id, note.title, note.content, note.created_at, note.updated_at),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError(f"note already exists: {note.note_id}") from exc
            finally:
                if self._connection is None:
                    connection.close()
        return note

    def get(self, note_id: str) -> MemoryNote:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute("SELECT * FROM notes WHERE note_id=?", (note_id,)).fetchone()
            finally:
                if self._connection is None:
                    connection.close()
        if row is None:
            raise KeyError(f"note does not exist: {note_id}")
        return _row_to_note(row)

    def search(self, query: str, *, limit: int = 20) -> tuple[MemoryNote, ...]:
        if not query.strip():
            raise ValueError("search query must not be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        pattern = f"%{query.strip()}%"
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, limit),
                ).fetchall()
            finally:
                if self._connection is None:
                    connection.close()
        return tuple(_row_to_note(row) for row in rows)

    def delete(self, note_id: str) -> None:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute("DELETE FROM notes WHERE note_id=?", (note_id,))
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise KeyError(f"note does not exist: {note_id}")
                connection.commit()
            finally:
                if self._connection is None:
                    connection.close()


def _row_to_note(row: sqlite3.Row) -> MemoryNote:
    return MemoryNote(
        note_id=row["note_id"],
        title=row["title"],
        content=row["content"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
