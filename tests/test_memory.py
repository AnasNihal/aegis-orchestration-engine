from aegis_engine.memory import SQLiteMemoryStore


def test_memory_store_saves_searches_and_deletes_explicit_notes(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    note = store.save("Project goal", "Build a local orchestration engine.")

    assert store.get(note.note_id) == note
    assert store.search("orchestration") == (note,)

    store.delete(note.note_id)
    assert store.search("orchestration") == ()
