"""
export_db.py — export all stored chunks from the Chroma store to JSON.

Assumes src/retriever.py exposes load_store() (opens the persisted store
without rebuilding embeddings).

Run from the project root:
    python export_db.py

Produces goldens/chunks_dump.json — every chunk's id, text, and metadata —
so you can pick ideal_context for the faithfulness dataset.
"""

import json
from collections import Counter
from pathlib import Path

from src.retriever import PROJECT_ROOT, load_store

OUT_PATH = PROJECT_ROOT / "goldens" / "chunks_dump.json"


def export_chunks() -> list[dict]:
    store = load_store()
    data = store._collection.get(include=["documents", "metadatas"])
    dump = [
        {"id": i, "text": d, "meta": m}
        for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    dump.sort(
        key=lambda c: (
            str(c["meta"].get("section_type", "")),
            str(c["meta"].get("grammar_point", "")),
            c["id"],
        )
    )
    return dump


if __name__ == "__main__":
    dump = export_chunks()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Dumped {len(dump)} chunks to {OUT_PATH}")

    counts = Counter(
        (
            str(c["meta"].get("section_type", "?")),
            str(c["meta"].get("grammar_point") or c["meta"].get("pos") or "?"),
        )
        for c in dump
    )
    for section_type, key in sorted(counts):
        print(f"  {section_type} / {key}: {counts[(section_type, key)]} chunks")
