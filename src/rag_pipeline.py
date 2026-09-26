"""End-to-end RAG: retrieve lesson chunks, then generate a grounded answer.

    python -m src.rag_pipeline
"""

from __future__ import annotations

import re
import sys

from src.generator import generate
from src.retriever import retrieve


def _preview(text: str, n: int = 160) -> str:
    return re.sub(r"\s+", " ", text)[:n]


def rag_pipeline(question: str, verbose: bool = False) -> str:
    if verbose:
        print(f"Question: {question}\n")

    docs = retrieve(question)
    context = [doc.page_content for doc in docs]

    if verbose:
        print(f"Retrieved {len(docs)} chunk(s):")
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            chunk_id = meta.get("chunk_id", "?")
            title = meta.get("title", "")
            print(f"  {i}. [{chunk_id} | {title}] {_preview(doc.page_content)}")
        print()

    answer = generate(question, context)

    if verbose:
        print("Answer:")
        print(answer)
        print()

    return answer


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    question = "How do I form the potential form of a u-verb like 行く?"
    rag_pipeline(question, verbose=True)
