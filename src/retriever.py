"""Lesson 13 retriever — load → chunk → embed once → persist in Chroma.

Usage:
    python src/retriever.py --chunks          # inspect chunks (no API key needed)
    python src/retriever.py                   # run a test query (needs GOOGLE_API_KEY)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── paths & constants ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_PATH     = PROJECT_ROOT / "output" / "cleaned_lesson13.md"
DB_DIR        = PROJECT_ROOT / "chroma_store" / "lesson13"
LESSON        = 13
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
DEFAULT_K     = 5
EMBED_MODEL   = "gemini-embedding-2"

# ── heading → section classification ────────────────────────────────────────
# Each rule: (regex, section_type, grammar_point_or_pos, grammar_number, title)
# grammar_point used for grammar; pos used for vocab; both "" when unused.

_RULES: list[tuple[re.Pattern, str, str, str, str]] = [
    # grammar
    (re.compile(r"potential",     re.I), "grammar", "potential",        "1", "Potential Verbs"),
    (re.compile(r"〜し|\bし$"),           "grammar", "shi",               "2", "〜し"),
    (re.compile(r"そうです|looks like", re.I), "grammar", "sou_desu",    "3", "〜そうです"),
    (re.compile(r"てみる"),               "grammar", "te_miru",          "4", "〜てみる"),
    (re.compile(r"なら"),                 "grammar", "nara",             "5", "なら"),
    (re.compile(r"一週間|三回|frequency", re.I), "grammar", "frequency", "6", "一週間に三回"),
    (re.compile(r"expression|表現", re.I), "grammar", "expression_notes", "", "Expression Notes"),
    # vocab
    (re.compile(r"な-adjective",  re.I), "vocab", "na-adjective",  "", "な-adjectives"),
    (re.compile(r"u-verb",        re.I), "vocab", "u-verb",        "", "U-verbs"),
    (re.compile(r"irregular",     re.I), "vocab", "irregular-verb","", "Irregular Verb"),
    (re.compile(r"adverb",        re.I), "vocab", "adverb",        "", "Adverbs and Other Expressions"),
    (re.compile(r"number",        re.I), "vocab", "counter",       "", "Numbers (used to count days)"),
    (re.compile(r"単語|vocab",    re.I), "vocab", "noun",          "", "Vocabulary"),
]

_HEADING_RE  = re.compile(r"^(?:#{1,3}\s+\S.*|logo:\s+\d+\s+\S.*|\*\*Expression Notes\*\*|logo:\s*表現ノート)$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SLUG_RE     = re.compile(r"[^a-z0-9]+")


def _plain(heading: str) -> str:
    """Strip markdown / HTML / logo prefixes from a heading line."""
    t = re.sub(r"^#+\s*|^logo:\s*", "", heading.strip())
    return re.sub(r"\s+", " ", _HTML_TAG_RE.sub("", t).replace("**", "")).strip()


def _classify(heading: str) -> dict:
    plain = _plain(heading)
    for pattern, stype, key, num, title in _RULES:
        if pattern.search(plain):
            return {"section_type": stype, "grammar_point": key if stype == "grammar" else "",
                    "pos": key if stype == "vocab" else "", "grammar_number": num, "title": title}
    if re.search(r"grammar|文法", plain, re.I):
        return {"section_type": "grammar", "grammar_point": "", "pos": "", "grammar_number": "", "title": "Grammar"}
    if re.search(r"会話|dialogue|conversation", plain, re.I):
        return {"section_type": "dialogue", "grammar_point": "", "pos": "", "grammar_number": "", "title": plain}
    return     {"section_type": "other",   "grammar_point": "", "pos": "", "grammar_number": "", "title": plain or "Untitled"}


def _slug(meta: dict) -> str:
    key = meta["grammar_point"] or meta["pos"] or meta["title"] or meta["section_type"]
    return _SLUG_RE.sub("-", key.lower()).strip("-") or "chunk"

# ── document loading & chunking ──────────────────────────────────────────────

def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on recognized headings; return [(heading, body), ...]."""
    lines  = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if _HEADING_RE.match(ln.strip())]
    if not starts:
        return [("Lesson 13", text.strip())]

    sections: list[tuple[str, str]] = []
    for idx, start in enumerate(starts):
        end  = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        sections.append((lines[start].strip(), body))
    return sections


def _drop_stubs(sections: list[tuple[str, str]], min_chars: int = 40) -> list[tuple[str, str]]:
    """Merge nearly-empty stub headings into the following section."""
    out: list[tuple[str, str]] = []
    carry = ""
    for heading, body in sections:
        combined = f"{carry}\n\n{body}".strip() if carry else body
        if len(combined) < min_chars:
            carry = f"{heading}\n{combined}".strip()
            continue
        out.append((heading, combined))
        carry = ""
    if carry:
        if out:
            h, b = out[-1]; out[-1] = (h, f"{b}\n\n{carry}".strip())
        else:
            out.append(("Untitled", carry))
    return out


def load_lesson13() -> list[Document]:
    """Parse cleaned Lesson 13 markdown into labelled, eval-friendly chunks."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run main.py first.")

    text     = DATA_PATH.read_text(encoding="utf-8")
    sections = _drop_stubs(_split_into_sections(text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    docs: list[Document] = []
    slug_counts: dict[str, int] = {}

    for heading, body in sections:
        if not body.strip():
            continue
        meta  = _classify(heading)
        slug  = _slug(meta)
        base  = {"lesson": LESSON, "source": "cleaned_lesson13.md", **meta}

        pieces = splitter.split_documents([Document(
            page_content=f"{_plain(heading)}\n\n{body}".strip(),
            metadata=base,
        )])

        for piece in pieces:
            slug_counts[slug] = slug_counts.get(slug, 0) + 1
            piece.metadata = {**base, "chunk_id": f"l13-{meta['section_type']}-{slug}-{slug_counts[slug]:02d}"}
            docs.append(piece)

    if not docs:
        raise ValueError(f"No chunks produced from {DATA_PATH}")
    return docs

# ── vector store & retriever ─────────────────────────────────────────────────

def _make_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing — add it to .env.\n"
            "To inspect chunks without an API key run:  python src/retriever.py --chunks"
        )
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=api_key)


def load_store() -> Chroma:
    """Embed once, reuse on every subsequent run."""
    embeddings = _make_embeddings()
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    chunks = load_lesson13()
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        chunks, embeddings,
        persist_directory=str(DB_DIR),
        ids=[c.metadata["chunk_id"] for c in chunks],
    )


def build_retriever(k: int = DEFAULT_K):
    return load_store().as_retriever(search_kwargs={"k": k})

# ── CLI helpers ──────────────────────────────────────────────────────────────

def _preview(text: str, n: int = 160) -> str:
    return re.sub(r"\s+", " ", text)[:n]


def print_chunks() -> None:
    chunks = load_lesson13()
    print(f"{len(chunks)} chunks from {DATA_PATH.name}\n")
    for doc in chunks:
        m = doc.metadata
        print(f"[{m['chunk_id']} | {m['section_type']} | {m['title']}] {_preview(doc.page_content)}...\n")


if __name__ == "__main__":
    # Force UTF-8 output so Japanese characters print on Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if "--chunks" in sys.argv:
        print_chunks()
    else:
        retriever = build_retriever()
        query     = "How do I make the potential form of an u-verb?"
        print(f"Query: {query}\n")
        for doc in retriever.invoke(query):
            m = doc.metadata
            print(f"[{m['chunk_id']} | {m['section_type']} | {m['title']}] {_preview(doc.page_content)}...\n")
