"""Generator — answer a student question from given lesson context only.

    from src.generator import generate
    answer = generate(query, ["chunk text 1", "chunk text 2"])

    python src/generator.py          # smoke test on faithfulness item f01
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GENERATE_MODEL = "gemini-3.5-flash-lite"
ABSTAIN = "I don't have enough information in the lesson material to answer that."

_PROMPT = ChatPromptTemplate.from_template(
    """You are Genki Mate, a patient Japanese tutor for the GENKI textbook.
The lesson context below is the only material you may use — it may come from
any GENKI lesson.

Answer the student using ONLY that context. If the context names a lesson or
grammar point, stay within it. Do not pull in other lessons from memory.

Rules:
- Ground every claim in the context. Do not add grammar, examples, or forms
  that are not in the context.
- Explain clearly, the way a teacher would speak. Prefer short paragraphs over
  bullet lists. Keep Japanese examples from the context when they help.
- If the question has more than one part, cover each part.
- Stop when the question is answered. Do not dump the whole context.
- If the context is not enough to answer, reply with exactly:
  {abstain}

<LESSON_CONTEXT>
{context}
</LESSON_CONTEXT>

<STUDENT_QUESTION>
{question}
</STUDENT_QUESTION>

Answer:"""
)


def _llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing — add it to .env.")
    return ChatGoogleGenerativeAI(
        model=GENERATE_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


def generate(query: str, context: list[str]) -> str:
    """Return a grounded answer from the query and context chunks."""
    chain = _PROMPT | _llm() | StrOutputParser()
    return chain.invoke(
        {
            "question": query,
            "context": "\n\n".join(context),
            "abstain": ABSTAIN,
        }
    )


if __name__ == "__main__":
    import json

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    goldens = json.loads(
        (PROJECT_ROOT / "goldens" / "faithfulness_dataset.json").read_text(
            encoding="utf-8"
        )
    )
    item = goldens[0]
    print(f"Query: {item['query']}\n")
    print(generate(item["query"], [item["ideal_context"]]))
