"""
evals/rag_pipeline_eval.py
==========================
End-to-end RAG triad on the live pipeline (retriever + generator).

Contextual relevancy: is the retrieved context about the question?
Faithfulness:        are the answer's claims supported by that retrieved context?
Answer relevancy:    does the answer address the question?

Unlike eval_generator.py, this does NOT inject ideal_context. A low
faithfulness score here can mean the generator hallucinated OR the
retriever handed it the wrong chunks.

    python -m evals.rag_pipeline_eval
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evals.eval_retriever import GeminiJudge
from evals.harness import load_goldens, print_summary, summarize_by_metric
from src.generator import GENERATE_MODEL, generate
from src.retriever import retrieve

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "gemini-3.5-flash-lite"
THRESHOLD = 0.7


def run():
    print(f"generator={GENERATE_MODEL}  judge={JUDGE_MODEL}")
    judge = GeminiJudge(JUDGE_MODEL)
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        docs = retrieve(g["query"])
        retrieval_context = [doc.page_content for doc in docs]
        answer = generate(g["query"], retrieval_context)
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=retrieval_context,
            )
        )

    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=1),
        error_config=ErrorConfig(ignore_errors=True),
        hyperparameters={
            "pipeline": "retrieve_then_generate",
            "generator_model": GENERATE_MODEL,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "context": "retrieved",
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_summary("rag triad", run())
