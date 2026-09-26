"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

Answer relevancy: is the answer actually about the question?

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low
faithfulness score is purely the generator's fault --- the context was
already correct.

    python -m evals.eval_generator
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.eval_retriever import GeminiJudge
from evals.harness import load_goldens, print_summary, summarize_by_metric
from src.generator import GENERATE_MODEL, generate

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = "gemini-3.5-flash-lite"
THRESHOLD = 0.7


def run():
    judge = GeminiJudge(JUDGE_MODEL)
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        context = [g["ideal_context"]]
        answer = generate(g["query"], context)
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=context,
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=2),
        error_config=ErrorConfig(ignore_errors=True),
        hyperparameters={
            "generator_model": GENERATE_MODEL,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "context": "ideal_context",
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_summary("generator", run())
