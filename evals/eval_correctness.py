"""
evals/eval_correctness.py
=========================
End-to-end correctness via GEval: compare live RAG answers to reference answers.

    python -m evals.eval_correctness
"""

from __future__ import annotations

import sys
from typing import Callable

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evals.eval_retriever import GeminiJudge
from evals.harness import load_goldens, print_summary, summarize_by_metric
from src.rag_pipeline import rag_pipeline

load_dotenv()

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL = "gemini-3.5-flash-lite"
THRESHOLD = 0.7


def run(answer_fn: Callable[[str], str] | None = None):
    if answer_fn is None:
        answer_fn = rag_pipeline

    judge = GeminiJudge(JUDGE_MODEL)
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []
    for g in goldens:
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer_fn(g["query"]),
                expected_output=g["expected_output"],
            )
        )

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points — omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine.",
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Mostly correct but one minor inaccuracy.",
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="Contains a clear factual error or a claim that contradicts the expected output.",
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT, 
        LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    result = evaluate(
        test_cases=test_cases,
        metrics=[correctness, completeness],
        async_config=AsyncConfig(max_concurrent=1),
        error_config=ErrorConfig(ignore_errors=True),
        hyperparameters={
            "golden_set": GOLDEN_PATH,
            "judge_model": JUDGE_MODEL,
            "pipeline": "rag_pipeline",
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print_summary("correctness", run())
