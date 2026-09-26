# eval_retriever.py
import os
import re
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retriever import build_retriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH  = "goldens/retrievers_goldens.json"
JUDGE_MODEL  = "gemini-3.5-flash-lite"   # highest free-tier RPM available
THRESHOLD    = 0.7

# A backslash not starting a legal JSON escape ("\/bfnrtu).
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


# ── Gemini judge (DeepEval needs a DeepEvalBaseLLM, not a raw string) ─────────

class GeminiJudge(DeepEvalBaseLLM):
    """Thin wrapper so DeepEval can use Gemini instead of OpenAI as judge."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        self._model_name = model_name
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # Gemini sometimes returns content as a list of blocks, not a plain string.
    # DeepEval's trimAndLoadJson always expects a str — this normalises both.
    @staticmethod
    def _to_str(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _repair_escapes(text: str) -> str:
        """Double any backslash that isn't a legal JSON escape.

        GENKI chunks carry \\*, \\| and $\\rightarrow$. When the judge quotes
        those back inside a JSON string the payload stops being valid JSON.
        """
        return _BAD_ESCAPE.sub(r"\\\\", text)

    def _structured(self, schema, prompt: str):
        """Try Gemini structured output; None if the schema is unsupported."""
        try:
            return self._llm.with_structured_output(schema).invoke(prompt)
        except Exception:
            return None

    async def _a_structured(self, schema, prompt: str):
        try:
            return await self._llm.with_structured_output(schema).ainvoke(prompt)
        except Exception:
            return None

    def load_model(self):
        return self._llm

    def supports_structured_outputs(self) -> bool:
        return True

    # DeepEval passes `schema=` (a pydantic model) on every metric call. A real
    # instance lets it skip trimAndLoadJson entirely. Some DeepEval schemas use
    # $defs/$ref, which Gemini cannot resolve — those fall back to raw text with
    # the escapes repaired so the JSON still parses.
    def generate(self, prompt: str, schema=None):
        if schema is not None:
            result = self._structured(schema, prompt)
            if result is not None:
                return result
            return self._repair_escapes(self._to_str(self._llm.invoke(prompt).content))
        return self._to_str(self._llm.invoke(prompt).content)

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            result = await self._a_structured(schema, prompt)
            if result is not None:
                return result
            response = await self._llm.ainvoke(prompt)
            return self._repair_escapes(self._to_str(response.content))
        response = await self._llm.ainvoke(prompt)
        return self._to_str(response.content)

    def get_model_name(self) -> str:
        return self._model_name


# ── eval ──────────────────────────────────────────────────────────────────────

def run(retriever=None):
    if retriever is None:
        retriever = build_retriever()

    judge   = GeminiJudge()
    goldens = load_goldens(GOLDEN_PATH)

    # 1. LOAD the golden set --- the fixed, human-authored truth
    # 2. RUN THE RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved         = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["expected_output"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=2),   # free tier: 2 cases at a time stays under RPM limit
        error_config=ErrorConfig(ignore_errors=True),  # one rate-limited case must not sink the whole run
        hyperparameters={
            "retriever":       "basic_similarity",
            "embedding_model": "gemini-embedding-2",
            "chunk_size":      1000,
            "chunk_overlap":   150,
            "top_k":           5,
            "judge_model":     JUDGE_MODEL,
            "golden_set":      GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("retriever", run())
