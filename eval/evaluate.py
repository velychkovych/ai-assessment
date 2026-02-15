import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent

load_dotenv()

from src.agent import ask
from src.ingest import ingest
from src.models import FaithfulnessVerdict, RAGResponse

EVAL_DATASET = Path(__file__).parent / "dataset.json"
DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDING_MODEL = "text-embedding-3-small"

# ── LLM-as-a-judge for faithfulness ──────────────────────────────────

faithfulness_judge = Agent(
    model=None,
    output_type=FaithfulnessVerdict,
    system_prompt=(
        "You are an impartial judge evaluating whether an answer is faithful to the provided context.\n\n"
        "Faithful means:\n"
        "- Every factual claim in the answer is supported by the context chunks.\n"
        "- Minor rephrasings and reasonable rounding of numbers are acceptable.\n"
        "- An answer that correctly states the context is insufficient or doesn't cover the topic IS faithful.\n\n"
        "Not faithful means:\n"
        "- The answer includes facts, numbers, or claims not present in the context.\n"
        "- The answer contradicts what the context states."
    ),
)


async def judge_faithfulness(
    question: str,
    answer: str,
    context_chunks: list[str],
) -> FaithfulnessVerdict:
    """Use an LLM to judge whether the answer is faithful to the retrieved context."""
    context = "\n---\n".join(context_chunks)
    prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Judge whether the answer is faithful to the context above. If not faithful, identify the specific unsupported claim in your reason."
    )
    result = await faithfulness_judge.run(prompt, model="openai:gpt-5-mini")
    return result.output


# ── Chunk-level retrieval metrics ────────────────────────────────────

ChunkRef = tuple[str, str]  # (source, section)


def _to_chunk_refs(expected_chunks: list[dict]) -> set[ChunkRef]:
    """Convert dataset expected_chunks list to a set of (source, section) tuples."""
    return {(c["source"], c["section"]) for c in expected_chunks}


def _retrieved_chunk_refs(sources) -> list[ChunkRef]:
    """Extract (source, section) pairs from response sources."""
    return [(s.source, s.section) for s in sources]


def precision(retrieved: list[ChunkRef], expected: set[ChunkRef]) -> float:
    """Fraction of retrieved chunks whose (source, section) matches any expected chunk."""
    if not retrieved or not expected:
        return 0.0
    relevant = sum(1 for r in retrieved if r in expected)
    return relevant / len(retrieved)


def recall(retrieved: list[ChunkRef], expected: set[ChunkRef]) -> float:
    """Fraction of expected (source, section) pairs found in the retrieved chunks."""
    if not expected:
        return 0.0
    retrieved_set = set(retrieved)
    found = sum(1 for e in expected if e in retrieved_set)
    return found / len(expected)


# ── Evaluated result for a single question ───────────────────────────


@dataclass
class EvalResult:
    pair: dict
    response: RAGResponse
    faithful: bool


# ── Fixtures ─────────────────────────────────────────────────────────


def _load_dataset() -> list[dict]:
    return json.loads(EVAL_DATASET.read_text())


@pytest.fixture(scope="session")
def dataset() -> list[dict]:
    return _load_dataset()


@pytest_asyncio.fixture(scope="session")
async def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


@pytest_asyncio.fixture(scope="session")
async def ingested(openai_client: AsyncOpenAI) -> None:
    """Ingest documents once for the entire test session."""
    async def embed(texts: list[str]) -> list[list[float]]:
        resp = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    await ingest(DATA_DIR, embed)


def _get_pairs_by_category(dataset: list[dict], category: str) -> list[dict]:
    return [p for p in dataset if p.get("confidence") == category]


async def _evaluate_single(pair: dict, openai_client: AsyncOpenAI) -> EvalResult:
    """Evaluate a single question: ask, then judge faithfulness."""
    response = await ask(pair["question"], openai_client)
    context_chunks = [s.text for s in response.sources]
    verdict = await judge_faithfulness(pair["question"], response.answer, context_chunks)
    return EvalResult(pair=pair, response=response, faithful=verdict.faithful)


async def _evaluate_pairs(pairs: list[dict], openai_client: AsyncOpenAI) -> list[EvalResult]:
    """Ask all questions concurrently, judge faithfulness, and return collected results."""
    return list(await asyncio.gather(*[_evaluate_single(p, openai_client) for p in pairs]))


# ── Per-category fixtures that run all questions once ────────────────


@pytest_asyncio.fixture(scope="session")
async def all_results(
    dataset: list[dict], openai_client: AsyncOpenAI, ingested: None,
) -> dict[str, list[EvalResult]]:
    """Evaluate all categories concurrently."""
    high, medium, low = await asyncio.gather(
        _evaluate_pairs(_get_pairs_by_category(dataset, "high"), openai_client),
        _evaluate_pairs(_get_pairs_by_category(dataset, "medium"), openai_client),
        _evaluate_pairs(_get_pairs_by_category(dataset, "low"), openai_client),
    )
    return {"high": high, "medium": medium, "low": low}


@pytest.fixture(scope="session")
def high_results(all_results: dict[str, list[EvalResult]]) -> list[EvalResult]:
    return all_results["high"]


@pytest.fixture(scope="session")
def medium_results(all_results: dict[str, list[EvalResult]]) -> list[EvalResult]:
    return all_results["medium"]


@pytest.fixture(scope="session")
def low_results(all_results: dict[str, list[EvalResult]]) -> list[EvalResult]:
    return all_results["low"]


# ── Console report ────────────────────────────────────────────────────


def _print_report(all_results: dict[str, list[EvalResult]]) -> None:
    """Print a detailed console report of evaluation results."""
    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)

    for category in ("high", "medium", "low"):
        results = all_results.get(category, [])
        if not results:
            continue

        print(f"\n{'─' * 70}")
        print(f"  {category.upper()} CONFIDENCE ({len(results)} questions)")
        print(f"{'─' * 70}")

        for r in results:
            q = r.pair["question"]
            conf = r.response.confidence.value
            faith = "faithful" if r.faithful else "NOT faithful"
            retrieved = _retrieved_chunk_refs(r.response.sources)
            expected = _to_chunk_refs(r.pair["expected_chunks"])
            rec = recall(retrieved, expected) if expected else None
            prec = precision(retrieved, expected) if expected else None

            print(f"\n  Q: {q}")
            print(f"  A: {r.response.answer[:150]}{'...' if len(r.response.answer) > 150 else ''}")
            print(f"  Confidence: {conf} | Faithfulness: {faith}")
            if rec is not None:
                print(f"  Recall: {rec:.2f} | Precision: {prec:.2f}")
            print(f"  Retrieved: {[f'{s}:{sec}' for s, sec in retrieved]}")

        # Category summary
        rec_scores = [
            recall(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in results if r.pair["expected_chunks"]
        ]
        prec_scores = [
            precision(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in results if r.pair["expected_chunks"]
        ]
        faith_rate = sum(1 for r in results if r.faithful) / len(results)

        print(f"\n  Summary:")
        if rec_scores:
            print(f"    Avg recall:      {sum(rec_scores) / len(rec_scores):.2f}")
        if prec_scores:
            print(f"    Avg precision:   {sum(prec_scores) / len(prec_scores):.2f}")
        print(f"    Faithfulness:    {faith_rate:.0%} ({sum(1 for r in results if r.faithful)}/{len(results)})")

    print(f"\n{'=' * 70}\n")


@pytest.fixture(scope="session", autouse=True)
def print_report(all_results: dict[str, list[EvalResult]]) -> None:
    """Print the evaluation report once all results are collected."""
    _print_report(all_results)


# ── High-confidence questions: answer exists directly in the corpus ──


class TestHighConfidence:
    """Questions whose answers are directly stated in the corpus."""

    @pytest.mark.asyncio
    async def test_recall(self, high_results: list[EvalResult]):
        """At least one expected (source, section) should be retrieved for every high-confidence question."""
        scores = [
            recall(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in high_results
        ]
        avg = sum(scores) / len(scores)
        assert avg >= 0.8, f"High recall {avg:.2f} is below 0.8 threshold"

    @pytest.mark.asyncio
    async def test_precision(self, high_results: list[EvalResult]):
        """Most retrieved chunks should match an expected (source, section)."""
        scores = [
            precision(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in high_results
        ]
        avg = sum(scores) / len(scores)
        assert avg >= 0.2, f"High precision {avg:.2f} is below 0.2 threshold"

    @pytest.mark.asyncio
    async def test_faithfulness(self, high_results: list[EvalResult]):
        """Answers to high-confidence questions should be faithful to retrieved context."""
        rate = sum(1 for r in high_results if r.faithful) / len(high_results)
        assert rate >= 0.8, f"High faithfulness {rate:.2f} is below 0.8 threshold"

    @pytest.mark.asyncio
    async def test_confidence_is_high(self, high_results: list[EvalResult]):
        """High-confidence questions should produce high confidence answers."""
        rate = sum(1 for r in high_results if r.response.confidence.value == "high") / len(high_results)
        assert rate >= 0.6, f"Only {rate:.0%} of high answers had high confidence"


# ── Medium-confidence questions: topic exists but no direct answer ───


class TestMediumConfidence:
    """Questions about topics in the corpus but without a direct answer."""

    @pytest.mark.asyncio
    async def test_recall(self, medium_results: list[EvalResult]):
        """Should retrieve chunks from a relevant (source, section)."""
        scores = [
            recall(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in medium_results
        ]
        avg = sum(scores) / len(scores)
        assert avg >= 0.6, f"Medium recall {avg:.2f} is below 0.6 threshold"

    @pytest.mark.asyncio
    async def test_precision(self, medium_results: list[EvalResult]):
        """Retrieved chunks should include at least some expected (source, section) pairs."""
        scores = [
            precision(_retrieved_chunk_refs(r.response.sources), _to_chunk_refs(r.pair["expected_chunks"]))
            for r in medium_results
        ]
        avg = sum(scores) / len(scores)
        assert avg >= 0.2, f"Medium precision {avg:.2f} is below 0.2 threshold"

    @pytest.mark.asyncio
    async def test_faithfulness(self, medium_results: list[EvalResult]):
        """Answers should be grounded in context, even if acknowledging gaps."""
        rate = sum(1 for r in medium_results if r.faithful) / len(medium_results)
        assert rate >= 0.6, f"Medium faithfulness {rate:.2f} is below 0.6 threshold"


# ── Low-confidence questions: cross-paper or not covered ─────────────


class TestLowConfidence:
    """Questions that span multiple papers or are hard to answer from the corpus."""

    @pytest.mark.asyncio
    async def test_confidence_is_low(self, low_results: list[EvalResult]):
        """Low-confidence questions should produce low confidence answers."""
        rate = sum(1 for r in low_results if r.response.confidence.value == "low") / len(low_results)
        assert rate >= 0.6, f"Only {rate:.0%} of low answers had low confidence"

    @pytest.mark.asyncio
    async def test_faithfulness(self, low_results: list[EvalResult]):
        """Answers should still be faithful (not hallucinate beyond retrieved context)."""
        rate = sum(1 for r in low_results if r.faithful) / len(low_results)
        assert rate >= 0.6, f"Low faithfulness {rate:.2f} is below 0.6 threshold"
