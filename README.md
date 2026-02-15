# RAG-LLM

A Retrieval-Augmented Generation system that answers questions about machine learning research papers using hybrid vector search and structured LLM outputs.

## How to Run

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the pipeline (ingest + ask sample questions)
uv run python run.py

# Run the evaluation suite
uv run pytest
```

## Design Decisions

### Model: GPT-5-mini

GPT-5-mini was chosen as the LLM for both the RAG agent and the faithfulness judge. It provides a great balance between language understanding and cost — capable enough to follow structured output schemas, ground answers in retrieved context, and assess faithfulness, while keeping API costs low for a system that makes multiple LLM calls per question (retrieval tool call, answer generation, and evaluation judging).

### Framework: PydanticAI

PydanticAI was selected because the assessment suggested it and I had no strong preference for an alternative. It turned out to be a good fit: it provides typed tool definitions, structured output validation via Pydantic models, and a clean agent abstraction that made it straightforward to wire up the retrieve-then-answer flow with dependency injection.

### Vector Store Documents

The vector store contains abstracts and key details from 5 popular machine learning papers:

- **Adam** (Kingma & Ba, 2015) — the dominant optimizer in deep learning
- **Attention Is All You Need** (Vaswani et al., 2017) — the Transformer architecture
- **BERT** (Devlin et al., 2019) — bidirectional pre-training for NLP
- **ImageNet Classification with Deep CNNs** (Krizhevsky et al., 2012) — AlexNet and the deep learning revolution
- **LoRA** (Hu et al., 2021) — parameter-efficient fine-tuning

Each file is a markdown document with the paper's abstract followed by 3 sections covering the method, key results, and impact.

## Chunking Strategy

Each markdown file is split into chunks at heading boundaries — one chunk per `#` or `##` section. This means each paper produces 4 chunks: the abstract (under the `#` title) and 3 detail sections (each under a `##` heading).

This approach works well for this corpus because:

- **Each section is a self-contained topic.** The abstract summarizes the paper, while each `##` section covers a distinct aspect (algorithm details, results, impact). Splitting at headings preserves these natural semantic boundaries.
- **Sections are small enough for a single embedding.** Each chunk is 400-900 characters — well within the token limit of `text-embedding-3-small` and small enough for the embedding to capture the full meaning without dilution.
- **No overlap needed.** Since sections don't share content across boundaries, there is no risk of cutting a sentence or idea in half, which eliminates the need for chunk overlap.
- **Title and source metadata travel with every chunk.** Each chunk carries the paper title, source filename, and section name, so the LLM always knows which paper a chunk came from.

## Hybrid Search

Retrieval uses hybrid search combining dense and sparse vectors, fused with Reciprocal Rank Fusion (RRF) in Qdrant:

- **Dense vectors** (`text-embedding-3-small`) capture semantic similarity — they understand that "learning rate adaptation" is related to "Adam optimizer" even without exact keyword overlap.
- **Sparse vectors** (BM25 via `fastembed`) capture exact keyword matches — they ensure that a query mentioning "BLEU score" or "GPT-3 175B" finds chunks containing those exact terms.

This combination matters because research paper questions often mix both styles: a user might ask a conceptual question ("how does the Transformer differ from RNNs?") that benefits from semantic search, or a specific factual query ("what BLEU score on WMT 2014?") where keyword matching is critical. RRF merges the ranked results from both retrieval paths without requiring tuned weights, giving robust results across both query types.

## Evaluation Thresholds

The evaluation suite (`eval/evaluate.py`) tests three metrics across question confidence categories. Each threshold is set to catch real regressions while accounting for the inherent variability of LLM outputs:

### Recall (high >= 0.8, medium >= 0.6)

Recall measures what fraction of expected source sections appear in the retrieved chunks. The high threshold of 0.8 ensures the retriever consistently finds the right section for straightforward factual questions. Medium questions often span multiple sections or require cross-paper retrieval, so the threshold is relaxed to 0.6 — finding at least some relevant context is sufficient for the LLM to synthesize an answer.

### Precision (high >= 0.2)

Precision measures what fraction of retrieved chunks match an expected section. The threshold is deliberately low at 0.2 because the retriever returns `TOP_K=5` chunks per query, but most questions only have 1-2 relevant sections in a corpus of 20 chunks. Even with perfect retrieval, average precision tops out around 0.33 for this dataset. The 0.2 threshold catches cases where the retriever returns entirely irrelevant chunks while accepting the expected noise from a small corpus.

### Faithfulness (high >= 0.8, medium >= 0.6, low >= 0.6)

Faithfulness uses an LLM-as-a-judge to verify that every claim in the answer is supported by the retrieved context. This catches hallucination — the most dangerous failure mode in RAG systems. High-confidence questions get a strict 0.8 threshold because the context directly contains the answer, so there is no excuse for unsupported claims. Medium and low questions use 0.6 because the LLM may need to acknowledge gaps or synthesize partial information, which the judge can sometimes flag as unfaithful even when the answer is reasonable.
