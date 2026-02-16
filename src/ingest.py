import asyncio
import re
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.models import Chunk

COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small

bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25", disable_stemmer=True)

MIN_CHUNK_LENGTH = 50  # Drop tiny fragments

# Matches markdown headings: # Title, ## Section, ### Subsection
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
    },
)


# ── Markdown chunker ──────────────────────────────────────────────────


def chunk_md(path: Path) -> list[Chunk]:
    """Split a markdown file into one chunk per section.

    Each ## heading starts a new chunk. The text under the top-level
    # heading (before the first ##) becomes the first chunk.
    """
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < MIN_CHUNK_LENGTH:
        return []

    source = path.name
    headings = list(MD_HEADING_RE.finditer(text))

    # Extract title from the first heading
    title = headings[0].group(2).strip() if headings else ""

    # Split into (section_name, section_text) pairs at each heading
    sections: list[tuple[str, str]] = []
    for i, match in enumerate(headings):
        section_name = match.group(2).strip()
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((section_name, text[start:end].strip()))

    chunks: list[Chunk] = []
    for idx, (section, section_text) in enumerate(sections):
        if len(section_text) < MIN_CHUNK_LENGTH:
            continue
        chunks.append(
            Chunk(
                id=f"{source}:{idx}",
                text=section_text,
                source=source,
                section=section,
                title=title,
                chunk_index=idx,
            )
        )

    return chunks


# ── Entry point ──────────────────────────────────────────────────────


def _compute_sparse(texts: list[str]) -> list[SparseVector]:
    """Compute BM25 sparse vectors for a list of texts."""
    results = []
    for embedding in bm25_model.embed(texts):
        results.append(
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return results


async def ingest(data_dir: Path, embed_fn) -> list[Chunk]:
    """Chunk all markdown documents in data_dir and upsert into Qdrant.

    Args:
        data_dir: Directory containing .md files.
        embed_fn: Async callable that takes a list of strings and returns a list of vectors.
    """
    loop = asyncio.get_running_loop()
    chunk_lists = await asyncio.gather(
        *[loop.run_in_executor(None, chunk_md, path) for path in sorted(data_dir.glob("*.md"))]
    )
    all_chunks: list[Chunk] = [c for chunks in chunk_lists for c in chunks]

    if not all_chunks:
        return all_chunks

    texts = [c.text for c in all_chunks]
    dense_vectors, sparse_vectors = await asyncio.gather(
        embed_fn(texts),
        loop.run_in_executor(None, _compute_sparse, texts),
    )

    points = [
        PointStruct(
            id=idx,
            vector={
                "dense": dense_vec,
                "sparse": sparse_vec,
            },
            payload=chunk.model_dump(),
        )
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(all_chunks, dense_vectors, sparse_vectors)
        )
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return all_chunks
