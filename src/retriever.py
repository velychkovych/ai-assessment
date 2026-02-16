from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from src.ingest import COLLECTION_NAME, bm25_model, client
from src.models import Chunk

TOP_K = 5


def _compute_query_sparse(text: str) -> SparseVector:
    """Compute a BM25 sparse vector for a single query string."""
    embedding = next(bm25_model.query_embed(text))
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


def retrieve(query_vector: list[float], query_text: str, top_k: int = TOP_K) -> list[Chunk]:
    """Hybrid search using dense + BM25 sparse vectors with RRF fusion."""
    sparse_vec = _compute_query_sparse(query_text)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=query_vector, using="dense", limit=top_k * 2),
            Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [Chunk(**point.payload) for point in results.points]
