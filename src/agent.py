from dataclasses import dataclass, field

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from src.models import AgentOutput, Chunk, RAGResponse, SourceReference
from src.retriever import retrieve

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class RAGDeps:
    openai_client: AsyncOpenAI
    retrieved_chunks: list[Chunk] = field(default_factory=list)


rag_agent = Agent(
    model=None,
    output_type=AgentOutput,
    deps_type=RAGDeps,
    system_prompt=(
        "You are a research paper Q&A assistant. You MUST call the retrieve_context tool before answering.\n\n"
        "Rules:\n"
        "- Base your answer ONLY on the retrieved context chunks. Never use prior knowledge.\n"
        "- When chunks come from multiple papers, reference each paper by its title.\n"
        "- Keep answers concise (2-4 sentences) and cite specific details from the context.\n\n"
        "Confidence levels:\n"
        '- "high": The context contains a clear, direct answer with specific numbers or definitions.\n'
        '- "medium": The context covers the topic but the answer requires inference or synthesis across chunks.\n'
        '- "low": The context is missing, tangential, or insufficient to answer the question. Say what is missing.'
    ),
)


@rag_agent.tool
async def retrieve_context(ctx: RunContext[RAGDeps], query: str) -> list[dict]:
    """Search the paper corpus for relevant chunks. Include specific terms, paper names, or technical keywords in the query for best results — the search uses both semantic and keyword matching.

    Args:
        query: A search query with specific terms from the user's question.
    """
    response = await ctx.deps.openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_vector = response.data[0].embedding
    chunks = retrieve(query_vector, query)
    ctx.deps.retrieved_chunks = chunks
    return [{"chunk_id": c.id, "source": c.source, "title": c.title, "section": c.section, "page_start": c.page_start, "text": c.text} for c in chunks]


async def ask(question: str, openai_client: AsyncOpenAI | None = None) -> RAGResponse:
    """Run the RAG agent and return a validated structured response."""
    if openai_client is None:
        openai_client = AsyncOpenAI()
    deps = RAGDeps(openai_client=openai_client)
    result = await rag_agent.run(
        question,
        deps=deps,
        model="openai:gpt-5-mini",
        usage_limits=UsageLimits(request_limit=10),
    )
    sources = [
        SourceReference(
            chunk_id=c.id,
            source=c.source,
            section=c.section,
            title=c.title,
            page_start=c.page_start,
            text=c.text,
        )
        for c in deps.retrieved_chunks
    ]
    return RAGResponse(
        answer=result.output.answer,
        sources=sources,
        confidence=result.output.confidence,
    )
