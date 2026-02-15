import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

from src.agent import ask, EMBEDDING_MODEL
from src.ingest import ingest

DATA_DIR = Path("data")

SAMPLE_QUESTIONS = [
    "What is the Transformer architecture and how does it differ from recurrent models?"
]


async def main():
    openai_client = AsyncOpenAI()

    # 1. Ingest documents
    print("Ingesting documents...")

    async def embed(texts: list[str]) -> list[list[float]]:
        resp = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    chunks = await ingest(DATA_DIR, embed)
    print(f"Indexed {len(chunks)} chunks from {DATA_DIR}\n")

    # 2. Ask sample questions
    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        response = await ask(question, openai_client)

        print(f"A: {response.answer}")
        print(f"Confidence: {response.confidence.value}")
        print("Sources:")
        for s in response.sources:
            print(f"  - [{s.chunk_id}] {s.source} | {s.section}")
            print(f"    {s.text[:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
