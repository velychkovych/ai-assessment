from enum import Enum

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    section: str = ""
    title: str = ""
    page_start: int = 0
    chunk_index: int


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceReference(BaseModel):
    chunk_id: str = Field(description="Unique chunk identifier, e.g. 'Attention.pdf:5'")
    source: str = Field(description="Filename the chunk came from")
    section: str = Field(description="Section heading the chunk belongs to")
    title: str = Field(description="Paper title extracted from the PDF")
    page_start: int = Field(description="Page number where the section starts")
    text: str = Field(description="The chunk text that was used")


class RAGQuery(BaseModel):
    question: str = Field(min_length=1, description="The user question")


class AgentOutput(BaseModel):
    answer: str = Field(description="A concise answer (2-4 sentences) based only on the retrieved context")
    confidence: Confidence = Field(description="high = context directly answers; medium = requires inference; low = context insufficient")


class RAGResponse(BaseModel):
    answer: str = Field(description="The generated answer")
    sources: list[SourceReference] = Field(description="Chunks used to generate the answer")
    confidence: Confidence = Field(description="Confidence in the answer: high, medium, or low")


class FaithfulnessVerdict(BaseModel):
    reason: str = Field(description="Brief explanation of the verdict")
    faithful: bool = Field(description="True if the answer is fully supported by the provided context")
