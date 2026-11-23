from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from arden_ai.contracts import AnswerRequest, AnswerResponse
from arden_ai.contracts.usage import ProviderUsageEvent


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    generation_id: UUID
    snapshot_version_id: UUID
    snapshot_content_hash: str
    content: str
    source_block_ids: tuple[str, ...]
    parent_section_block_id: str | None


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    insufficient_evidence: bool = False
    usage_events: tuple[ProviderUsageEvent, ...] = ()


class PermissionAwareRetriever(Protocol):
    async def retrieve(
        self, request: AnswerRequest, limit: int
    ) -> tuple[tuple[RetrievedChunk, ...], ProviderUsageEvent]: ...


class GroundedAnswerGenerator(Protocol):
    async def generate(
        self,
        query: str,
        evidence: tuple[RetrievedChunk, ...],
        request_id: UUID,
    ) -> GeneratedAnswer: ...


class PermissionAwareAnswerer:
    def __init__(
        self,
        retriever: PermissionAwareRetriever,
        generator: GroundedAnswerGenerator,
        *,
        retrieval_limit: int = 8,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._retrieval_limit = retrieval_limit

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        evidence, retrieval_event = await self._retriever.retrieve(request, self._retrieval_limit)
        allowed = set(request.allowed_generation_ids)
        if any(chunk.generation_id not in allowed for chunk in evidence):
            raise RuntimeError("retriever returned evidence outside the Core allow-list")
        if not evidence:
            return self._insufficient(request, usage_events=[retrieval_event])

        generated = await self._generator.generate(request.query, evidence, request.request_id)
        usage_events = [retrieval_event, *generated.usage_events]
        if generated.insufficient_evidence:
            return self._insufficient(request, generated.answer, usage_events)

        by_id = {chunk.chunk_id: chunk for chunk in evidence}
        if len(set(generated.cited_chunk_ids)) != len(generated.cited_chunk_ids):
            raise RuntimeError("answer generator returned duplicate citations")
        try:
            cited = [by_id[chunk_id] for chunk_id in generated.cited_chunk_ids]
        except KeyError as error:
            raise RuntimeError("answer generator cited evidence that was not retrieved") from error
        if not cited:
            return self._insufficient(request, usage_events=usage_events)

        return AnswerResponse.model_validate(
            {
                "requestId": str(request.request_id),
                "status": "ANSWERED",
                "answer": generated.answer,
                "usageContractVersion": 1,
                "usageEvents": [event.model_dump(by_alias=True) for event in usage_events],
                "citations": [
                    {
                        "generationId": str(chunk.generation_id),
                        "snapshotVersionId": str(chunk.snapshot_version_id),
                        "snapshotContentHash": chunk.snapshot_content_hash,
                        "chunkId": chunk.chunk_id,
                        "sourceBlockIds": list(chunk.source_block_ids),
                        "parentSectionBlockId": chunk.parent_section_block_id,
                        "excerpt": chunk.content[:1_200],
                    }
                    for chunk in cited[:12]
                ],
            }
        )

    @staticmethod
    def _insufficient(
        request: AnswerRequest,
        answer: str = (
            "I could not find enough authorized published knowledge to answer that question."
        ),
        usage_events: list[ProviderUsageEvent] | None = None,
    ) -> AnswerResponse:
        return AnswerResponse.model_validate(
            {
                "requestId": str(request.request_id),
                "status": "INSUFFICIENT_EVIDENCE",
                "answer": answer,
                "citations": [],
                "usageContractVersion": 1,
                "usageEvents": [event.model_dump(by_alias=True) for event in usage_events or []],
            }
        )
