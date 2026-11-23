from typing import Protocol

from arden_ai.application.reader_question import reader_question_payload
from arden_ai.contracts.governed import (
    DialogueTurn,
    Evidence,
    GenerationRequest,
    GenerationResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from arden_ai.contracts.usage import ProviderUsageEvent, UsageOperation
from arden_ai.domain.quotations import recover_source_quote


class GovernedRetriever(Protocol):
    async def retrieve(
        self, request: RetrievalRequest, query: str
    ) -> tuple[list[Evidence], str, list[ProviderUsageEvent]]: ...


class GovernedGenerator(Protocol):
    async def contextualize(
        self,
        query: str,
        history: list[DialogueTurn],
        request_id: str,
        operation: UsageOperation,
    ) -> tuple[str, ProviderUsageEvent]: ...
    async def generate_grounded(
        self, request: GenerationRequest, attempt: int
    ) -> GenerationResponse: ...


class GovernedAnswerer:
    """Two independent operations. Core alone authorizes the handoff between them."""

    def __init__(self, retriever: GovernedRetriever, generator: GovernedGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        operation: UsageOperation = (
            "READER_QUESTION_RETRIEVAL"
            if request.purpose == "READER_QUESTION"
            else "CHAT_RETRIEVAL"
        )
        usage_events: list[ProviderUsageEvent] = []
        if request.history:
            query, context_event = await self._generator.contextualize(
                request.query,
                request.history,
                str(request.request_id),
                operation,
            )
            usage_events.append(context_event)
        else:
            query = request.query
        if not query.strip() or len(query) > 4_000:
            raise ValueError("invalid standalone question")
        evidence, mode, retrieval_events = await self._retriever.retrieve(request, query)
        usage_events.extend(retrieval_events)
        if len({chunk.chunk_id for chunk in evidence}) != len(evidence):
            raise ValueError("duplicate retrieved identities")
        if any(chunk.generation_id not in request.allowed_generation_ids for chunk in evidence):
            raise ValueError("retrieval exceeded the Core allow-list")
        return RetrievalResponse.model_validate(
            {
                "requestId": str(request.request_id),
                "standaloneQuery": query,
                "retrievalMode": mode,
                "evidence": [chunk.model_dump(by_alias=True) for chunk in evidence],
                "usageContractVersion": 1,
                "usageEvents": [event.model_dump(by_alias=True) for event in usage_events],
            }
        )

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.purpose == "READER_QUESTION":
            reader_question_payload(request)  # Before even an injected generator can run.
        expected = {chunk.chunk_id for chunk in request.evidence}
        by_id = {chunk.chunk_id: chunk for chunk in request.evidence}
        # One bounded repair attempt, always the identical Core-authorized input set.
        usage_events: list[ProviderUsageEvent] = []
        for attempt in range(2):
            try:
                response = await self._generator.generate_grounded(request, attempt)
                usage_events.extend(response.usage_events)
                if (
                    response.request_id != request.request_id
                    or set(response.used_evidence) != expected
                ):
                    raise ValueError("generation input identity mismatch")
                citations = [
                    {
                        "chunkId": citation.chunk_id,
                        "quote": recover_source_quote(
                            citation.quote, by_id[citation.chunk_id].content
                        ),
                    }
                    for citation in response.citations
                ]
                # Revalidate the original schema/bounds after raw-span recovery.
                # Core still receives a literal source quote, not normalized text.
                return GenerationResponse.model_validate(
                    {
                        **response.model_dump(by_alias=True),
                        "citations": citations,
                        "usageContractVersion": 1,
                        "usageEvents": [event.model_dump(by_alias=True) for event in usage_events],
                    }
                )
            except (ValueError, KeyError):
                if attempt:
                    raise
        raise RuntimeError("unreachable generation retry")
