import asyncio
import json
import time
from contextlib import suppress
from typing import Any, cast
from urllib.parse import urlsplit

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from arden_ai.application import PreparedGeneration
from arden_ai.contracts import IndexGenerationRequest
from arden_ai.contracts.usage import ProviderUsageEvent, make_usage_event, token_count
from arden_ai.domain.chunking import CHUNKING_VERSION, PreparedChunk
from arden_ai.infrastructure.settings import Settings


def _chroma_endpoint(url: str) -> tuple[str, int, bool]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ARDEN_AI_CHROMA_URL must be an http(s) server URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
        raise ValueError(
            "ARDEN_AI_CHROMA_URL must not include credentials, path, query, or fragment"
        )
    return (
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
        parsed.scheme == "https",
    )


def _chunk_metadata(
    request: IndexGenerationRequest, prepared: PreparedGeneration, chunk: PreparedChunk
) -> dict[str, str | int]:
    return {
        "organization_id": str(request.organization_id),
        "submission_id": str(request.submission_id),
        "generation_id": str(request.generation_id),
        "snapshot_version_id": str(request.snapshot_version_id),
        "snapshot_content_hash": request.snapshot_content_hash,
        "chunk_ordinal": chunk.ordinal,
        "chunk_content_hash": chunk.content_hash,
        "token_count": chunk.token_count,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "source_block_ids_json": json.dumps(chunk.source_block_ids, separators=(",", ":")),
        "parent_section_block_id": chunk.parent_section_block_id or "",
        "heading_path_json": json.dumps(
            chunk.heading_path, ensure_ascii=False, separators=(",", ":")
        ),
        "compiler_version": prepared.compiled.compiler_version,
        "chunking_version": CHUNKING_VERSION,
    }


class ChromaCandidateGenerationStore:
    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise ValueError("ARDEN_AI_OPENAI_API_KEY is required for indexing")
        host, port, ssl = _chroma_endpoint(settings.chroma_url)
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            api_key=settings.openai_api_key,
            max_retries=4,
            timeout=45,
            chunk_size=1_000,
        )
        self._embedding_model = settings.openai_embedding_model
        self._embedding_dimensions = settings.openai_embedding_dimensions
        self._store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self._embeddings,
            host=host,
            port=port,
            ssl=ssl,
        )

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def embedding_dimensions(self) -> int:
        return self._embedding_dimensions

    async def replace_candidate(
        self, request: IndexGenerationRequest, prepared: PreparedGeneration
    ) -> list[ProviderUsageEvent]:
        generation_filter = {"generation_id": str(request.generation_id)}
        await self._store.adelete(where=generation_filter)
        try:
            ids = [chunk.chunk_id for chunk in prepared.chunks]
            texts = [chunk.embedding_text for chunk in prepared.chunks]
            vectors: list[list[float]] = []
            usage_events: list[ProviderUsageEvent] = []
            for offset in range(0, len(texts), 1_000):
                batch = texts[offset : offset + 1_000]
                started = time.perf_counter()
                vectors.extend(await self._embeddings.aembed_documents(batch))
                usage_events.append(
                    make_usage_event(
                        "INDEX_EMBEDDING",
                        self._embedding_model,
                        round((time.perf_counter() - started) * 1_000),
                        {
                            "input_tokens": token_count(self._embedding_model, batch),
                            "output_tokens": 0,
                        },
                        event_key=f"{request.generation_id}:INDEX_EMBEDDING:batch:{offset}",
                    )
                )
            metadatas = [_chunk_metadata(request, prepared, chunk) for chunk in prepared.chunks]
            collection = cast(Any, self._store._collection)
            await asyncio.to_thread(
                collection.upsert,
                embeddings=cast(Any, vectors),
                documents=texts,
                metadatas=cast(Any, metadatas),
                ids=ids,
            )
            stored = await asyncio.to_thread(self._store.get, ids=ids, include=[])
            if stored["ids"] != ids:
                raise RuntimeError("Chroma did not acknowledge the complete candidate generation")
            return usage_events
        except Exception:
            # Arden core never activates a generation without a complete manifest.
            with suppress(Exception):
                await self._store.adelete(where=generation_filter)
            raise
