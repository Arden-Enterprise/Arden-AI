"""Versioned, content-free provider usage metadata shared with Arden Core."""

from typing import Annotated, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from arden_ai.contracts.base import ContractModel

UsageOperation = Literal[
    "CHAT_RETRIEVAL",
    "CHAT_GENERATION",
    "READER_QUESTION_RETRIEVAL",
    "READER_QUESTION_GENERATION",
    "READER_SUMMARY",
    "INDEX_EMBEDDING",
    "MAINTENANCE_ANALYSIS",
]


class ProviderUsageEvent(ContractModel):
    event_id: UUID
    operation: UsageOperation
    provider: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Z0-9_-]+$")]
    model: Annotated[str, Field(min_length=1, max_length=120)]
    input_tokens: int | None = Field(ge=0, strict=True)
    output_tokens: int | None = Field(ge=0, strict=True)
    latency_ms: int | None = Field(ge=0, strict=True)
    usage_complete: bool

    @model_validator(mode="after")
    def validate_completeness(self) -> "ProviderUsageEvent":
        if self.usage_complete and (self.input_tokens is None or self.output_tokens is None):
            raise ValueError("complete usage requires both token counts")
        return self


def make_usage_event(
    operation: UsageOperation,
    model: str,
    latency_ms: int,
    usage: object = None,
    *,
    event_key: str,
) -> ProviderUsageEvent:
    if not event_key.strip():
        raise ValueError("usage event key is required")
    metadata = usage if isinstance(usage, dict) else {}
    input_tokens = _token_count(metadata.get("input_tokens"))
    output_tokens = _token_count(metadata.get("output_tokens"))
    complete = input_tokens is not None and output_tokens is not None
    return ProviderUsageEvent.model_validate(
        {
            "eventId": str(uuid5(NAMESPACE_URL, f"arden-ai-usage-contract-v1:{event_key}")),
            "operation": operation,
            "provider": "OPENAI",
            "model": model,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latencyMs": max(0, latency_ms),
            "usageComplete": complete,
        }
    )


def usage_metadata(message: object) -> object:
    metadata = getattr(message, "usage_metadata", None)
    return metadata if isinstance(metadata, dict) else None


def token_count(model: str, texts: list[str]) -> int | None:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return sum(len(encoding.encode(text, disallowed_special=())) for text in texts)
    except (KeyError, ValueError):
        return None


def _token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
