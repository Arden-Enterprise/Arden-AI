"""Summary-only provider budget; existing chat configuration is unchanged."""

import time

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from arden_ai.application.reader_summary import (
    SYSTEM_PROMPT,
    InvalidSummary,
    SummaryWindow,
    WindowOutput,
)
from arden_ai.contracts.usage import ProviderUsageEvent, make_usage_event, usage_metadata
from arden_ai.infrastructure.settings import Settings


class OpenAISummaryGenerator:
    def __init__(self, settings: Settings) -> None:
        if settings.openai_api_key is None:
            raise ValueError("Summary provider is not configured")
        model = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            reasoning={"effort": settings.openai_reasoning_effort},
            use_responses_api=True,
            max_completion_tokens=4_096,
            max_retries=0,
            timeout=45,
            output_version="responses/v1",
            store=False,
        )
        self._structured = model.with_structured_output(
            WindowOutput,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        self._model_name = settings.openai_chat_model

    async def generate(self, window: SummaryWindow) -> tuple[object, ProviderUsageEvent]:
        started = time.perf_counter()
        try:
            result = await self._structured.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=window.payload),
                ]
            )
        except OutputParserException as error:
            raise InvalidSummary("provider structured output was invalid") from error
        raw = result.get("raw") if isinstance(result, dict) and "raw" in result else None
        parsed = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
        event = make_usage_event(
            "READER_SUMMARY",
            self._model_name,
            round((time.perf_counter() - started) * 1_000),
            usage_metadata(raw),
            event_key=(
                f"{window.request_id}:READER_SUMMARY:window:{window.index}:attempt:{window.attempt}"
            ),
        )
        if parsed is None:
            raise InvalidSummary("provider structured output was invalid")
        return parsed, event
