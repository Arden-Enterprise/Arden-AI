import asyncio
import hashlib
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from arden_ai.contracts.maintenance import (
    MaintenanceAnalysisRequest,
    MaintenanceAnalysisResponse,
    MaintenanceFinding,
)
from arden_ai.contracts.usage import make_usage_event
from arden_ai.infrastructure.maintenance import (
    OpenAIMaintenanceAnalyzer,
    _AnalysisOutput,
    _Finding,
)
from arden_ai.infrastructure.settings import Settings
from arden_ai.presentation import create_api


def request() -> MaintenanceAnalysisRequest:
    text = "The policy applies to all active accounts."
    return MaintenanceAnalysisRequest.model_validate(
        {
            "contractVersion": 2,
            "requestId": str(uuid4()),
            "caseId": str(uuid4()),
            "sources": [
                {
                    "versionId": str(uuid4()),
                    "text": text,
                    "textHash": hashlib.sha256(text.encode()).hexdigest(),
                }
            ],
        }
    )


def test_maintenance_provider_returns_only_bounded_quote_and_usage(monkeypatch) -> None:
    current = request()
    source = current.sources[0]
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_structured_output(self, schema, **kwargs):
            assert schema is _AnalysisOutput
            assert kwargs == {"method": "json_schema", "strict": True, "include_raw": True}
            return self

        async def ainvoke(self, messages):
            captured["messages"] = messages
            return {
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 21, "output_tokens": 8}),
                "parsed": _AnalysisOutput(
                    findings=[
                        _Finding(
                            version_id=source.version_id,
                            summary="Clarify the policy's account scope.",
                            text_hash=source.text_hash,
                            quote="all active accounts",
                        )
                    ]
                ),
            }

    monkeypatch.setattr("arden_ai.infrastructure.maintenance.ChatOpenAI", FakeModel)
    analyzer = OpenAIMaintenanceAnalyzer(
        Settings(openai_api_key=SecretStr("test-only"), _env_file=None)
    )
    result = asyncio.run(analyzer.analyze(current))

    assert result.state == "succeeded"
    assert result.findings[0].quote == "all active accounts"
    assert result.usage_contract_version == 1
    assert len(result.usage_events) == 1
    assert result.usage_events[0].operation == "MAINTENANCE_ANALYSIS"
    assert result.usage_events[0].input_tokens == 21
    assert captured["max_retries"] == 0 and captured["store"] is False
    assert source.text in captured["messages"][1].content
    assert source.text not in result.model_dump_json(by_alias=True)


def test_maintenance_provider_rejects_quote_not_in_exact_source(monkeypatch) -> None:
    current = request()
    source = current.sources[0]

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def with_structured_output(self, _schema, **_kwargs):
            return self

        async def ainvoke(self, _messages):
            return {
                "raw": SimpleNamespace(usage_metadata={}),
                "parsed": _AnalysisOutput(
                    findings=[
                        _Finding(
                            version_id=source.version_id,
                            summary="An unsupported issue.",
                            text_hash=source.text_hash,
                            quote="not source text",
                        )
                    ]
                ),
            }

    monkeypatch.setattr("arden_ai.infrastructure.maintenance.ChatOpenAI", FakeModel)
    analyzer = OpenAIMaintenanceAnalyzer(
        Settings(openai_api_key=SecretStr("test-only"), _env_file=None)
    )
    try:
        asyncio.run(analyzer.analyze(current))
    except ValueError as error:
        assert str(error) == "Maintenance analysis citation was invalid"
    else:
        raise AssertionError("invalid maintenance quote was accepted")


def test_maintenance_analysis_route_requires_service_auth_and_validates_contract() -> None:
    current = request()

    class FakeAnalyzer:
        async def analyze(self, value: MaintenanceAnalysisRequest) -> MaintenanceAnalysisResponse:
            source = value.sources[0]
            return MaintenanceAnalysisResponse.model_validate(
                {
                    "contractVersion": 2,
                    "state": "succeeded",
                    "findings": [
                        MaintenanceFinding.model_validate(
                            {
                                "versionId": source.version_id,
                                "summary": "The version needs review.",
                                "textHash": source.text_hash,
                                "quote": source.text[:20],
                            }
                        )
                    ],
                    "usageContractVersion": 1,
                    "usageEvents": [
                        make_usage_event(
                            "MAINTENANCE_ANALYSIS",
                            "gpt-6-luna",
                            10,
                            event_key=f"{value.request_id}:MAINTENANCE_ANALYSIS:analysis",
                        )
                    ],
                }
            )

    client = TestClient(
        create_api(
            service_token=SecretStr("test-only"),
            maintenance_analyzer=FakeAnalyzer(),
        )
    )
    payload = current.model_dump(mode="json", by_alias=True)

    assert client.post("/v2/maintenance-analyses", json=payload).status_code == 401
    response = client.post(
        "/v2/maintenance-analyses",
        json=payload,
        headers={"Authorization": "Bearer test-only"},
    )
    assert response.status_code == 200
    assert response.json()["usageContractVersion"] == 1
    assert response.json()["findings"][0]["quote"] == current.sources[0].text[:20]
    assert "text" not in response.json()["findings"][0]

    payload["sources"][0]["textHash"] = "0" * 64
    invalid = client.post(
        "/v2/maintenance-analyses",
        json=payload,
        headers={"Authorization": "Bearer test-only"},
    )
    assert invalid.status_code == 422
