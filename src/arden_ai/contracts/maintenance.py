"""Versioned Flow 4 analysis contract; source text is request-only."""

import hashlib
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from arden_ai.contracts.base import ContractModel
from arden_ai.contracts.usage import ProviderUsageEvent

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class MaintenanceSource(ContractModel):
    version_id: UUID
    text: Annotated[str, Field(min_length=1, max_length=8_000)]
    text_hash: Sha256

    @model_validator(mode="after")
    def validate_hash(self) -> "MaintenanceSource":
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_hash:
            raise ValueError("maintenance source hash mismatch")
        return self


class MaintenanceAnalysisRequest(ContractModel):
    contract_version: Literal[2] = 2
    request_id: UUID
    case_id: UUID
    sources: Annotated[list[MaintenanceSource], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def validate_sources(self) -> "MaintenanceAnalysisRequest":
        if len({source.version_id for source in self.sources}) != len(self.sources):
            raise ValueError("maintenance sources must be unique")
        return self


class MaintenanceFinding(ContractModel):
    version_id: UUID
    summary: Annotated[str, Field(min_length=1, max_length=350)]
    text_hash: Sha256
    quote: Annotated[str, Field(min_length=1, max_length=240)]


class MaintenanceAnalysisResponse(ContractModel):
    contract_version: Literal[2] = 2
    state: Literal["succeeded"]
    findings: Annotated[list[MaintenanceFinding], Field(min_length=1, max_length=4)]
    usage_contract_version: Literal[1]
    usage_events: Annotated[list[ProviderUsageEvent], Field(max_length=1)]

    @model_validator(mode="after")
    def validate_usage(self) -> "MaintenanceAnalysisResponse":
        if any(event.operation != "MAINTENANCE_ANALYSIS" for event in self.usage_events):
            raise ValueError("maintenance response contains an unrelated usage event")
        return self
