"""Application boundary for Flow 4 maintenance analysis."""

from typing import Protocol

from arden_ai.contracts.maintenance import MaintenanceAnalysisRequest, MaintenanceAnalysisResponse


class MaintenanceAnalyzer(Protocol):
    async def analyze(self, request: MaintenanceAnalysisRequest) -> MaintenanceAnalysisResponse: ...
