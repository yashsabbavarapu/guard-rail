"""guard-rail: a dual-path prompt-injection and jailbreak firewall."""

from guardrail.gateway import Gateway, inspect
from guardrail.models import (
    GatewayConfig,
    InspectionPath,
    InspectionResult,
    SemanticAssessment,
    Signal,
    ThreatType,
    Verdict,
)

__all__ = [
    "Gateway",
    "GatewayConfig",
    "InspectionPath",
    "InspectionResult",
    "SemanticAssessment",
    "Signal",
    "ThreatType",
    "Verdict",
    "inspect",
]
__version__ = "0.1.0"
