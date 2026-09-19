"""Typed contracts shared by both inspection paths.

Everything crossing a module boundary in guard-rail is one of these models, so
the fast path and the semantic path stay swappable behind a single result type.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ThreatType(str, Enum):
    """Taxonomy of what the gateway believes it is looking at."""

    INJECTION_DELIMITER = "INJECTION_DELIMITER"
    CANARY_LEAK = "CANARY_LEAK"
    HIGH_ENTROPY_OBFUSCATION = "HIGH_ENTROPY_OBFUSCATION"
    ADVERSARIAL_ROLEPLAY = "ADVERSARIAL_ROLEPLAY"
    SYSTEM_OVERRIDE = "SYSTEM_OVERRIDE"
    BENIGN = "BENIGN"


class Verdict(str, Enum):
    """ALLOW forwards verbatim, SANITIZE forwards a defanged copy, BLOCK drops."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    SANITIZE = "SANITIZE"


class InspectionPath(str, Enum):
    """Which stage produced the terminal verdict (for latency attribution)."""

    FAST = "fast_path"
    SEMANTIC = "semantic_path"


class Signal(BaseModel):
    """One detector firing. Confidence is calibrated against `GatewayConfig`."""

    model_config = ConfigDict(frozen=True)

    threat: ThreatType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SemanticAssessment(BaseModel):
    """Diagnostics from the adversarial vector gate, kept for the eval harness."""

    backend: str
    family: str | None = None
    centroid_similarity: float = 0.0
    exemplar_similarity: float = 0.0
    score: float = 0.0
    threshold: float = 1.0
    latency_ms: float = 0.0


class InspectionResult(BaseModel):
    """The single object the gateway hands back to a caller."""

    verdict: Verdict
    threat_detected: ThreatType
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    reasons: list[str] = Field(default_factory=list)
    sanitized_prompt: str | None = None

    # Diagnostics (not part of the block/allow decision).
    path: InspectionPath = InspectionPath.FAST
    fast_path_ms: float = 0.0
    semantic_path_ms: float = 0.0
    semantic: SemanticAssessment | None = None

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK


class GatewayConfig(BaseModel):
    """Tunables. Defaults are the calibration documented in README.md."""

    model_config = ConfigDict(frozen=True)

    fast_path_only: bool = False
    # Threshold for a true embedding backend (Gemini, taskType=SEMANTIC_SIMILARITY).
    semantic_threshold: float = 0.88
    # Hashed-ngram fallback lives on a different similarity scale; see README.
    local_semantic_threshold: float = 0.25
    # Scores within this margin below the threshold are flagged (SANITIZE), not
    # dropped -- the review band that keeps FPR at zero without going blind.
    # Backend-relative for the same reason the thresholds are: Gemini packs all
    # English text into a narrow high band, so a margin sized for the local
    # backend's wide scale would flag nearly all legitimate traffic.
    semantic_review_margin: float = 0.02
    local_semantic_review_margin: float = 0.10
    canary_token: str = "SECRET_CANARY_REV_77"
    # Fast-path confidence >= block_confidence drops without touching vectors.
    block_confidence: float = 0.90
    # Fast-path confidence >= escalate_confidence is "borderline": verify semantically.
    escalate_confidence: float = 0.45
