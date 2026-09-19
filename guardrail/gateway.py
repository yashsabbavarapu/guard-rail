"""Dual-path coordinator.

    prompt -> fast path -> BLOCK?            -> drop (no vectors, no network)
                        -> borderline/clean  -> semantic path -> final verdict

The ordering is the whole point: the deterministic path costs microseconds and
absorbs the majority of real traffic, so the vector gate only ever sees prompts
that survived it.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from guardrail import fast_path
from guardrail.models import (
    GatewayConfig,
    InspectionPath,
    InspectionResult,
    SemanticAssessment,
    Signal,
    ThreatType,
    Verdict,
)
from guardrail.semantic_path import EmbeddingBackend, SemanticGate


class Gateway:
    """Stateless per-request; the embedded attack bank is built once at init."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        gate: SemanticGate | None = None,
        backend: EmbeddingBackend | None = None,
        fixtures: Path | None = None,
    ) -> None:
        self.config = config or GatewayConfig()
        if self.config.fast_path_only and gate is None and backend is None:
            self.gate: SemanticGate | None = None
        else:
            self.gate = gate or SemanticGate(self.config, backend=backend, fixtures=fixtures)

    def inspect(self, prompt: str) -> InspectionResult:
        started = perf_counter()
        report = fast_path.inspect(prompt, self.config)

        # Step 1 -- deterministic drop. Nothing downstream is touched.
        if report.verdict is Verdict.BLOCK:
            return self._result(
                verdict=Verdict.BLOCK,
                signal=report.top,
                reasons=[s.reason for s in report.signals],
                path=InspectionPath.FAST,
                fast_ms=report.latency_ms,
                started=started,
            )

        # Step 2 -- semantic verification of borderline and clean traffic.
        if self.gate is None:
            verdict = Verdict.SANITIZE if report.borderline else Verdict.ALLOW
            return self._result(
                verdict=verdict,
                signal=report.top,
                reasons=[s.reason for s in report.signals],
                path=InspectionPath.FAST,
                fast_ms=report.latency_ms,
                started=started,
                sanitized=report.sanitized if verdict is Verdict.SANITIZE else None,
            )

        signal, assessment = self.gate.inspect(report.normalized)
        reasons = [s.reason for s in report.signals]

        if signal is not None:
            semantic_verdict = (
                Verdict.BLOCK if signal.confidence >= self.config.block_confidence else Verdict.SANITIZE
            )
            return self._result(
                verdict=semantic_verdict,
                signal=signal,
                reasons=reasons + [signal.reason],
                path=InspectionPath.SEMANTIC,
                fast_ms=report.latency_ms,
                semantic_ms=assessment.latency_ms,
                started=started,
                sanitized=report.sanitized if semantic_verdict is Verdict.SANITIZE else None,
                assessment=assessment,
            )

        # Step 3 -- survived both gates. A borderline fast-path hit that the
        # semantic gate cleared is forwarded defanged rather than dropped.
        if report.borderline:
            return self._result(
                verdict=Verdict.SANITIZE,
                signal=report.top,
                reasons=reasons + [f"semantic gate cleared it ({assessment.score:.3f} < {assessment.threshold:.2f})"],
                path=InspectionPath.SEMANTIC,
                fast_ms=report.latency_ms,
                semantic_ms=assessment.latency_ms,
                started=started,
                sanitized=report.sanitized,
                assessment=assessment,
            )

        return self._result(
            verdict=Verdict.ALLOW,
            signal=fast_path.BENIGN_SIGNAL,
            reasons=[],
            path=InspectionPath.SEMANTIC,
            fast_ms=report.latency_ms,
            semantic_ms=assessment.latency_ms,
            started=started,
            assessment=assessment,
        )

    def _result(
        self,
        *,
        verdict: Verdict,
        signal: Signal,
        reasons: list[str],
        path: InspectionPath,
        started: float,
        fast_ms: float = 0.0,
        semantic_ms: float = 0.0,
        sanitized: str | None = None,
        assessment: SemanticAssessment | None = None,
    ) -> InspectionResult:
        threat = signal.threat if verdict is not Verdict.ALLOW else ThreatType.BENIGN
        return InspectionResult(
            verdict=verdict,
            threat_detected=threat,
            confidence=signal.confidence,
            latency_ms=(perf_counter() - started) * 1000.0,
            reasons=reasons,
            sanitized_prompt=sanitized,
            path=path,
            fast_path_ms=fast_ms,
            semantic_path_ms=semantic_ms,
            semantic=assessment,
        )


def inspect(prompt: str, config: GatewayConfig | None = None) -> InspectionResult:
    """One-shot helper. Prefer reusing a `Gateway` -- init embeds the bank."""
    return Gateway(config).inspect(prompt)
