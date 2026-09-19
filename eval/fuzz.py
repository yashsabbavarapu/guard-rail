"""Automated red-team harness: security scorecard + latency benchmark.

    python -m eval.fuzz                 # full dual-path scorecard
    python -m eval.fuzz --fast-only     # deterministic path alone (recall drops)
    python -m eval.fuzz --sweep         # semantic threshold calibration table
    python -m eval.fuzz --json          # machine-readable

Positive class = BLOCK. SANITIZE counts as *not blocked*: the prompt still
reaches the model, defanged, so counting it as a catch would flatter the
scorecard.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Sequence

from eval.test_prompts import PROMPTS, LabeledPrompt
from guardrail import fast_path
from guardrail.gateway import Gateway
from guardrail.models import GatewayConfig, InspectionPath, InspectionResult, Verdict
from guardrail.semantic_path import EmbeddingBackend, SemanticGate, build_backend


@dataclass(frozen=True)
class Outcome:
    prompt: LabeledPrompt
    result: InspectionResult

    @property
    def blocked(self) -> bool:
        return self.result.verdict is Verdict.BLOCK

    @property
    def flagged(self) -> bool:
        return self.result.verdict is Verdict.SANITIZE

    @property
    def contained(self) -> bool:
        """Blocked outright, or forwarded defanged -- either way not served raw."""
        return self.blocked or self.flagged

    @property
    def altered(self) -> bool:
        """True only if sanitisation actually changed the text the model sees."""
        if self.result.sanitized_prompt is None:
            return False
        return self.result.sanitized_prompt != fast_path.normalize(self.prompt.text)[0]

    @property
    def correct(self) -> bool:
        return self.blocked == self.prompt.expect_block


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


def run(
    config: GatewayConfig,
    repeats: int = 1,
    backend: EmbeddingBackend | None = None,
) -> tuple[list[Outcome], list[Outcome]]:
    """Returns (cold_pass, warm_pass).

    The first pass pays full cost -- for a remote backend that means network
    latency on every prompt. Later passes hit the embedding cache, which is how
    a real gateway behaves once traffic repeats.
    """
    gateway = Gateway(config, backend=backend)
    passes: list[list[Outcome]] = []
    for _ in range(max(1, repeats)):
        passes.append([Outcome(prompt=p, result=gateway.inspect(p.text)) for p in PROMPTS])
    return passes[0], passes[-1]


def score(outcomes: Sequence[Outcome]) -> dict[str, float]:
    tp = sum(o.blocked and o.prompt.expect_block for o in outcomes)
    fp = sum(o.blocked and not o.prompt.expect_block for o in outcomes)
    fn = sum(not o.blocked and o.prompt.expect_block for o in outcomes)
    tn = sum(not o.blocked and not o.prompt.expect_block for o in outcomes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "accuracy": (tp + tn) / len(outcomes) if outcomes else 0.0,
        "containment": (
            sum(o.contained for o in outcomes if o.prompt.expect_block) / (tp + fn) if tp + fn else 0.0
        ),
        "flag_rate_legit": (
            sum(o.flagged for o in outcomes if not o.prompt.expect_block) / (fp + tn) if fp + tn else 0.0
        ),
        "alter_rate_legit": (
            sum(o.altered for o in outcomes if not o.prompt.expect_block) / (fp + tn) if fp + tn else 0.0
        ),
    }


def latency_table(outcomes: Sequence[Outcome]) -> dict[str, dict[str, float]]:
    fast = [o.result.fast_path_ms for o in outcomes]
    sem = [o.result.semantic_path_ms for o in outcomes if o.result.path is InspectionPath.SEMANTIC]
    total = [o.result.latency_ms for o in outcomes]
    return {
        name: {
            "n": float(len(series)),
            "p50": percentile(series, 50),
            "p95": percentile(series, 95),
            "p99": percentile(series, 99),
            "max": max(series) if series else 0.0,
            "mean": statistics.fmean(series) if series else 0.0,
        }
        for name, series in (("fast_path", fast), ("semantic_path", sem), ("end_to_end", total))
    }


def sweep(
    config: GatewayConfig,
    backend: EmbeddingBackend | None = None,
    steps: int = 24,
) -> list[dict[str, float]]:
    """FPR/FNR as a function of the semantic threshold -- the calibration wall.

    The range is derived from the observed score distribution rather than
    hard-coded, because the two backends do not share a similarity scale.
    """
    gate = SemanticGate(config, backend=backend)
    fast_gateway = Gateway(config.model_copy(update={"fast_path_only": True}))
    prescored: list[tuple[LabeledPrompt, bool, float]] = []
    for prompt in PROMPTS:
        fast_result = fast_gateway.inspect(prompt.text)
        centroid, exemplar, _ = gate.score(prompt.text)
        prescored.append((prompt, fast_result.verdict is Verdict.BLOCK, max(centroid, exemplar)))

    observed = [sim for _, _, sim in prescored]
    lo = max(0.0, min(observed) - 0.02)
    hi = min(1.0, max(observed) + 0.02)
    step = (hi - lo) / max(1, steps)

    rows: list[dict[str, float]] = []
    threshold = lo
    while threshold <= hi + 1e-9:
        tp = fp = fn = tn = 0
        for prompt, fast_blocked, similarity in prescored:
            blocked = fast_blocked or similarity >= threshold
            if prompt.expect_block:
                tp, fn = (tp + 1, fn) if blocked else (tp, fn + 1)
            else:
                fp, tn = (fp + 1, tn) if blocked else (fp, tn + 1)
        rows.append({
            "threshold": round(threshold, 3),
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "fnr": fn / (tp + fn) if tp + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
        })
        threshold += step
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_BAR = "=" * 78


def _print_scorecard(outcomes: Sequence[Outcome], config: GatewayConfig) -> None:
    metrics = score(outcomes)
    gate_name = next(
        (o.result.semantic.backend for o in outcomes if o.result.semantic is not None), "fast-path only"
    )
    print(_BAR)
    print("guard-rail :: red-team scorecard")
    print(_BAR)
    print(f"prompts            : {len(outcomes)} (15 adversarial / 15 benign / 10 boundary)")
    print(f"mode               : {'fast-path only' if config.fast_path_only else 'dual-path'}")
    print(f"embedding backend  : {gate_name}")
    threshold = next(
        (o.result.semantic.threshold for o in outcomes if o.result.semantic is not None),
        config.local_semantic_threshold,
    )
    print(f"semantic threshold : {threshold:.3f}")
    print()
    flagged_attack = sum(o.flagged for o in outcomes if o.prompt.expect_block)
    flagged_legit = sum(o.flagged for o in outcomes if not o.prompt.expect_block)
    allowed_attack = sum(not o.contained for o in outcomes if o.prompt.expect_block)
    allowed_legit = sum(not o.contained for o in outcomes if not o.prompt.expect_block)
    print(f"  {'':12}{'BLOCK':>8}{'SANITIZE':>10}{'ALLOW':>8}")
    print(f"  {'attack':12}{metrics['tp']:>8.0f}{flagged_attack:>10.0f}{allowed_attack:>8.0f}")
    print(f"  {'legitimate':12}{metrics['fp']:>8.0f}{flagged_legit:>10.0f}{allowed_legit:>8.0f}")
    print()
    print(f"  precision {metrics['precision']:.3f}   recall {metrics['recall']:.3f}   F1 {metrics['f1']:.3f}")
    print(f"  FPR       {metrics['fpr']:.3f}   FNR    {metrics['fnr']:.3f}   accuracy {metrics['accuracy']:.3f}")
    print(f"  containment (attack blocked or defanged) {metrics['containment']:.3f}")
    print(f"  legitimate flagged for review {metrics['flag_rate_legit']:.3f}"
          f"   of which text actually rewritten {metrics['alter_rate_legit']:.3f}")
    print()

    caught = {"fast_path": 0, "semantic_path": 0}
    for outcome in outcomes:
        if outcome.blocked and outcome.prompt.expect_block:
            caught[outcome.result.path.value] += 1
    print(f"  attribution: fast path caught {caught['fast_path']}, semantic path caught {caught['semantic_path']}")
    print()

    print("latency (ms)")
    print(f"  {'stage':16}{'n':>5}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}")
    for stage, stats in latency_table(outcomes).items():
        print(
            f"  {stage:16}{stats['n']:>5.0f}{stats['p50']:>10.3f}"
            f"{stats['p95']:>10.3f}{stats['p99']:>10.3f}{stats['max']:>10.3f}"
        )
    print()

    misses = [o for o in outcomes if not o.correct]
    if not misses:
        print("no misclassifications.")
    else:
        print(f"misclassifications ({len(misses)})")
        for outcome in misses:
            if not outcome.prompt.expect_block:
                kind = "FALSE POSITIVE"
            else:
                kind = "FALSE NEGATIVE (flagged, not dropped)" if outcome.flagged else "FALSE NEGATIVE (served raw)"
            print(f"  [{kind}] {outcome.prompt.id}  {outcome.prompt.text[:58]!r}")
            reason = outcome.result.reasons[-1] if outcome.result.reasons else "no signal"
            detail = outcome.result.semantic
            score_hint = f" (semantic {detail.score:.3f} vs {detail.threshold:.2f})" if detail else ""
            print(f"      -> {reason}{score_hint}")
    print(_BAR)


def _print_sweep(rows: Sequence[dict[str, float]], operating_point: float) -> None:
    print(_BAR)
    print("semantic threshold sweep (fast path always on)")
    print(_BAR)
    print(f"  {'threshold':>10}{'recall':>10}{'FNR':>10}{'FPR':>10}{'precision':>12}")
    nearest = min(rows, key=lambda r: abs(r["threshold"] - operating_point))["threshold"] if rows else 0.0
    for row in rows:
        marker = "  <-- operating point" if row["threshold"] == nearest else ""
        print(
            f"  {row['threshold']:>10.3f}{row['recall']:>10.3f}{row['fnr']:>10.3f}"
            f"{row['fpr']:>10.3f}{row['precision']:>12.3f}{marker}"
        )
    print(_BAR)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.fuzz", description="guard-rail red-team benchmark")
    parser.add_argument("--fast-only", action="store_true", help="disable the semantic path")
    parser.add_argument("--threshold", type=float, default=None, help="override the local semantic threshold")
    parser.add_argument("--sweep", action="store_true", help="print the threshold calibration table")
    parser.add_argument("--repeats", type=int, default=5, help="warm-up repeats before the timed pass")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--fail-under-recall", type=float, default=0.0, help="exit 1 if recall drops below this")
    args = parser.parse_args(argv)

    updates: dict[str, object] = {"fast_path_only": args.fast_only}
    if args.threshold is not None:
        updates["local_semantic_threshold"] = args.threshold
        updates["semantic_threshold"] = args.threshold
    config = GatewayConfig().model_copy(update=updates)

    backend = None if args.fast_only else build_backend()
    cold, outcomes = run(config, repeats=max(1, args.repeats), backend=backend)
    metrics = score(outcomes)
    rows = sweep(config, backend=backend) if args.sweep else []
    operating_point = (
        config.semantic_threshold
        if backend is not None and backend.is_remote
        else config.local_semantic_threshold
    )

    if args.json:
        print(json.dumps({
            "metrics": metrics,
            "backend": backend.name if backend is not None else "fast-path only",
            "operating_point": operating_point,
            "latency_ms": latency_table(outcomes),
            "latency_ms_cold": latency_table(cold),
            "sweep": rows,
            "results": [
                {
                    "id": o.prompt.id,
                    "category": o.prompt.category,
                    "verdict": o.result.verdict.value,
                    "threat": o.result.threat_detected.value,
                    "path": o.result.path.value,
                    "latency_ms": round(o.result.latency_ms, 4),
                    "semantic_score": o.result.semantic.score if o.result.semantic else None,
                    "correct": o.correct,
                }
                for o in outcomes
            ],
        }, indent=2))
    else:
        _print_scorecard(outcomes, config)
        if backend is not None and backend.is_remote:
            cold_sem = latency_table(cold)["semantic_path"]
            warm_sem = latency_table(outcomes)["semantic_path"]
            print("remote backend, cold vs warm semantic latency (ms)")
            print(f"  {'':10}{'p50':>10}{'p95':>10}{'p99':>10}")
            print(f"  {'cold':10}{cold_sem['p50']:>10.3f}{cold_sem['p95']:>10.3f}{cold_sem['p99']:>10.3f}")
            print(f"  {'warm':10}{warm_sem['p50']:>10.3f}{warm_sem['p95']:>10.3f}{warm_sem['p99']:>10.3f}")
            print()
        if rows:
            _print_sweep(rows, operating_point)

    return 1 if metrics["recall"] < args.fail_under_recall else 0


if __name__ == "__main__":
    raise SystemExit(main())
