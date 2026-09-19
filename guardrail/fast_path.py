"""Deterministic heuristic gate. No vectors, no network, sub-millisecond.

Design note: every lexical detector returns a *calibrated* confidence rather
than a boolean. A raw pattern hit is worthless on its own -- `<|im_start|>`
inside a question about tokenizers is not an attack. Confidence is therefore
discounted by the surrounding context (quoted/code span, educational or
reporting framing), and only scores at or above `block_confidence` short
circuit the pipeline. Everything in between is handed to the semantic path.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from guardrail.models import GatewayConfig, Signal, ThreatType, Verdict

# --------------------------------------------------------------------------- #
# Pattern banks
# --------------------------------------------------------------------------- #

Rule = tuple[re.Pattern[str], str, float]

_DELIMITER_RULES: tuple[Rule, ...] = (
    (re.compile(r"<\|[a-z_]{2,24}\|>", re.I), "chatml control token", 0.93),
    (re.compile(r"\[/?INST\]"), "llama instruction delimiter", 0.93),
    (re.compile(r"<</?SYS>>"), "llama system delimiter", 0.93),
    (re.compile(r"-{2,}\s*(?:begin|end)\s+system\s+prompt\s*-{2,}", re.I), "fake system prompt banner", 0.95),
    (re.compile(r"^[ \t]*#{2,}[ \t]*(?:instruction|system|context|new\s+prompt)\b", re.I | re.M), "markdown instruction header", 0.88),
    (re.compile(r"^[ \t]*(?:system|assistant|developer)[ \t]*:", re.I | re.M), "role prefix line", 0.85),
    (re.compile(r"</?(?:system|instructions?)>", re.I), "pseudo-xml system tag", 0.90),
)

_OVERRIDE_RULES: tuple[Rule, ...] = (
    (re.compile(r"ignore\s+(?:all\s+)?(?:of\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|preceding|earlier|foregoing)\s+(?:\w+\s+){0,2}(?:instructions?|prompts?|rules?|directions?|messages?|context)", re.I), "instruction negation", 0.96),
    (re.compile(r"disregard\s+(?:all\s+)?(?:of\s+)?(?:the\s+|your\s+)?(?:previous|prior|earlier|above|safety|content|system|ethical)\b", re.I), "directive negation", 0.95),
    (re.compile(r"forget\s+(?:everything|all)\s+(?:you\b|above|before|prior|that)", re.I), "context wipe", 0.94),
    (re.compile(r"(?:reveal|repeat|print|output|show|display|echo|recite)\s+(?:me\s+)?(?:your|the)\s+(?:\w+\s+){0,2}(?:prompt|instructions?|directive|rules)\b", re.I), "prompt exfiltration", 0.95),
    (re.compile(r"repeat\s+(?:the\s+)?(?:words?|text|everything|content)\s+(?:above|before)", re.I), "context echo attack", 0.95),
    (re.compile(r"what\s+(?:were|are)\s+your\s+(?:original|initial|system|exact)\s+instructions", re.I), "instruction interrogation", 0.94),
    (re.compile(r"(?:without|bypass|ignore|drop|remove)\s+(?:any\s+|all\s+)?(?:your\s+)?(?:safety|ethical|content|moral|usage)\s+(?:guidelines?|filters?|restrictions?|policies|constraints?|rules)", re.I), "safety bypass", 0.95),
    (re.compile(r"you\s+are\s+(?:no\s+longer|not)\s+(?:bound|restricted|limited|constrained)\s+by", re.I), "constraint removal", 0.94),
    (re.compile(r"(?:enable|activate|enter|switch\s+to)\s+(?:\w+\s+){0,2}(?:developer|god|debug|unrestricted|admin|sudo)\s+mode", re.I), "privileged mode request", 0.93),
    (re.compile(r"\byour\s+system\s+prompt\b", re.I), "references the system prompt", 0.75),
)

_ROLEPLAY_RULES: tuple[Rule, ...] = (
    (re.compile(r"\bD\.?A\.?N\.?\b[^.\n]{0,24}\bmode\b", re.I), "named DAN persona", 0.93),
    (re.compile(r"\bdo\s+anything\s+now\b", re.I), "DAN expansion", 0.93),
    (re.compile(r"\bjail\s?break(?:ing|s)?\b(?!\s+(?:detection|defen|prevent|mitigat))", re.I), "explicit jailbreak reference", 0.88),
    (re.compile(r"\bstay\s+in\s+character\b", re.I), "persona lock-in", 0.85),
)

_CANARY_RULES: tuple[Rule, ...] = (
    (re.compile(r"canary\s+(?:token|string|value|word|phrase)", re.I), "canary token probe", 0.92),
    (re.compile(r"(?:secret|hidden|confidential)\s+(?:token|phrase|word|string|key)\s+(?:above|you\s+were\s+given|in\s+your|from\s+your)", re.I), "secret extraction probe", 0.90),
)

_QUOTED = re.compile(r"```.*?```|`[^`\n]+`|\"[^\"\n]{3,200}\"|'[^'\n]{6,200}'|“[^”\n]{3,200}”", re.S)
_EDUCATIONAL = re.compile(
    r"\b(?:how\s+(?:do|can|would|should)\s+(?:i|we|you)|what\s+(?:does|do|is|are)\b|why\s+does|explain|regex|regular\s+expression|escape|sanitiz|unit\s+test|parser|parsing|tokeni[sz]|documentation|example\s+of|difference\s+between|write\s+a\s+(?:function|script|validator))\b",
    re.I,
)
_REPORTING = re.compile(
    r"\b(?:summari[sz]e|triage|classify|moderate|review|translate)\b|\b(?:the\s+)?(?:user|customer|attacker|ticket|email|transcript|log|report)\s+(?:wrote|said|says|sent|contains|submitted)\b|\blog\s+entry\b|\bflagged\s+(?:this|the)\b",
    re.I,
)

_INVISIBLE = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}[\s:,-]?){16,}")
_LONG_TOKEN = re.compile(r"\S{24,}")

_MIN_CONFIDENCE = 0.20
_ENTROPY_FLOOR = 4.1


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalize(raw: str) -> tuple[str, list[Signal]]:
    """Strip invisible characters and fold homoglyphs *before* lexical matching.

    Returns the normalised text plus any signals raised by the normalisation
    itself (an attacker hiding `ignore previous instructions` behind zero-width
    joiners produces both an obfuscation signal and, once folded, an override
    signal -- the second one is what blocks).
    """
    signals: list[Signal] = []
    stripped = _INVISIBLE.sub("", raw)
    if stripped != raw:
        signals.append(
            Signal(
                threat=ThreatType.HIGH_ENTROPY_OBFUSCATION,
                confidence=0.60,
                reason=f"{len(raw) - len(stripped)} invisible/bidi control character(s) removed",
            )
        )
    folded = unicodedata.normalize("NFKC", stripped)
    if folded != stripped:
        signals.append(
            Signal(
                threat=ThreatType.HIGH_ENTROPY_OBFUSCATION,
                confidence=0.55,
                reason="unicode homoglyphs folded via NFKC",
            )
        )
    return folded, signals


def shannon_entropy(text: str) -> float:
    """Bits per character. English prose sits near 3.5-4.0; base64 above 4.8."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------- #
# Context-aware scoring
# --------------------------------------------------------------------------- #


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _QUOTED.finditer(text)]


def _discount(text: str, start: int, spans: list[tuple[int, int]]) -> tuple[float, str]:
    """How much benefit of the doubt a hit at `start` earns from its context."""
    quoted = any(lo <= start < hi for lo, hi in spans)
    framed = bool(_EDUCATIONAL.search(text) or _REPORTING.search(text))
    if quoted and framed:
        return 0.50, "quoted inside an educational/reporting request"
    if framed:
        return 0.25, "educational/reporting framing"
    if quoted:
        return 0.15, "appears inside a quoted span"
    return 0.0, ""


def _scan(text: str, rules: tuple[Rule, ...], threat: ThreatType, *, discountable: bool = True) -> list[Signal]:
    spans = _quoted_spans(text) if discountable else []
    out: list[Signal] = []
    for pattern, label, base in rules:
        match = pattern.search(text)
        if match is None:
            continue
        penalty, why = (_discount(text, match.start(), spans) if discountable else (0.0, ""))
        confidence = max(_MIN_CONFIDENCE, base - penalty)
        excerpt = match.group(0).strip()[:48]
        reason = f"{label}: {excerpt!r}"
        if why:
            reason += f" (discounted -{penalty:.2f}: {why})"
        out.append(Signal(threat=threat, confidence=round(confidence, 3), reason=reason))
    return out


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


def detect_delimiters(text: str) -> list[Signal]:
    """System-prompt boundary escapes: ChatML, Llama, pseudo-XML, role prefixes."""
    return _scan(text, _DELIMITER_RULES, ThreatType.INJECTION_DELIMITER)


def detect_override_phrases(text: str) -> list[Signal]:
    """Direct command negation, safety bypass and system-prompt exfiltration."""
    signals = _scan(text, _OVERRIDE_RULES, ThreatType.SYSTEM_OVERRIDE)
    signals += _scan(text, _ROLEPLAY_RULES, ThreatType.ADVERSARIAL_ROLEPLAY)
    return signals


def detect_canary_leak(text: str, canary: str) -> list[Signal]:
    """Verbatim canary echo, or coaxing language aimed at the canary."""
    signals: list[Signal] = []
    if canary and canary.lower() in text.lower():
        signals.append(
            Signal(
                threat=ThreatType.CANARY_LEAK,
                confidence=0.99,
                reason=f"verbatim canary token {canary!r} present in user input",
            )
        )
    signals += _scan(text, _CANARY_RULES, ThreatType.CANARY_LEAK)
    return signals


def detect_obfuscation(text: str, canary: str) -> list[Signal]:
    """Encoded payloads: base64/hex blocks are decoded and re-scanned in place."""
    signals: list[Signal] = []
    for pattern, decoder, label in (
        (_B64, _try_b64, "base64"),
        (_HEX, _try_hex, "hex"),
    ):
        for match in pattern.finditer(text):
            blob = match.group(0)
            decoded = decoder(blob)
            if decoded is None:
                continue
            inner = detect_delimiters(decoded) + detect_override_phrases(decoded) + detect_canary_leak(decoded, canary)
            if inner:
                worst = max(inner, key=lambda s: s.confidence)
                signals.append(
                    Signal(
                        threat=ThreatType.HIGH_ENTROPY_OBFUSCATION,
                        confidence=min(0.98, worst.confidence + 0.02),
                        reason=f"{label} payload decodes to an attack: {worst.reason}",
                    )
                )
            else:
                signals.append(
                    Signal(
                        threat=ThreatType.HIGH_ENTROPY_OBFUSCATION,
                        confidence=0.62,
                        reason=f"{label} payload decodes to text ({decoded.strip()[:40]!r})",
                    )
                )

    if not signals:
        for match in _LONG_TOKEN.finditer(text):
            token = match.group(0)
            entropy = shannon_entropy(token)
            if entropy >= _ENTROPY_FLOOR:
                signals.append(
                    Signal(
                        threat=ThreatType.HIGH_ENTROPY_OBFUSCATION,
                        confidence=0.50,
                        reason=f"high-entropy token ({entropy:.2f} bits/char, len {len(token)}) -- possible encoded payload",
                    )
                )
                break
    return signals


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    return sum(32 <= b < 127 or b in (9, 10, 13) for b in data) / len(data)


def _try_b64(blob: str) -> str | None:
    padded = blob + "=" * (-len(blob) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 8 or _printable_ratio(raw) < 0.85:
        return None
    return raw.decode("utf-8", errors="replace")


def _try_hex(blob: str) -> str | None:
    cleaned = re.sub(r"[^0-9a-fA-F]", "", blob)
    if len(cleaned) % 2 or len(cleaned) < 16:
        cleaned = cleaned[: len(cleaned) - (len(cleaned) % 2)]
    try:
        raw = bytes.fromhex(cleaned)
    except ValueError:
        return None
    if len(raw) < 8 or _printable_ratio(raw) < 0.85:
        return None
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FastPathReport:
    verdict: Verdict
    top: Signal
    signals: tuple[Signal, ...]
    normalized: str
    sanitized: str
    latency_ms: float

    @property
    def borderline(self) -> bool:
        return self.verdict is not Verdict.BLOCK and self.top.threat is not ThreatType.BENIGN


BENIGN_SIGNAL = Signal(threat=ThreatType.BENIGN, confidence=0.0, reason="no deterministic signal")


def sanitize(text: str) -> str:
    """Defang a prompt we are willing to forward: strip control tokens/phrases."""
    out = _INVISIBLE.sub("", text)
    for rules, tag in ((_DELIMITER_RULES, "[REDACTED_DELIMITER]"), (_OVERRIDE_RULES, "[REDACTED_INSTRUCTION]")):
        for pattern, _, _ in rules:
            out = pattern.sub(tag, out)
    return out


def inspect(prompt: str, config: GatewayConfig | None = None) -> FastPathReport:
    """Run every deterministic detector. Target budget: < 5ms, typical < 0.5ms."""
    cfg = config or GatewayConfig()
    started = perf_counter()

    normalized, signals = normalize(prompt)
    signals += detect_delimiters(normalized)
    signals += detect_override_phrases(normalized)
    signals += detect_canary_leak(normalized, cfg.canary_token)
    signals += detect_obfuscation(normalized, cfg.canary_token)

    top = max(signals, key=lambda s: s.confidence) if signals else BENIGN_SIGNAL
    if top.confidence >= cfg.block_confidence:
        verdict = Verdict.BLOCK
    elif top.confidence >= cfg.escalate_confidence:
        verdict = Verdict.SANITIZE
    else:
        verdict = Verdict.ALLOW

    return FastPathReport(
        verdict=verdict,
        top=top,
        signals=tuple(sorted(signals, key=lambda s: s.confidence, reverse=True)),
        normalized=normalized,
        sanitized=sanitize(normalized),
        latency_ms=(perf_counter() - started) * 1000.0,
    )
