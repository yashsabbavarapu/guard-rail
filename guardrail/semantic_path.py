"""Adversarial vector gate: cosine distance to known jailbreak clusters.

Two backends behind one protocol:

* `GeminiEmbedder`   -- real embeddings, taskType=SEMANTIC_SIMILARITY. Used only
  when GEMINI_API_KEY (or GOOGLE_API_KEY) is exported.
* `HashingEmbedder`  -- deterministic, offline, zero-cost lexical proxy. It is a
  hashed bag of stopword-filtered unigrams/bigrams plus low-weight character
  4-grams, so paraphrases of a seeded attack stay close while ordinary business
  questions fall away.

The two backends do NOT share a similarity scale, which is why `GatewayConfig`
carries a threshold for each (see README: "Threshold calibration").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from guardrail.models import GatewayConfig, SemanticAssessment, Signal, ThreatType

Matrix = npt.NDArray[np.float64]

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "jailbreaks.json"

_TOKEN = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have how i if in
    into is it its me my of on or our so that the their them then there these they this to us
    was we were what when which who will with would you your""".split()
)


class EmbeddingBackend(Protocol):
    """Anything that turns text into L2-normalised row vectors."""

    name: str
    is_remote: bool

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def _l2(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised: Matrix = matrix / np.maximum(norms, 1e-12)
    return normalised


class HashingEmbedder:
    """Deterministic offline embedder. Same text -> same vector, always."""

    name = "local-hashing-v1"
    is_remote = False

    def __init__(self, dims: int = 768) -> None:
        self.dims = dims

    def _features(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        words = _TOKEN.findall(lowered)
        content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        feats: dict[str, float] = {}
        for word in content:
            feats[f"u:{word}"] = feats.get(f"u:{word}", 0.0) + 1.0
        for left, right in zip(content, content[1:]):
            key = f"b:{left}_{right}"
            feats[key] = feats.get(key, 0.0) + 0.8
        squashed = " ".join(content)
        for i in range(max(0, len(squashed) - 3)):
            key = f"c:{squashed[i : i + 4]}"
            feats[key] = feats.get(key, 0.0) + 0.25
        return feats

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self.dims), dtype=np.float64)
        for row, text in enumerate(texts):
            for key, weight in self._features(text).items():
                digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dims
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign * (1.0 + np.log(weight)) if weight > 1 else sign * weight
        return _l2(out)


class GeminiEmbedder:
    """Gemini embeddings pinned to taskType=SEMANTIC_SIMILARITY.

    SEMANTIC_SIMILARITY is not optional: RETRIEVAL_* task types optimise for
    query/document asymmetry and collapse the separation between a jailbreak
    and a benign question of the same topic.
    """

    name = "gemini-semantic-similarity"
    is_remote = True

    def __init__(self, api_key: str, model: str | None = None, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("GUARDRAIL_EMBED_MODEL", "gemini-embedding-001")
        self.timeout = timeout
        self.name = f"gemini:{self.model}"

    def embed(self, texts: Sequence[str]) -> Matrix:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":batchEmbedContents?key={self.api_key}"
        )
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "SEMANTIC_SIMILARITY",
                }
                for text in texts
            ]
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        vectors = [np.asarray(item["values"], dtype=np.float64) for item in body["embeddings"]]
        return _l2(np.vstack(vectors))


def build_backend(prefer_remote: bool = True) -> EmbeddingBackend:
    """Remote when a key is exported and reachable, deterministic local otherwise."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if prefer_remote and key:
        remote = GeminiEmbedder(key)
        try:
            remote.embed(["warmup"])
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
            return HashingEmbedder()
        return remote
    return HashingEmbedder()


class SemanticGate:
    """Scores a prompt against attack-family centroids and their exemplars."""

    def __init__(
        self,
        config: GatewayConfig | None = None,
        backend: EmbeddingBackend | None = None,
        fixtures: Path | None = None,
    ) -> None:
        self.config = config or GatewayConfig()
        self.backend = backend or build_backend()
        raw = json.loads((fixtures or FIXTURES).read_text(encoding="utf-8"))
        self.families: list[str] = []
        exemplars: list[str] = []
        owners: list[int] = []
        for index, family in enumerate(raw["families"]):
            self.families.append(str(family["id"]))
            for variant in family["variants"]:
                exemplars.append(str(variant))
                owners.append(index)
        self.exemplar_owner = np.asarray(owners, dtype=np.int64)
        self.exemplars: Matrix = self.backend.embed(exemplars)
        self.centroids: Matrix = _l2(
            np.vstack([self.exemplars[self.exemplar_owner == i].mean(axis=0) for i in range(len(self.families))])
        )

    @property
    def threshold(self) -> float:
        """Backend-specific: the two similarity scales are not interchangeable."""
        return self.config.semantic_threshold if self.backend.is_remote else self.config.local_semantic_threshold

    def score(self, text: str) -> tuple[float, float, int]:
        vector = self.backend.embed([text])
        centroid_sims = (self.centroids @ vector.T).ravel()
        exemplar_sims = (self.exemplars @ vector.T).ravel()
        best_centroid = int(np.argmax(centroid_sims))
        best_exemplar = int(np.argmax(exemplar_sims))
        centroid_top = float(centroid_sims[best_centroid])
        exemplar_top = float(exemplar_sims[best_exemplar])
        owner = best_centroid if centroid_top >= exemplar_top else int(self.exemplar_owner[best_exemplar])
        return centroid_top, exemplar_top, owner

    def inspect(self, text: str, threshold: float | None = None) -> tuple[Signal | None, SemanticAssessment]:
        started = perf_counter()
        centroid_top, exemplar_top, owner = self.score(text)
        gate = self.threshold if threshold is None else threshold
        best = max(centroid_top, exemplar_top)
        assessment = SemanticAssessment(
            backend=self.backend.name,
            family=self.families[owner],
            centroid_similarity=round(centroid_top, 4),
            exemplar_similarity=round(exemplar_top, 4),
            score=round(best, 4),
            threshold=gate,
            latency_ms=(perf_counter() - started) * 1000.0,
        )
        margin = self.config.semantic_review_margin
        if best < gate - margin:
            return None, assessment

        if best >= gate:
            headroom = (best - gate) / max(1e-6, 1.0 - gate)
            confidence = min(0.99, 0.90 + 0.09 * headroom)
            band = "block"
        else:
            # Review band: close enough to an attack cluster to defang, not to drop.
            confidence = 0.60 + 0.29 * (best - (gate - margin)) / max(1e-6, margin)
            band = "review"
        signal = Signal(
            threat=ThreatType.ADVERSARIAL_ROLEPLAY,
            confidence=round(confidence, 3),
            reason=(
                f"semantic {band} match to attack family '{self.families[owner]}' "
                f"(centroid={centroid_top:.3f}, exemplar={exemplar_top:.3f}, threshold={gate:.2f})"
            ),
        )
        return signal, assessment
