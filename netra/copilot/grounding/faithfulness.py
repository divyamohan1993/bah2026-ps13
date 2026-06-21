"""Faithfulness / groundedness scoring — HHEM-2.1 (optional) or heuristic NLI.

A post-generation gate that scores whether the copilot's answer is **supported by
its cited context** (research 05 §5.2). The heavy path is Vectara
**HHEM-2.1-Open** (a DeBERTa-v3 NLI factual-consistency model, <600 MB, CPU) via
``transformers``; given ``(premise=context, hypothesis=claim)`` it returns a
0-1 consistency score. It is import-guarded and loaded offline only.

When the model is absent — the CPU-only default — we fall back to a deterministic
**lexical-overlap NLI heuristic**: each answer claim is scored by the fraction of
its salient (content) tokens that appear in the cited context, which approximates
entailment well enough to flag clearly-ungrounded claims. ``faithfulness`` is the
mean claim score; the orchestrator can gate/retry/abstain on it and writes it to
``CopilotResponse.grounding_score``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")
# Common words excluded from the overlap heuristic so scoring tracks *content*.
_STOPWORDS = frozenset(
    """a an the of to and or is are was were be been being for on in at by with
    this that these those it its as from into than then so such not no nor but if
    can will may should would could about over under within without across your
    you we they he she them their our up down out off via per which who whom whose
    what when where why how also more most less least very much many any some all
    each every both either neither one two three new now before after while during
    has have had do does did done make made get got""".split()
)


def _content_tokens(text: str) -> list[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def _split_claims(text: str) -> list[str]:
    """Split an answer into atomic claims (sentence-ish units) for scoring."""
    parts = re.split(r"(?<=[.;!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


@dataclass
class FaithfulnessResult:
    """The groundedness verdict for one answer against its context."""

    score: float  # mean claim consistency in [0,1]
    claim_scores: list[float]
    backend: str  # "hhem" or "lexical_overlap"
    grounded: bool  # score >= threshold


class FaithfulnessScorer:
    """Score answer-vs-context consistency (HHEM if present, else lexical NLI)."""

    def __init__(
        self,
        *,
        prefer_model: bool = False,
        model_name: str = "vectara/hallucination_evaluation_model",
        threshold: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self.backend = "lexical_overlap"
        if prefer_model:
            self._try_load(model_name)

    def _try_load(self, model_name: str) -> None:
        try:  # optional-heavy: HHEM via transformers
            from transformers import AutoModelForSequenceClassification  # type: ignore

            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.backend = "hhem"
        except Exception:
            self._model = None
            self.backend = "lexical_overlap"

    def score(
        self, *, answer_claims: Sequence[str], context: str
    ) -> FaithfulnessResult:
        """Score each claim against ``context``; return the aggregate verdict."""
        claims = [c for c in answer_claims if c and c.strip()]
        if not claims:
            return FaithfulnessResult(1.0, [], self.backend, True)

        if self.backend == "hhem" and self._model is not None:
            try:
                pairs = [(context, c) for c in claims]
                raw = self._model.predict(pairs)  # type: ignore[attr-defined]
                scores = [float(s) for s in raw]
            except Exception:
                scores = [self._lexical(c, context) for c in claims]
        else:
            scores = [self._lexical(c, context) for c in claims]

        mean = sum(scores) / len(scores)
        return FaithfulnessResult(
            score=mean,
            claim_scores=scores,
            backend=self.backend,
            grounded=mean >= self.threshold,
        )

    @staticmethod
    def _lexical(claim: str, context: str) -> float:
        """Fraction of a claim's content tokens present in the context [0,1]."""
        claim_toks = _content_tokens(claim)
        if not claim_toks:
            return 1.0  # no content to contradict
        ctx = set(_content_tokens(context))
        if not ctx:
            return 0.0
        hits = sum(1 for t in claim_toks if t in ctx)
        return hits / len(claim_toks)

    # Convenience: score a free-text answer (auto-split into claims).
    def score_text(self, *, answer: str, context: str) -> FaithfulnessResult:
        return self.score(answer_claims=_split_claims(answer), context=context)


__all__ = ["FaithfulnessScorer", "FaithfulnessResult"]
