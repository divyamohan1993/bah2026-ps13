"""Local ``llama-server`` client — grammar-constrained, localhost-only, no egress.

:class:`LlamaCppClient` talks to a llama.cpp ``llama-server`` over its
OpenAI-compatible ``/v1/chat/completions`` API, constraining decoding with the
**GBNF grammar** derived from :class:`~netra.contracts.CopilotResponse` so the
model *cannot* emit malformed JSON or an out-of-vocabulary issue class
(research 05 §3-§4). It is the production "lit-up" path; when the server is not
running the orchestrator silently uses :class:`~netra.copilot.llm.template_client.TemplateClient`.

Verifiable no-egress (the 20% compliance lever):
  * The base URL is **forced to a loopback host** at construction
    (:func:`_assert_loopback`) — a non-localhost URL raises immediately, so this
    client can never be pointed at a remote API.
  * No other network calls are made; ``httpx`` is import-guarded so the module
    imports fine on the light tier even if ``httpx`` is absent (``available()``
    then returns False and the template fallback is used).

Grounding/robustness:
  * ``confidence_score`` and ``time_to_impact_minutes`` from the model are
    **overwritten** with the authoritative analytics values from
    :class:`~netra.copilot.llm.base.CopilotGrounding` — the LLM explains the
    numbers, it never gets to fabricate them.
  * The response is validated against the contract; on validation failure one
    stricter constrained retry is attempted, then the call falls back to the
    deterministic template client so the pipeline never returns an invalid object.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlparse

from netra.contracts import (
    AffectedScope,
    CopilotAction,
    CopilotResponse,
    CopilotSignal,
    IssueType,
    Urgency,
)

from .base import CopilotGrounding, CopilotPrompt, LLMClient
from .grammar import copilot_gbnf
from .template_client import TemplateClient

try:  # heavy/optional — but httpx is in the core tier; guard anyway.
    import httpx  # type: ignore

    _HAS_HTTPX = True
except Exception:  # pragma: no cover - exercised only when httpx is missing
    httpx = None  # type: ignore
    _HAS_HTTPX = False

#: Hosts considered loopback (localhost only — enforces no-egress).
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _assert_loopback(base_url: str) -> None:
    """Raise unless ``base_url`` points at a loopback host (no-egress guard)."""
    host = urlparse(base_url).hostname or ""
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"LlamaCppClient refuses a non-loopback base_url ({base_url!r}); "
            "the air-gapped copilot may only talk to a local llama-server. "
            f"Allowed hosts: {sorted(_LOOPBACK_HOSTS)}."
        )


class LlamaCppClient(LLMClient):
    """OpenAI-compatible client for a local, grammar-constrained llama-server."""

    is_fallback = False

    def __init__(
        self,
        base_url: str | None = None,
        *,
        model_id: str = "qwen2.5-7b-instruct-q4_k_m",
        timeout_s: float = 60.0,
        connect_timeout_s: float = 1.0,
    ) -> None:
        """Configure the client (does not connect yet).

        Parameters
        ----------
        base_url:
            llama-server base URL; defaults to ``$NETRA_LLAMA_URL`` or
            ``http://127.0.0.1:8080``. Must be loopback.
        model_id:
            Identifier reported in ``CopilotResponse.model_id``.
        timeout_s / connect_timeout_s:
            Read / connect timeouts; the short connect timeout keeps
            :meth:`available` fast when the server is down.
        """
        self.base_url = (
            base_url or os.environ.get("NETRA_LLAMA_URL", "http://127.0.0.1:8080")
        ).rstrip("/")
        _assert_loopback(self.base_url)
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._grammar = copilot_gbnf()
        self._fallback = TemplateClient()

    # -- health / selection -----------------------------------------------------
    def available(self) -> bool:
        """Return True if httpx is present and the local server answers a health probe."""
        if not _HAS_HTTPX:
            return False
        for path in ("/health", "/v1/models"):
            try:
                resp = httpx.get(
                    f"{self.base_url}{path}",
                    timeout=httpx.Timeout(self.connect_timeout_s),
                )
                if resp.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    # -- generation -------------------------------------------------------------
    def complete_copilot(
        self, prompt: CopilotPrompt, grounding: CopilotGrounding
    ) -> CopilotResponse:
        """Generate a validated response; fall back to the template on any failure."""
        if not _HAS_HTTPX:
            return self._fallback.complete_copilot(prompt, grounding)

        for attempt in range(2):  # one constrained retry, then fall back
            strict = attempt == 1
            try:
                raw = self._chat(prompt, strict=strict)
                obj = json.loads(raw)
                return self._assemble(obj, grounding)
            except Exception:
                continue
        # Both attempts failed -> deterministic template fallback (never invalid).
        return self._fallback.complete_copilot(prompt, grounding)

    def _chat(self, prompt: CopilotPrompt, *, strict: bool) -> str:
        """POST one grammar-constrained chat completion; return the content string."""
        system = prompt.system
        if strict:
            system += (
                "\n\nSTRICT RETRY: emit ONLY the JSON object conforming to the "
                "schema. Cite ONLY ids present in the CONTEXT. Do not add prose "
                "outside the JSON."
            )
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
            # llama.cpp native GBNF constraint (the structural guarantee).
            "grammar": self._grammar,
        }
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- map raw model JSON -> the full contract (authoritative fields injected) -
    def _assemble(self, obj: dict, g: CopilotGrounding) -> CopilotResponse:
        """Build a validated CopilotResponse, enforcing grounded numeric fields."""
        # Closed-set citation enforcement: keep only ids actually in context.
        universe = set(g.citation_universe)
        cited = [c for c in obj.get("citations", []) if c in universe]
        insufficient = bool(obj.get("insufficient_context", False))
        if not cited:
            # Nothing valid cited -> treat as abstain with a sentinel citation.
            cited = ["no-context"]
            insufficient = True

        actions = [
            CopilotAction(
                step=a.get("step", "Review the predicted issue and gather data."),
                runbook_ref=a.get("runbook_ref"),
                urgency=Urgency(a.get("urgency", Urgency.SOON.value)),
                requires_approval=bool(a.get("requires_approval", True)),
            )
            for a in obj.get("recommended_actions", [])
        ]
        if not actions:  # contract requires >=1
            actions = [
                CopilotAction(
                    step="Gather more data and escalate; evidence was insufficient.",
                    urgency=Urgency.SOON,
                    requires_approval=False,
                )
            ]

        signals = [
            CopilotSignal(
                signal=s.get("signal", "unknown"),
                observation=s.get("observation", ""),
                shap_contribution=s.get("shap_contribution"),
            )
            for s in obj.get("contributing_signals", [])
        ]

        scope_obj = obj.get("affected_scope") or {}
        affected_scope = AffectedScope(
            sites=list(scope_obj.get("sites", []) or g.affected_scope.sites),
            devices=list(scope_obj.get("devices", []) or g.affected_scope.devices),
            services_or_vpns=list(
                scope_obj.get("services_or_vpns", [])
                or g.affected_scope.services_or_vpns
            ),
        )

        # predicted_issue: trust the grammar's closed set but prefer the analytics
        # engine's class when the model emitted the healthy/none default.
        issue = IssueType(obj.get("predicted_issue", g.predicted_issue.value))
        if issue == IssueType.NONE and g.predicted_issue != IssueType.NONE:
            issue = g.predicted_issue

        root_cause = (obj.get("root_cause_hypothesis") or "").strip()
        if not root_cause:
            root_cause = (
                g.root_cause_hypothesis
                or f"Predicted {issue.value}; see cited context."
            )

        return CopilotResponse(
            request_id=g.request_id,
            predicted_issue=issue,
            # AUTHORITATIVE: confidence + ETA come from analytics, not the model.
            confidence_score=max(0.0, min(1.0, g.confidence_score)),
            time_to_impact_minutes=g.time_to_impact_minutes,
            root_cause_hypothesis=root_cause[:1200],
            contributing_signals=signals or list(g.contributing_signals),
            affected_scope=affected_scope,
            recommended_actions=actions,
            citations=list(dict.fromkeys(cited)),
            insufficient_context=insufficient,
            used_fallback=False,
            model_id=self.model_id,
        )


__all__ = ["LlamaCppClient"]
