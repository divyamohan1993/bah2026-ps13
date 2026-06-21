"""GBNF grammar + JSON schema derived from the ``CopilotResponse`` contract.

``llama-server`` can constrain decoding either with a **GBNF grammar** (passed as
``grammar``) or a **JSON schema** (passed as ``response_format: json_schema``).
Both guarantee the model emits schema-valid JSON at the *sampler* level — it
cannot produce malformed JSON or an out-of-vocabulary ``predicted_issue``
(research 05 §4). We provide both here:

  * :func:`copilot_json_schema` — built directly from the Pydantic contract via
    ``CopilotResponse.model_json_schema()`` and then **tightened** (the LLM only
    needs to emit the small generative subset of fields; the post-gen fields like
    ``grounding_score`` / ``used_fallback`` are filled by the client, not the
    model). Using the contract as the source means the schema can never drift
    from :mod:`netra.contracts.copilot`.
  * :func:`copilot_gbnf` — a hand-maintained GBNF mirroring that subset, kept in
    lockstep with the closed :class:`~netra.contracts.IssueType` and
    :class:`~netra.contracts.Urgency` enums (which are *generated from the
    contract* so the alternation can never go stale). The static file
    ``grammar.gbnf`` is written from this function.

Only light deps (pydantic) are needed; importing this module never pulls in
llama.cpp.
"""

from __future__ import annotations

from netra.contracts import CopilotResponse, IssueType, Urgency

# --- the generative subset the model is asked to produce -----------------------
# The orchestrator/client own the rest (request_id echo, grounding_score,
# used_fallback, model_id). Keeping the model's surface minimal improves both
# latency and grounding (fewer free-form fields to hallucinate into).
_LLM_EMITTED_FIELDS: tuple[str, ...] = (
    "predicted_issue",
    "confidence_score",
    "time_to_impact_minutes",
    "root_cause_hypothesis",
    "contributing_signals",
    "affected_scope",
    "recommended_actions",
    "citations",
    "insufficient_context",
)


def _enum_alternation(values: list[str]) -> str:
    """Render a list of string literals as a GBNF alternation of quoted tokens."""
    return " | ".join('"\\"' + v + '\\""' for v in values)


def issue_type_values() -> list[str]:
    """The closed IssueType vocabulary (drives the grammar + schema enum)."""
    return [e.value for e in IssueType]


def urgency_values() -> list[str]:
    """The closed Urgency vocabulary for recommended-action steps."""
    return [e.value for e in Urgency]


def copilot_json_schema(*, llm_subset: bool = True) -> dict:
    """Return a JSON schema for the copilot response.

    Parameters
    ----------
    llm_subset:
        When True (default) return the tightened schema covering only the fields
        the model must generate (:data:`_LLM_EMITTED_FIELDS`), with
        ``additionalProperties: false`` so the grammar/JSON-schema constraint
        forbids stray keys. When False, return the full contract schema
        (``CopilotResponse.model_json_schema()``) unmodified — useful for docs
        and for clients that prefer to validate the whole object.
    """
    full = CopilotResponse.model_json_schema()
    if not llm_subset:
        return full

    props = full.get("properties", {})
    subset_props = {k: props[k] for k in _LLM_EMITTED_FIELDS if k in props}
    return {
        "type": "object",
        "properties": subset_props,
        "required": list(_LLM_EMITTED_FIELDS),
        "additionalProperties": False,
        "$defs": full.get("$defs", {}),
    }


def copilot_gbnf() -> str:
    """Return the GBNF grammar string constraining the model's JSON output.

    Mirrors :func:`copilot_json_schema` (the LLM subset). The ``issue`` and
    ``urgency`` non-terminals are built from the live enums so they can never
    drift from the contract.
    """
    issue_alt = _enum_alternation(issue_type_values())
    urgency_alt = _enum_alternation(urgency_values())

    return f'''# GBNF grammar for netra.contracts.CopilotResponse (LLM-emitted subset).
# Generated from the Pydantic contract by netra.copilot.llm.grammar.copilot_gbnf().
# Guarantees llama-server emits schema-valid JSON committed to the closed
# IssueType / Urgency vocabularies. Keep in lockstep with the contract.

root        ::= "{{" ws
                  "\\"predicted_issue\\":" ws issue "," ws
                  "\\"confidence_score\\":" ws unit "," ws
                  "\\"time_to_impact_minutes\\":" ws ttimpact "," ws
                  "\\"root_cause_hypothesis\\":" ws string "," ws
                  "\\"contributing_signals\\":" ws signals "," ws
                  "\\"affected_scope\\":" ws scope "," ws
                  "\\"recommended_actions\\":" ws actions "," ws
                  "\\"citations\\":" ws citations "," ws
                  "\\"insufficient_context\\":" ws boolean ws
                "}}"

issue       ::= {issue_alt}
urgency     ::= {urgency_alt}

# confidence_score is a probability in [0,1]; the grammar admits any JSON number
# and the client clamps/derives the authoritative value from the analytics engine.
unit        ::= number
ttimpact    ::= "null" | number

signals     ::= "[" ws ( signal ( "," ws signal )* ws )? "]"
signal      ::= "{{" ws
                  "\\"signal\\":" ws string "," ws
                  "\\"observation\\":" ws string "," ws
                  "\\"shap_contribution\\":" ws ( "null" | number ) ws
                "}}"

scope       ::= "{{" ws
                  "\\"sites\\":" ws strlist "," ws
                  "\\"devices\\":" ws strlist "," ws
                  "\\"services_or_vpns\\":" ws strlist ws
                "}}"

actions     ::= "[" ws action ( "," ws action )* ws "]"
action      ::= "{{" ws
                  "\\"step\\":" ws string "," ws
                  "\\"runbook_ref\\":" ws ( "null" | string ) "," ws
                  "\\"urgency\\":" ws urgency "," ws
                  "\\"requires_approval\\":" ws boolean ws
                "}}"

citations   ::= "[" ws string ( "," ws string )* ws "]"
strlist     ::= "[" ws ( string ( "," ws string )* ws )? "]"

boolean     ::= "true" | "false"
number      ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
string      ::= "\\"" ( [^"\\\\] | "\\\\" . )* "\\""
ws          ::= [ \\t\\n]*
'''


__all__ = [
    "copilot_gbnf",
    "copilot_json_schema",
    "issue_type_values",
    "urgency_values",
]
