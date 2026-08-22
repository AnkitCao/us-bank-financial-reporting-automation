"""Constrained LLM review for ambiguous data-quality candidates."""

from __future__ import annotations

import json
import os
from functools import lru_cache

MIN_MAPPING_CONFIDENCE = 0.92


def _api_key() -> str:
    """Read an API key from the process or Streamlit secrets when available."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def _extract_json(text: str) -> dict:
    """Parse a JSON object even when a model wraps it in a markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(cleaned)


@lru_cache(maxsize=32)
def _review_cached(payload_json: str, model: str, api_key: str) -> tuple[int, ...]:
    """Return candidate positions approved by the LLM as genuine anomalies."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You are reviewing financial data-quality candidates, not forecasting business performance. "
            "For each candidate, decide whether the value is a data anomaly using only the supplied value, "
            "same-metric/scenario Q1, Q3, IQR fences, median, and neighboring values. Do not flag legitimate business movement "
            "without strong numerical evidence. Return JSON only in this exact shape: "
            '{"anomaly_positions":[0,1],"reasons":{"0":"short reason"}}. '
            "Positions must come from the supplied candidates."
        ),
        input=payload_json,
        store=False,
    )
    parsed = _extract_json(response.output_text)
    valid_positions = {int(item["position"]) for item in json.loads(payload_json)["candidates"]}
    return tuple(
        position for position in (int(value) for value in parsed.get("anomaly_positions", []))
        if position in valid_positions
    )


def review_outlier_candidates(payload: dict) -> set[int] | None:
    """Ask the configured LLM to confirm candidates; return None when unavailable."""
    api_key = _api_key()
    if not api_key or not payload.get("candidates"):
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    try:
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return set(_review_cached(payload_json, model, api_key))
    except Exception:
        return None


@lru_cache(maxsize=64)
def _mapping_review_cached(payload_json: str, model: str, api_key: str) -> str:
    """Return the raw JSON mapping decision for a semantic-quality review."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You are resolving ambiguous schema or categorical labels in a financial data-quality pipeline. "
            "Use spelling similarity, examples, data type, business meaning, and the supplied allowed targets. "
            "Never invent a target and never alter a numeric value. Return JSON only with this exact shape: "
            '{"mappings":[{"source":"raw value","target":"allowed target","confidence":0.98,"reason":"short reason"}]}. '
            "Use target null when the evidence is ambiguous. Confidence must reflect actual certainty."
        ),
        input=payload_json,
        store=False,
    )
    return response.output_text


def review_semantic_mappings(
    *, kind: str, values: list[str], allowed_targets: list[str], context: str
) -> dict[str, str] | None:
    """Map unfamiliar labels only to allowlisted targets at high confidence."""
    api_key = _api_key()
    if not api_key or not values:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    payload = {
        "review_type": kind,
        "unrecognized_values": sorted(set(str(value) for value in values)),
        "allowed_targets": allowed_targets,
        "context": context,
    }
    try:
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        parsed = _extract_json(_mapping_review_cached(payload_json, model, api_key))
        allowed_sources = set(payload["unrecognized_values"])
        allowed = set(allowed_targets)
        accepted: dict[str, str] = {}
        for item in parsed.get("mappings", []):
            source = str(item.get("source", ""))
            target = item.get("target")
            confidence = float(item.get("confidence", 0))
            if source in allowed_sources and target in allowed and confidence >= MIN_MAPPING_CONFIDENCE:
                accepted[source] = str(target)
        return accepted
    except Exception:
        return None
