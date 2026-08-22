"""Classify incoming workbook structures before financial cleaning begins."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.llm_data_quality import MIN_MAPPING_CONFIDENCE, _api_key, _extract_json

ALLOWED_STRUCTURES = {"long_table", "wide_table", "cross_tab_workbook"}
ALLOWED_PARSERS = {
    "commercial_banking_long",
    "commercial_real_estate_wide",
    "capital_markets_cross_tab",
}

# A new business unit never enters this registry automatically. Human approval is
# required before its identity, accepted structures, and parser are recorded here.
APPROVED_SOURCES = {
    "commercial_banking_monthly.xlsx": {
        "business_unit": "Commercial Banking",
        "routes": {"long_table": "commercial_banking_long"},
    },
    "commercial_real_estate_monthly.xlsx": {
        "business_unit": "Commercial Real Estate",
        "routes": {"wide_table": "commercial_real_estate_wide"},
    },
    "capital_markets_monthly.xlsx": {
        "business_unit": "Capital Markets",
        "routes": {"cross_tab_workbook": "capital_markets_cross_tab"},
    },
}


class SourcePendingReviewError(ValueError):
    """Raised when a source cannot safely enter an approved cleaning route."""


@dataclass(frozen=True)
class StructureDecision:
    source_name: str
    business_unit: str | None
    structure: str | None
    parser: str | None
    date_field: str | None
    scenario_field: str | None
    confidence: float
    method: str
    status: str
    reason: str

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _workbook_snapshot(path: Path) -> dict[str, object]:
    """Return headers and a tiny sample; financial values are never sent in full."""
    workbook = pd.ExcelFile(path)
    sheets: list[dict[str, object]] = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name, nrows=5)
        sheets.append({
            "sheet_name": sheet_name,
            "columns": [str(column) for column in frame.columns],
            "sample": frame.head(3).where(frame.notna(), None).astype(object).to_dict("records"),
        })
    return {"source_name": path.name, "sheet_count": len(sheets), "sheets": sheets}


def _deterministic_structure(snapshot: dict[str, object]) -> tuple[str | None, str | None, str | None, float]:
    sheets = snapshot["sheets"]
    if len(sheets) > 1:
        scenario_candidates = {"scenario", "planscenario", "case", "plantype", "plancase"}
        if all(any(_key(column) in scenario_candidates for column in sheet["columns"]) for sheet in sheets):
            return "cross_tab_workbook", None, "scenario", 0.99

    columns = list(sheets[0]["columns"])
    keys = {_key(column): str(column) for column in columns}
    date_keys = {"date", "reportingdate", "period", "reportmonth"}
    scenario_keys = {"scenario", "planscenario", "case", "plantype", "plancase"}
    metric_keys = {"metric", "measurename", "sourcemetric"}
    amount_keys = {"amount", "valuemm", "amountmillions"}
    date_field = next((keys[value] for value in date_keys if value in keys), None)
    scenario_field = next((keys[value] for value in scenario_keys if value in keys), None)
    if any(value in keys for value in metric_keys) and any(value in keys for value in amount_keys):
        return "long_table", date_field, scenario_field, 0.99
    if date_field and scenario_field and len(columns) >= 4:
        return "wide_table", date_field, scenario_field, 0.97
    return None, date_field, scenario_field, 0.0


@lru_cache(maxsize=32)
def _llm_classify_cached(payload_json: str, model: str, api_key: str) -> str:
    from openai import OpenAI

    response = OpenAI(api_key=api_key).responses.create(
        model=model,
        instructions=(
            "Classify a financial workbook's structure using only its sheet names, headers, and tiny sample. "
            "Do not calculate, clean, map, or change any value. Choose structure only from long_table, "
            "wide_table, cross_tab_workbook; choose parser only from the supplied allowed parsers or null. "
            "Return JSON only: {\"structure\":\"long_table\",\"parser\":\"allowed parser\","
            "\"date_field\":\"raw header or null\",\"scenario_field\":\"raw header or null\","
            "\"confidence\":0.98,\"reason\":\"short evidence\"}."
        ),
        input=payload_json,
        store=False,
    )
    return response.output_text


def classify_source_structure(path: Path) -> StructureDecision:
    """Classify a source and route only approved source/structure combinations."""
    path = Path(path)
    approved = APPROVED_SOURCES.get(path.name)
    if approved is None:
        return StructureDecision(
            path.name, None, None, None, None, None, 0.0, "registry", "Pending Review",
            "Unrecognized source; human approval is required before adding a business unit or parser route.",
        )

    snapshot = _workbook_snapshot(path)
    structure, date_field, scenario_field, confidence = _deterministic_structure(snapshot)
    method = "deterministic"
    reason = "Headers and workbook layout match an approved structure."

    if structure is None:
        api_key = _api_key()
        if api_key:
            payload = {
                "workbook": snapshot,
                "allowed_structures": sorted(ALLOWED_STRUCTURES),
                "allowed_parsers": sorted(set(approved["routes"].values())),
            }
            try:
                model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
                parsed = _extract_json(_llm_classify_cached(
                    json.dumps(payload, sort_keys=True, default=str), model, api_key
                ))
                proposed_structure = parsed.get("structure")
                proposed_parser = parsed.get("parser")
                proposed_confidence = float(parsed.get("confidence", 0))
                if (
                    proposed_structure in ALLOWED_STRUCTURES
                    and proposed_parser in ALLOWED_PARSERS
                    and approved["routes"].get(proposed_structure) == proposed_parser
                    and proposed_confidence >= MIN_MAPPING_CONFIDENCE
                ):
                    structure = proposed_structure
                    date_field = parsed.get("date_field")
                    scenario_field = parsed.get("scenario_field")
                    confidence = proposed_confidence
                    method = "llm_proposal"
                    reason = str(parsed.get("reason", "Approved route proposed from workbook metadata."))
            except Exception:
                pass

    parser = approved["routes"].get(structure)
    if not parser:
        return StructureDecision(
            path.name, approved["business_unit"], structure, None, date_field, scenario_field,
            confidence, method, "Pending Review",
            "Structure has no human-approved parser route for this business unit.",
        )
    return StructureDecision(
        path.name, approved["business_unit"], structure, parser, date_field, scenario_field,
        confidence, method, "Approved", reason,
    )


def require_approved_source(path: Path, expected_parser: str) -> StructureDecision:
    decision = classify_source_structure(path)
    if decision.status != "Approved" or decision.parser != expected_parser:
        raise SourcePendingReviewError(
            f"Unrecognized Source — Pending Review: {path.name}. {decision.reason}"
        )
    return decision


def require_approved_source_directory(raw_dir: Path) -> list[StructureDecision]:
    """Reject unregistered workbooks instead of silently excluding them."""
    decisions = [classify_source_structure(path) for path in sorted(Path(raw_dir).glob("*.xlsx"))]
    pending = [decision for decision in decisions if decision.status != "Approved"]
    if pending:
        names = ", ".join(decision.source_name for decision in pending)
        raise SourcePendingReviewError(
            f"Unrecognized Source — Pending Review: {names}. Human approval is required before ingestion."
        )
    return decisions
