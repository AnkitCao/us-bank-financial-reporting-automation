"""Profile and normalize three intentionally messy Excel packages."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.llm_data_quality import review_outlier_candidates, review_semantic_mappings
from src.paths import RAW_DIR

SCENARIO_NAMES = {"actual": "Actual", "budget": "Budget", "forecast": "Forecast", "prior year": "Prior Year"}
LEDGER_KEY = ["business_unit", "period", "scenario", "source_metric"]
VALUE_GROUP = ["business_unit", "source_metric", "scenario"]
IQR_MULTIPLIER = 1.5
RELATIVE_MATERIALITY_FLOOR = 0.50


def quality_rule_catalog() -> pd.DataFrame:
    """Return the control catalog used by cleaning and shown in audit outputs."""
    return pd.DataFrame(
        [
            ("Column aliases", "Apply approved aliases; LLM reviews unfamiliar headers against an allowlist; reject unresolved required fields."),
            ("Scenario labels", "Normalize known labels; LLM reviews unfamiliar spellings against the allowed scenarios; reject unresolved labels."),
            ("Missing amounts", "Sort by month and linearly interpolate only within the same business, metric and scenario; fail if unresolved."),
            ("Exact duplicates", "Remove rows only when every standardized field and amount is identical."),
            ("Conflicting keys", "Fail the pipeline when non-identical rows share business, month, scenario and metric."),
            ("Extreme values", "Within the same business, metric and scenario, calculate Q1 and Q3; flag values outside the 1.5×IQR fence after applying a 50% median materiality floor, then impute in-group."),
            ("Dates", "Parse source dates and standardize them to month end; fail when a required date cannot be parsed."),
        ],
        columns=["issue_type", "deterministic_repair_rule"],
    )


def _header_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _rename_aliases(
    frame: pd.DataFrame,
    aliases: dict[str, str],
    *,
    allowed_targets: list[str] | None = None,
    context: str = "",
) -> pd.DataFrame:
    rename = {column: aliases[_header_key(column)] for column in frame.columns if _header_key(column) in aliases}
    if allowed_targets:
        unknown = [str(column) for column in frame.columns if column not in rename]
        reviewed = review_semantic_mappings(
            kind="column headers", values=unknown, allowed_targets=allowed_targets, context=context
        )
        if reviewed:
            rename.update({column: reviewed[column] for column in frame.columns if column in reviewed})
    return frame.rename(columns=rename)


def _standardize_scenario(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    standardized = normalized.map(SCENARIO_NAMES)
    unresolved = normalized.loc[standardized.isna() & normalized.notna()].drop_duplicates().tolist()
    reviewed = review_semantic_mappings(
        kind="scenario labels",
        values=[str(value) for value in unresolved],
        allowed_targets=list(SCENARIO_NAMES.values()),
        context="Allowed reporting scenarios are Actual, Budget, Forecast and Prior Year.",
    )
    if reviewed:
        standardized = standardized.fillna(normalized.map(reviewed))
    return standardized


def _require_columns(frame: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Unresolved required fields in {source_name}: {missing}")


def _iqr_outlier_details(
    frame: pd.DataFrame, value_column: str, groups: list[str]
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return an IQR candidate mask and its same-group statistics."""
    values = pd.to_numeric(frame[value_column], errors="coerce")
    grouped = values.groupby([frame[column] for column in groups])
    q1 = grouped.transform(lambda series: series.quantile(0.25))
    q3 = grouped.transform(lambda series: series.quantile(0.75))
    iqr = q3 - q1
    medians = grouped.transform("median")
    # The floor prevents gradual finance trends from being mistaken for data corruption.
    fence_width = pd.concat(
        [IQR_MULTIPLIER * iqr, medians.abs() * RELATIVE_MATERIALITY_FLOOR], axis=1
    ).max(axis=1)
    lower_fence = q1 - fence_width
    upper_fence = q3 + fence_width
    mask = values.notna() & fence_width.gt(0) & (values.lt(lower_fence) | values.gt(upper_fence))
    return mask, q1, q3, lower_fence, upper_fence


def _iqr_outlier_mask(frame: pd.DataFrame, value_column: str, groups: list[str]) -> pd.Series:
    """Flag values outside a same-group IQR range."""
    return _iqr_outlier_details(frame, value_column, groups)[0]


def _finalize_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, auditable repairs without crossing metric/scenario groups."""
    frame = frame.copy()
    frame["source_metric"] = frame["source_metric"].astype("string").str.strip()
    frame["amount_millions"] = pd.to_numeric(frame["amount_millions"], errors="coerce")
    frame = frame.drop_duplicates(keep="first")
    if frame.duplicated(LEDGER_KEY, keep=False).any():
        raise ValueError("Conflicting rows share a business key; only exact duplicate rows may be removed.")
    extreme, q1, q3, lower_fence, upper_fence = _iqr_outlier_details(
        frame, "amount_millions", VALUE_GROUP
    )
    if extreme.any():
        values = pd.to_numeric(frame["amount_millions"], errors="coerce")
        grouped = values.groupby([frame[column] for column in VALUE_GROUP])
        medians = grouped.transform("median")
        candidates = []
        for position in frame.index[extreme]:
            same_group = pd.Series(True, index=frame.index)
            for column in VALUE_GROUP:
                same_group &= frame[column].eq(frame.at[position, column])
            peers = values.loc[same_group & frame.index.to_series().ne(position)].dropna()
            candidates.append({
                "position": int(position),
                "business_unit": str(frame.at[position, "business_unit"]),
                "metric": str(frame.at[position, "source_metric"]),
                "scenario": str(frame.at[position, "scenario"]),
                "period": str(frame.at[position, "period"]),
                "value": round(float(values.at[position]), 6),
                "group_median": round(float(medians.at[position]), 6),
                "q1": round(float(q1.at[position]), 6),
                "q3": round(float(q3.at[position]), 6),
                "iqr_lower_fence": round(float(lower_fence.at[position]), 6),
                "iqr_upper_fence": round(float(upper_fence.at[position]), 6),
                "peer_min": round(float(peers.min()), 6) if not peers.empty else None,
                "peer_max": round(float(peers.max()), 6) if not peers.empty else None,
                "prior_value": round(float(peers.iloc[-2]), 6) if len(peers) >= 2 else None,
                "next_value": round(float(peers.iloc[-1]), 6) if len(peers) >= 1 else None,
            })
        approved_positions = review_outlier_candidates({"candidates": candidates})
        if approved_positions is not None:
            extreme = frame.index.to_series().isin(approved_positions)
    frame.loc[extreme, "amount_millions"] = pd.NA
    frame = frame.sort_values(VALUE_GROUP + ["period"])
    frame["amount_millions"] = frame.groupby(VALUE_GROUP, group_keys=False)["amount_millions"].transform(
        lambda values: values.interpolate(limit_direction="both")
    )
    if frame[LEDGER_KEY].isna().any().any() or frame["amount_millions"].isna().any():
        raise ValueError("Cleaning could not repair all required values.")
    return frame[LEDGER_KEY + ["amount_millions"]]


def clean_commercial_banking(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Monthly Detail")
    frame = _rename_aliases(frame, {
        "date": "period", "reportingdate": "period", "metric": "source_metric", "measurename": "source_metric",
        "scenario": "scenario", "plantype": "scenario", "amount": "amount_millions", "valuemm": "amount_millions",
    }, allowed_targets=["period", "source_metric", "scenario", "amount_millions"], context="Commercial Banking long-form monthly ledger.")
    _require_columns(frame, ["period", "source_metric", "scenario", "amount_millions"], path.name)
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    frame["scenario"] = _standardize_scenario(frame["scenario"])
    frame["business_unit"] = "Commercial Banking"
    return _finalize_ledger(frame[["business_unit", "period", "scenario", "source_metric", "amount_millions"]])


def clean_commercial_real_estate(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="CRE Monthly")
    frame = _rename_aliases(frame, {
        "period": "period", "reportmonth": "period", "case": "scenario", "plancase": "scenario",
        "nii": "NII", "netinterestincome": "NII", "origfee": "Orig Fee", "originationfees": "Orig Fee",
        "prepayfee": "Prepay Fee", "prepaymentfees": "Prepay Fee", "servicingfee": "Servicing Fee",
        "otherfee": "Other Fee", "creditlossprovision": "Credit Loss Provision", "opex": "Opex",
        "operatingcosts": "Opex", "creloanbalmm": "CRE Loan Bal ($MM)", "nplproxymm": "NPL Proxy ($MM)",
    }, allowed_targets=[
        "period", "scenario", "NII", "Orig Fee", "Prepay Fee", "Servicing Fee", "Other Fee",
        "Credit Loss Provision", "Opex", "CRE Loan Bal ($MM)", "NPL Proxy ($MM)",
    ], context="Commercial Real Estate wide monthly financial table.")
    _require_columns(frame, ["period", "scenario"], path.name)
    metric_names = {
        "NII": "Interest Income", "Orig Fee": "Origination Fee", "Prepay Fee": "Prepayment Fee",
        "Servicing Fee": "Loan Servicing Fee", "Other Fee": "Other Fee Income",
        "Credit Loss Provision": "Credit Provision", "Opex": "Operating Expense",
        "CRE Loan Bal ($MM)": "Commercial Mortgage Balance", "NPL Proxy ($MM)": "Nonperforming Loan Proxy",
    }
    frame["scenario"] = _standardize_scenario(frame["scenario"])
    frame = frame.melt(id_vars=["period", "scenario"], var_name="raw_metric", value_name="amount_millions")
    frame["source_metric"] = frame["raw_metric"].map(metric_names)
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    frame["business_unit"] = "Commercial Real Estate"
    return _finalize_ledger(frame[["business_unit", "period", "scenario", "source_metric", "amount_millions"]])


def clean_capital_markets(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = _rename_aliases(
            frame,
            {"scenario": "scenario", "planscenario": "scenario", "case": "scenario"},
            allowed_targets=["scenario"],
            context=f"Capital Markets sheet {sheet_name}; month columns are dates and only the scenario field should be renamed.",
        )
        _require_columns(frame, ["scenario"], f"{path.name}/{sheet_name}")
        frame["scenario"] = _standardize_scenario(frame["scenario"])
        frame = frame.melt(id_vars="scenario", var_name="period", value_name="amount_millions")
        period_text = frame["period"].astype(str).str.strip().str.replace("_", "-", regex=False)
        frame["period"] = pd.to_datetime(period_text, format="%b-%y", errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        frame["source_metric"] = sheet_name.strip()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["business_unit"] = "Capital Markets"
    return _finalize_ledger(combined[["business_unit", "period", "scenario", "source_metric", "amount_millions"]])


def clean_all_sources(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = [
        clean_commercial_banking(raw_dir / "commercial_banking_monthly.xlsx"),
        clean_commercial_real_estate(raw_dir / "commercial_real_estate_monthly.xlsx"),
        clean_capital_markets(raw_dir / "capital_markets_monthly.xlsx"),
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["business_unit", "period", "scenario", "source_metric"]
    ).reset_index(drop=True)


def _extreme_count(frame: pd.DataFrame, value_column: str, groups: list[str]) -> int:
    normalized = frame.copy()
    for column in groups:
        normalized[column] = normalized[column].astype(str).str.strip().str.lower()
    return int(_iqr_outlier_mask(normalized, value_column, groups).sum())


def profile_raw_sources(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return EDA counts for the deliberately messy raw workbooks."""
    profiles: list[dict[str, object]] = []
    banking = pd.read_excel(raw_dir / "commercial_banking_monthly.xlsx", sheet_name="Monthly Detail")
    profiles.append({
        "business_unit": "Commercial Banking", "structure": "Long table", "rows": len(banking), "columns": len(banking.columns),
        "missing_values": int(banking.isna().sum().sum()), "duplicate_rows": int(banking.duplicated().sum()),
        "duplicate_keys": int(banking.duplicated(["Reporting Date ", "Measure Name", "Plan Type"]).sum()),
        "outliers": _extreme_count(banking, "Value_MM", ["Measure Name", "Plan Type"]),
        "label_issues": int(banking["Plan Type"].astype(str).ne(banking["Plan Type"].astype(str).str.strip()).sum() + banking["Measure Name"].astype(str).ne(banking["Measure Name"].astype(str).str.strip()).sum()),
        "column_aliases": 5,
    })
    real_estate = pd.read_excel(raw_dir / "commercial_real_estate_monthly.xlsx", sheet_name="CRE Monthly")
    numeric_columns = real_estate.select_dtypes(include="number").columns
    cre_outliers = sum(_extreme_count(real_estate.assign(_all="all"), column, ["_all"]) for column in numeric_columns)
    profiles.append({
        "business_unit": "Commercial Real Estate", "structure": "Wide table", "rows": len(real_estate), "columns": len(real_estate.columns),
        "missing_values": int(real_estate.isna().sum().sum()), "duplicate_rows": int(real_estate.duplicated().sum()),
        "duplicate_keys": int(real_estate.duplicated(["Report Month", "Plan Case"]).sum()), "outliers": int(cre_outliers),
        "label_issues": int(real_estate["Plan Case"].astype(str).ne(real_estate["Plan Case"].astype(str).str.strip()).sum()), "column_aliases": 6,
    })
    capital_path = raw_dir / "capital_markets_monthly.xlsx"
    workbook = pd.ExcelFile(capital_path)
    capital_frames = []
    capital_outliers = 0
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(capital_path, sheet_name=sheet)
        frame["Source Metric"] = sheet
        capital_frames.append(frame)
        for column in [column for column in frame.columns if column not in {"Plan Scenario", "Source Metric"}]:
            capital_outliers += _extreme_count(frame.assign(_sheet=sheet), column, ["_sheet"])
    capital = pd.concat(capital_frames, ignore_index=True)
    profiles.append({
        "business_unit": "Capital Markets", "structure": "Cross tab wide workbook", "rows": len(capital), "columns": len(capital.columns),
        "missing_values": int(capital.isna().sum().sum()), "duplicate_rows": int(capital.duplicated().sum()),
        "duplicate_keys": int(capital.duplicated(["Source Metric", "Plan Scenario"]).sum()), "outliers": int(capital_outliers),
        "label_issues": int(capital["Plan Scenario"].astype(str).ne(capital["Plan Scenario"].astype(str).str.strip()).sum()), "column_aliases": 2,
    })
    result = pd.DataFrame(profiles)
    result["problem_records"] = result[["missing_values", "duplicate_rows", "outliers", "label_issues"]].sum(axis=1)
    result["problem_rate"] = result["problem_records"] / result["rows"]
    return result
