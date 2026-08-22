"""Profile and normalize three intentionally messy Excel packages."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR

SCENARIO_NAMES = {"actual": "Actual", "budget": "Budget", "forecast": "Forecast", "prior year": "Prior Year"}
LEDGER_KEY = ["business_unit", "period", "scenario", "source_metric"]
VALUE_GROUP = ["business_unit", "source_metric", "scenario"]
ROBUST_Z_LIMIT = 6.0


def quality_rule_catalog() -> pd.DataFrame:
    """Return the control catalog used by cleaning and shown in audit outputs."""
    return pd.DataFrame(
        [
            ("Column aliases", "Normalize case/spacing/punctuation, then map only approved source aliases to the canonical schema."),
            ("Scenario labels", "Trim whitespace, collapse spaces and map approved labels to Actual, Budget, Forecast or Prior Year."),
            ("Missing amounts", "Sort by month and linearly interpolate only within the same business, metric and scenario; fail if unresolved."),
            ("Exact duplicates", "Remove rows only when every standardized field and amount is identical."),
            ("Conflicting keys", "Fail the pipeline when non-identical rows share business, month, scenario and metric."),
            ("Extreme values", "Within the same business, metric and scenario, flag values beyond the median/MAD robust range, then interpolate in-group."),
            ("Dates", "Parse source dates and standardize them to month end; fail when a required date cannot be parsed."),
        ],
        columns=["issue_type", "deterministic_repair_rule"],
    )


def _header_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _rename_aliases(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    rename = {column: aliases[_header_key(column)] for column in frame.columns if _header_key(column) in aliases}
    return frame.rename(columns=rename)


def _standardize_scenario(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return normalized.map(SCENARIO_NAMES)


def _robust_outlier_mask(frame: pd.DataFrame, value_column: str, groups: list[str]) -> pd.Series:
    """Flag values outside a same-group median/MAD range."""
    values = pd.to_numeric(frame[value_column], errors="coerce")
    grouped = values.groupby([frame[column] for column in groups])
    medians = grouped.transform("median")
    absolute_deviation = (values - medians).abs()
    mad = absolute_deviation.groupby([frame[column] for column in groups]).transform("median")
    robust_scale = 1.4826 * mad
    # A relative floor keeps nearly flat finance series from treating normal cents-level movement as an outlier.
    tolerance = pd.concat([ROBUST_Z_LIMIT * robust_scale, medians.abs() * 0.50], axis=1).max(axis=1)
    return values.notna() & tolerance.gt(0) & absolute_deviation.gt(tolerance)


def _finalize_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, auditable repairs without crossing metric/scenario groups."""
    frame = frame.copy()
    frame["source_metric"] = frame["source_metric"].astype("string").str.strip()
    frame["amount_millions"] = pd.to_numeric(frame["amount_millions"], errors="coerce")
    frame = frame.drop_duplicates(keep="first")
    if frame.duplicated(LEDGER_KEY, keep=False).any():
        raise ValueError("Conflicting rows share a business key; only exact duplicate rows may be removed.")
    extreme = _robust_outlier_mask(frame, "amount_millions", VALUE_GROUP)
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
    })
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
    })
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
        frame = _rename_aliases(frame, {"scenario": "scenario", "planscenario": "scenario", "case": "scenario"})
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
    return int(_robust_outlier_mask(normalized, value_column, groups).sum())


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
