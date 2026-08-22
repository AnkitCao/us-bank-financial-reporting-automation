"""Normalize three heterogeneous Excel packages into one long ledger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR


def clean_commercial_banking(path: Path) -> pd.DataFrame:
    """Clean the stacked Commercial Banking source format."""
    frame = pd.read_excel(path, sheet_name="Monthly Detail")
    frame = frame.rename(columns={"Date": "period", "Metric": "source_metric", "Scenario": "scenario", "Amount": "amount_millions"})
    frame["period"] = pd.to_datetime(frame["period"]).dt.to_period("M").dt.to_timestamp("M")
    frame["business_unit"] = "Commercial Banking"
    return frame[["business_unit", "period", "scenario", "source_metric", "amount_millions"]]


def clean_commercial_real_estate(path: Path) -> pd.DataFrame:
    """Unpivot the wide Commercial Real Estate source format."""
    frame = pd.read_excel(path, sheet_name="CRE Monthly")
    frame = frame.rename(columns={"Period": "period", "Case": "scenario"})
    metric_names = {
        "NII": "Interest Income",
        "Orig Fee": "Origination Fee",
        "Prepay Fee": "Prepayment Fee",
        "Servicing Fee": "Loan Servicing Fee",
        "Other Fee": "Other Fee Income",
        "Credit Loss Provision": "Credit Provision",
        "Opex": "Operating Expense",
        "CRE Loan Bal ($MM)": "Commercial Mortgage Balance",
        "NPL Proxy ($MM)": "Nonperforming Loan Proxy",
    }
    frame = frame.melt(id_vars=["period", "scenario"], var_name="raw_metric", value_name="amount_millions")
    frame["source_metric"] = frame["raw_metric"].map(metric_names)
    frame["period"] = pd.to_datetime(frame["period"]).dt.to_period("M").dt.to_timestamp("M")
    frame["business_unit"] = "Commercial Real Estate"
    return frame[["business_unit", "period", "scenario", "source_metric", "amount_millions"]]


def clean_capital_markets(path: Path) -> pd.DataFrame:
    """Transpose month columns from every Capital Markets metric sheet."""
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name)
        frame = frame.melt(id_vars="Scenario", var_name="period", value_name="amount_millions")
        frame = frame.rename(columns={"Scenario": "scenario"})
        frame["period"] = pd.to_datetime(frame["period"], format="%b-%y").dt.to_period("M").dt.to_timestamp("M")
        frame["source_metric"] = sheet_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["business_unit"] = "Capital Markets"
    return combined[["business_unit", "period", "scenario", "source_metric", "amount_millions"]]


def clean_all_sources(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read, validate, and combine every raw source package."""
    frames = [
        clean_commercial_banking(raw_dir / "commercial_banking_monthly.xlsx"),
        clean_commercial_real_estate(raw_dir / "commercial_real_estate_monthly.xlsx"),
        clean_capital_markets(raw_dir / "capital_markets_monthly.xlsx"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["amount_millions"] = pd.to_numeric(combined["amount_millions"], errors="coerce")
    if combined["amount_millions"].isna().any():
        raise ValueError("Cleaning produced missing or non-numeric amounts.")
    return combined.sort_values(["business_unit", "period", "scenario", "source_metric"]).reset_index(drop=True)

