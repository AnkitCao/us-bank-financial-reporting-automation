"""Calculate management KPIs from the mapped financial ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while returning NaN when the denominator is zero."""
    return numerator.div(denominator.replace(0, np.nan))


def build_monthly_kpis(mapped: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue, expense, credit, profit, and comparison KPIs."""
    grouped = (
        mapped.groupby(["business_unit", "period", "scenario", "metric_type"], as_index=False)["amount_millions"]
        .sum()
        .pivot_table(index=["business_unit", "period", "scenario"], columns="metric_type", values="amount_millions", fill_value=0)
        .reset_index()
    )
    for column in ["Revenue", "Expense", "Credit"]:
        if column not in grouped:
            grouped[column] = 0.0
    grouped = grouped.rename(columns={"Revenue": "revenue", "Expense": "operating_expense", "Credit": "credit_provision"})
    grouped["pre_provision_profit"] = grouped["revenue"] - grouped["operating_expense"]
    grouped["adjusted_profit"] = grouped["pre_provision_profit"] - grouped["credit_provision"]
    grouped["cost_to_income_ratio"] = _safe_divide(grouped["operating_expense"], grouped["revenue"])
    grouped["profit_margin"] = _safe_divide(grouped["adjusted_profit"], grouped["revenue"])

    source_values = (
        mapped.pivot_table(
            index=["business_unit", "period", "scenario"],
            columns="source_metric",
            values="amount_millions",
            aggfunc="sum",
        )
        .reset_index()
    )
    grouped = grouped.merge(source_values, on=["business_unit", "period", "scenario"], how="left", validate="one_to_one")
    grouped["loan_to_deposit_ratio"] = _safe_divide(grouped.get("Loan Balance"), grouped.get("Deposit Balance"))
    grouped["npl_ratio"] = _safe_divide(grouped.get("Nonperforming Loan Proxy"), grouped.get("Commercial Mortgage Balance"))
    capital_markets_mix = {
        "advisory_mix": "Advisory Fee",
        "underwriting_mix": "Underwriting Revenue",
        "trading_mix": "Trading Revenue",
        "structuring_mix": "Structuring Fee",
        "syndication_mix": "Syndication Fee",
    }
    for output_column, source_metric in capital_markets_mix.items():
        grouped[output_column] = _safe_divide(grouped.get(source_metric), grouped["revenue"])

    value_columns = [
        "revenue", "operating_expense", "credit_provision", "adjusted_profit", "cost_to_income_ratio", "profit_margin",
        "loan_to_deposit_ratio", "npl_ratio", *capital_markets_mix,
    ]
    wide = grouped.pivot(index=["business_unit", "period"], columns="scenario", values=value_columns)
    wide.columns = [f"{metric}_{scenario.lower().replace(' ', '_')}" for metric, scenario in wide.columns]
    wide = wide.reset_index()
    for metric in ["revenue", "operating_expense", "credit_provision", "adjusted_profit"]:
        wide[f"{metric}_vs_budget"] = _safe_divide(wide[f"{metric}_actual"] - wide[f"{metric}_budget"], wide[f"{metric}_budget"])
        wide[f"{metric}_vs_forecast"] = _safe_divide(wide[f"{metric}_actual"] - wide[f"{metric}_forecast"], wide[f"{metric}_forecast"])
        wide[f"{metric}_vs_prior_year"] = _safe_divide(wide[f"{metric}_actual"] - wide[f"{metric}_prior_year"], wide[f"{metric}_prior_year"])

    wide = wide.sort_values(["business_unit", "period"])
    wide["revenue_mom"] = wide.groupby("business_unit")["revenue_actual"].pct_change()
    wide["revenue_yoy"] = wide["revenue_vs_prior_year"]
    wide["t12m_revenue"] = wide.groupby("business_unit")["revenue_actual"].transform(lambda values: values.rolling(12, min_periods=1).sum())
    wide["forecast_accuracy"] = 1 - _safe_divide((wide["revenue_actual"] - wide["revenue_forecast"]).abs(), wide["revenue_forecast"].abs())
    return wide.reset_index(drop=True)


def build_metric_detail(mapped: pd.DataFrame) -> pd.DataFrame:
    """Create a driver-level Actual/Budget/Forecast/Prior Year comparison table."""
    detail = mapped.pivot_table(
        index=["business_unit", "period", "source_metric", "management_category", "metric_type"],
        columns="scenario",
        values="amount_millions",
        aggfunc="sum",
    ).reset_index()
    detail.columns.name = None
    detail = detail.rename(columns={scenario: scenario.lower().replace(" ", "_") for scenario in ["Actual", "Budget", "Forecast", "Prior Year"]})
    detail["variance_to_budget"] = detail["actual"] - detail["budget"]
    detail["variance_to_forecast"] = detail["actual"] - detail["forecast"]
    return detail
