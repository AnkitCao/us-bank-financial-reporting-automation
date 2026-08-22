"""Apply deterministic exception rules before narrative generation."""

from __future__ import annotations

import pandas as pd
import yaml

from src.paths import CONFIG_DIR


def _alert(business_unit: str, period: pd.Timestamp, rule: str, severity: str, metric: str, value: float, message: str) -> dict:
    """Return one standardized alert record."""
    return {
        "business_unit": business_unit,
        "period": period,
        "rule": rule,
        "severity": severity,
        "metric": metric,
        "value": value,
        "message": message,
    }


def detect_exceptions(kpis: pd.DataFrame) -> pd.DataFrame:
    """Evaluate configured red, yellow, and green management exceptions."""
    rules = yaml.safe_load((CONFIG_DIR / "business_rules.yaml").read_text())["exception_rules"]
    alerts: list[dict] = []
    for business_unit, unit_frame in kpis.groupby("business_unit"):
        unit_frame = unit_frame.sort_values("period").reset_index(drop=True)
        for index, row in unit_frame.iterrows():
            period_label = row["period"].strftime("%b %Y")
            if row["revenue_yoy"] < rules["revenue_yoy_decline"]["threshold"]:
                alerts.append(_alert(business_unit, row["period"], "Revenue YoY decline", "Red", "Revenue YoY", row["revenue_yoy"], f"{business_unit} revenue declined {abs(row['revenue_yoy']):.1%} YoY in {period_label}."))
            if row["operating_expense_vs_budget"] > rules["expense_over_budget"]["threshold"]:
                alerts.append(_alert(business_unit, row["period"], "Expense over budget", "Yellow", "Expense vs Budget", row["operating_expense_vs_budget"], f"{business_unit} expense exceeded budget by {row['operating_expense_vs_budget']:.1%} in {period_label}."))
            if row["credit_provision_actual"] > 0 and row["credit_provision_vs_budget"] > rules["credit_provision_over_budget"]["threshold"]:
                alerts.append(_alert(business_unit, row["period"], "Credit provision over budget", "Yellow", "Credit Provision vs. Budget", row["credit_provision_vs_budget"], f"{business_unit} credit provision exceeded budget by {row['credit_provision_vs_budget']:.1%} in {period_label}."))

            trailing = unit_frame.loc[:index, "profit_margin_actual"].tail(12)
            if len(trailing) >= 4 and row["profit_margin_actual"] <= trailing.min():
                alerts.append(_alert(business_unit, row["period"], "Trailing 12M low margin", "Yellow", "Profit Margin", row["profit_margin_actual"], f"{business_unit} reached its lowest trailing-window margin ({row['profit_margin_actual']:.1%}) in {period_label}."))

            if index >= 3:
                prior_three = unit_frame.loc[index - 2:index, "revenue_mom"]
                if (prior_three < 0).all():
                    alerts.append(_alert(business_unit, row["period"], "Three-month revenue decline", "Red", "Revenue MoM", row["revenue_mom"], f"{business_unit} recorded three consecutive monthly revenue declines through {period_label}."))
                previous_declines = unit_frame.loc[index - 3:index - 1, "revenue_mom"]
                if (previous_declines < 0).all() and row["revenue_mom"] > 0:
                    alerts.append(_alert(business_unit, row["period"], "Revenue recovery", "Green", "Revenue MoM", row["revenue_mom"], f"{business_unit} returned to positive monthly growth in {period_label}."))
    columns = ["business_unit", "period", "rule", "severity", "metric", "value", "message"]
    return pd.DataFrame(alerts, columns=columns).sort_values(["period", "severity"], ascending=[False, True]).reset_index(drop=True)
