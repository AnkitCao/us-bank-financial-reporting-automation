"""Create an auditable executive narrative from validated KPI outputs."""

from __future__ import annotations

import pandas as pd


def generate_executive_summary(kpis: pd.DataFrame, alerts: pd.DataFrame, selected_period: pd.Timestamp) -> tuple[str, list[str]]:
    """Generate a deterministic summary without asking an LLM to calculate."""
    current = kpis.loc[kpis["period"] == selected_period].copy()
    strongest = current.sort_values("revenue_vs_budget", ascending=False).iloc[0]
    weakest_margin = current.sort_values("profit_margin_actual").iloc[0]
    summary = (
        f"{strongest['business_unit']} led performance at {strongest['revenue_vs_budget']:+.1%} versus budget. "
        f"{weakest_margin['business_unit']} reported the lowest profit margin at {weakest_margin['profit_margin_actual']:.1%}. "
        "Results reflect synthetic monthly management reporting calibrated to public U.S. Bancorp trends."
    )
    current_alerts = alerts.loc[alerts["period"] == selected_period]
    priority = {"Red": 0, "Yellow": 1, "Green": 2}
    if current_alerts.empty:
        items = ["No configured management exceptions were triggered for the selected period."]
    else:
        current_alerts = current_alerts.assign(priority=current_alerts["severity"].map(priority)).sort_values("priority")
        items = current_alerts["message"].head(4).tolist()
    return summary, items

