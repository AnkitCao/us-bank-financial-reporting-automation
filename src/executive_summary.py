"""Create an auditable executive narrative from validated KPI outputs."""

from __future__ import annotations

import pandas as pd


def generate_executive_summary(kpis: pd.DataFrame, alerts: pd.DataFrame, period_label: str) -> tuple[str, list[str]]:
    """Generate a deterministic summary for the active reporting window."""
    try:
        display_period = pd.Timestamp(period_label).strftime("%B %Y")
    except (TypeError, ValueError):
        display_period = str(period_label)
    current = kpis.groupby("business_unit", as_index=False).agg(
        revenue_actual=("revenue_actual", "sum"),
        revenue_budget=("revenue_budget", "sum"),
        adjusted_profit_actual=("adjusted_profit_actual", "sum"),
    )
    current["revenue_vs_budget"] = current["revenue_actual"] / current["revenue_budget"] - 1
    current["profit_margin_actual"] = current["adjusted_profit_actual"] / current["revenue_actual"]
    strongest = current.sort_values("revenue_vs_budget", ascending=False).iloc[0]
    weakest_margin = current.sort_values("profit_margin_actual").iloc[0]
    summary = (
        f"For {display_period}, {strongest['business_unit']} led revenue performance ({strongest['revenue_vs_budget']:+.1%}) versus target. "
        f"{weakest_margin['business_unit']} reported the lowest profit margin ({weakest_margin['profit_margin_actual']:.1%})."
    )
    priority = {"Red": 0, "Yellow": 1, "Green": 2}
    if alerts.empty:
        items = ["No configured management exceptions were triggered for the selected period."]
    else:
        current_alerts = alerts.assign(priority=alerts["severity"].map(priority)).sort_values(["priority", "period"], ascending=[True, False])
        items = current_alerts["message"].head(4).tolist()
    return summary, items
