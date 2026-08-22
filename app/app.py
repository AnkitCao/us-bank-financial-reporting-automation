"""Streamlit entry point for the Monthly LOB Financial Review."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_summary import generate_executive_summary
from src.paths import PROCESSED_DIR

COLORS = {
    "navy": "#0B1F3A",
    "blue": "#1F5A94",
    "gold": "#C89B3C",
    "orange": "#D97732",
    "teal": "#3C7C7A",
    "red": "#B74242",
    "gray": "#687386",
    "light": "#F4F7FA",
}
UNIT_COLORS = {
    "Commercial Banking": COLORS["blue"],
    "Commercial Real Estate": COLORS["gold"],
    "Capital Markets": COLORS["teal"],
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load validated pipeline outputs from disk."""
    kpis = pd.read_csv(PROCESSED_DIR / "monthly_kpis.csv", parse_dates=["period"])
    alerts = pd.read_csv(PROCESSED_DIR / "executive_alerts.csv", parse_dates=["period"])
    detail = pd.read_csv(PROCESSED_DIR / "metric_detail.csv", parse_dates=["period"])
    return kpis, alerts, detail


def fmt_money(value: float) -> str:
    """Format a USD millions value for executive display."""
    return f"${value:,.1f}M"


def base_layout(fig: go.Figure, title: str, subtitle: str) -> go.Figure:
    """Apply one consistent visual system to every Plotly chart."""
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01, "xanchor": "left"},
        margin=dict(l=30, r=20, t=75, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color=COLORS["navy"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D9E0E8")
    fig.update_yaxes(gridcolor="#E9EEF3", zerolinecolor="#8C97A5")
    return fig


def trend_chart(kpis: pd.DataFrame, selected_units: list[str]) -> go.Figure:
    """Show actual revenue over the trailing monthly window."""
    fig = go.Figure()
    for unit in selected_units:
        frame = kpis.loc[kpis["business_unit"] == unit]
        fig.add_trace(
            go.Scatter(
                x=frame["period"],
                y=frame["revenue_actual"],
                name=unit,
                mode="lines+markers",
                line=dict(color=UNIT_COLORS[unit], width=3),
                marker=dict(size=6),
                hovertemplate="%{x|%b %Y}<br>$%{y:.1f}M<extra></extra>",
            )
        )
    fig.update_yaxes(title="Revenue ($M)")
    return base_layout(fig, "Revenue Trend", "Actual monthly revenue; synthetic USD millions")


def waterfall_chart(detail: pd.DataFrame, period: pd.Timestamp) -> go.Figure:
    """Bridge total budget to actual using driver-level variances."""
    frame = detail.loc[(detail["period"] == period) & (detail["metric_type"] == "Revenue")].copy()
    frame = frame.groupby("management_category", as_index=False)[["actual", "budget"]].sum()
    frame["variance"] = frame["actual"] - frame["budget"]
    budget = frame["budget"].sum()
    actual = frame["actual"].sum()
    labels = ["Budget"] + frame["management_category"].tolist() + ["Actual"]
    values = [budget] + frame["variance"].tolist() + [actual]
    measures = ["absolute"] + ["relative"] * len(frame) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            text=[f"${value:+.1f}M" if index else f"${value:.1f}M" for index, value in enumerate(values)],
            textposition="outside",
            connector={"line": {"color": "#AAB4C0"}},
            increasing={"marker": {"color": COLORS["blue"]}},
            decreasing={"marker": {"color": "#D7A07E"}},
            totals={"marker": {"color": COLORS["navy"]}},
            hovertemplate="%{x}<br>$%{y:.1f}M<extra></extra>",
        )
    )
    return base_layout(fig, "Revenue Variance Bridge", f"Budget to Actual for {period:%b %Y}; focused management view")


def scorecard_rows(current: pd.DataFrame) -> pd.DataFrame:
    """Create an executive-ready scorecard table."""
    result = current[["business_unit", "revenue_actual", "revenue_vs_budget", "revenue_yoy", "adjusted_profit_actual", "profit_margin_actual", "forecast_accuracy"]].copy()
    result["status"] = result.apply(
        lambda row: "Green" if row["revenue_vs_budget"] >= 0 and row["profit_margin_actual"] >= 0.35 else ("Yellow" if row["revenue_vs_budget"] >= -0.05 else "Red"),
        axis=1,
    )
    return result


def render_scorecard(scorecard: pd.DataFrame) -> None:
    """Render a bounded HTML scorecard with clear status indicators."""
    header = "<tr><th>Business Unit</th><th>Status</th><th>Revenue</th><th>vs Budget</th><th>YoY</th><th>Profit</th><th>Margin</th><th>Forecast Accuracy</th></tr>"
    rows = []
    status_colors = {"Green": "#3C7C7A", "Yellow": "#C89B3C", "Red": "#B74242"}
    for _, row in scorecard.iterrows():
        badge = f"<span class='status' style='background:{status_colors[row['status']]}'>{row['status']}</span>"
        rows.append(
            "<tr>"
            f"<td class='unit'>{row['business_unit']}</td><td>{badge}</td>"
            f"<td>{fmt_money(row['revenue_actual'])}</td><td>{row['revenue_vs_budget']:+.1%}</td>"
            f"<td>{row['revenue_yoy']:+.1%}</td><td>{fmt_money(row['adjusted_profit_actual'])}</td>"
            f"<td>{row['profit_margin_actual']:.1%}</td><td>{row['forecast_accuracy']:.1%}</td></tr>"
        )
    st.markdown(f"<div class='score-wrap'><table class='scorecard'>{header}{''.join(rows)}</table></div>", unsafe_allow_html=True)


st.set_page_config(page_title="Monthly LOB Financial Review", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #F4F7FA; }
      .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #0B1F3A; letter-spacing: -0.02em; }
      [data-testid="stMetric"] { background: white; border: 1px solid #DFE6ED; border-radius: 10px; padding: 16px; box-shadow: 0 2px 10px rgba(11,31,58,.04); }
      .eyebrow { color:#1F5A94; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      .brief { background:white; border-left:5px solid #1F5A94; padding:18px 20px; border-radius:8px; box-shadow:0 2px 10px rgba(11,31,58,.04); }
      .alert { background:white; border:1px solid #DFE6ED; border-radius:8px; padding:12px 14px; margin-bottom:8px; }
      .alert-red { border-left:5px solid #B74242; } .alert-yellow { border-left:5px solid #C89B3C; } .alert-green { border-left:5px solid #3C7C7A; }
      .score-wrap { overflow-x:auto; background:white; border-radius:10px; border:1px solid #DFE6ED; }
      .scorecard { width:100%; border-collapse:collapse; font-size:.92rem; }
      .scorecard th { background:#0B1F3A; color:white; padding:12px; text-align:right; }
      .scorecard th:first-child, .scorecard td:first-child { text-align:left; }
      .scorecard td { padding:12px; border-bottom:1px solid #E8EDF2; text-align:right; color:#26384D; }
      .scorecard .unit { font-weight:700; color:#0B1F3A; }
      .status { color:white; font-size:.72rem; font-weight:700; padding:4px 8px; border-radius:12px; }
      .disclaimer { color:#687386; font-size:.78rem; padding-top:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    kpis, alerts, detail = load_data()
except FileNotFoundError:
    st.error("Processed data is missing. Run `python -m src.pipeline` from the project root.")
    st.stop()

period_options = sorted(kpis["period"].unique(), reverse=True)
unit_options = sorted(kpis["business_unit"].unique())
with st.sidebar:
    st.markdown("### Review Controls")
    selected_period = pd.Timestamp(st.selectbox("Reporting month", period_options, format_func=lambda value: pd.Timestamp(value).strftime("%B %Y")))
    selected_units = st.multiselect("Business units", unit_options, default=unit_options)
    st.markdown("---")
    st.caption("Latest public calibration: U.S. Bancorp Q2 2026. All monthly values are synthetic.")

if not selected_units:
    st.warning("Select at least one business unit.")
    st.stop()

current = kpis.loc[(kpis["period"] == selected_period) & (kpis["business_unit"].isin(selected_units))]
current_alerts = alerts.loc[(alerts["period"] == selected_period) & (alerts["business_unit"].isin(selected_units))]

st.markdown("<div class='eyebrow'>Executive Decision Support</div>", unsafe_allow_html=True)
st.title("Monthly LOB Financial Review")
st.caption(f"Reporting period: {selected_period:%B %Y} · Synthetic USD millions · Actual vs Budget, Forecast and Prior Year")

summary, attention_items = generate_executive_summary(kpis[kpis["business_unit"].isin(selected_units)], alerts[alerts["business_unit"].isin(selected_units)], selected_period)
st.markdown(f"<div class='brief'><strong>Executive Summary</strong><br>{summary}</div>", unsafe_allow_html=True)

total_revenue = current["revenue_actual"].sum()
total_budget = current["revenue_budget"].sum()
total_profit = current["adjusted_profit_actual"].sum()
total_expense = current["operating_expense_actual"].sum()
total_forecast = current["revenue_forecast"].sum()
cards = st.columns(5)
cards[0].metric("Revenue", fmt_money(total_revenue), f"{(total_revenue / total_budget - 1):+.1%} vs budget")
cards[1].metric("Adjusted Profit", fmt_money(total_profit), f"{(total_profit / current['adjusted_profit_budget'].sum() - 1):+.1%} vs budget")
cards[2].metric("Operating Expense", fmt_money(total_expense), f"{(total_expense / current['operating_expense_budget'].sum() - 1):+.1%} vs budget", delta_color="inverse")
cards[3].metric("Budget Attainment", f"{total_revenue / total_budget:.1%}")
cards[4].metric("Forecast Accuracy", f"{1 - abs(total_revenue - total_forecast) / total_forecast:.1%}")

st.subheader("Items Requiring Attention")
if current_alerts.empty:
    st.success("No configured exceptions were triggered for the selected period.")
else:
    severity_rank = {"Red": 0, "Yellow": 1, "Green": 2}
    for _, alert in current_alerts.assign(rank=current_alerts["severity"].map(severity_rank)).sort_values("rank").iterrows():
        st.markdown(f"<div class='alert alert-{alert['severity'].lower()}'><strong>{alert['severity']} · {alert['business_unit']}</strong><br>{alert['message']}</div>", unsafe_allow_html=True)

st.subheader("Business Unit Scorecard")
render_scorecard(scorecard_rows(current))

left, right = st.columns([1.1, 0.9])
with left:
    trend_data = kpis.loc[kpis["business_unit"].isin(selected_units)]
    st.plotly_chart(trend_chart(trend_data, selected_units), use_container_width=True, config={"displayModeBar": False})
with right:
    selected_detail = detail.loc[detail["business_unit"].isin(selected_units)]
    st.plotly_chart(waterfall_chart(selected_detail, selected_period), use_container_width=True, config={"displayModeBar": False})

with st.expander("Driver detail and audit trail"):
    driver_table = detail.loc[(detail["period"] == selected_period) & (detail["business_unit"].isin(selected_units))].copy()
    driver_table = driver_table[["business_unit", "source_metric", "management_category", "metric_type", "actual", "budget", "forecast", "prior_year", "variance_to_budget"]]
    st.dataframe(
        driver_table.style.format({column: "${:,.2f}M" for column in ["actual", "budget", "forecast", "prior_year", "variance_to_budget"]}),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("<div class='disclaimer'>Disclaimer: This portfolio project uses simplified synthetic figures calibrated to public U.S. Bancorp disclosures. It does not represent U.S. Bank internal reporting, accounting definitions, forecasts or organizational structure.</div>", unsafe_allow_html=True)

