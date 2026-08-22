"""Streamlit entry point for the Monthly LOB Financial Review."""

from __future__ import annotations

import sys
import textwrap
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_summary import generate_executive_summary
from src.paths import PROCESSED_DIR

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "us-bank-logo.svg"

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
PERFORMANCE_COLORS = {"negative": "#FF3B30", "watch": "#F4B400", "positive": "#2563EB"}
UNIT_COLORS = {
    "Commercial Banking": PERFORMANCE_COLORS["watch"],
    "Commercial Real Estate": PERFORMANCE_COLORS["negative"],
    "Capital Markets": PERFORMANCE_COLORS["positive"],
}
MIX_COLORS = ["#0B2E6F", "#2563EB", "#0F9D8A", "#F59E0B", "#DC5A5A"]
BUSINESS_UNIT_ORDER = ["Commercial Banking", "Capital Markets", "Commercial Real Estate"]


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


def local_chart_summary(chart_key: str, values: dict) -> str:
    """Return a concise deterministic summary when an API key is unavailable."""
    if chart_key == "cost_to_income":
        return f"{values['highest_unit']} recorded the highest {values['period']} ratio at {values['highest']:.1f}%, while {values['lowest_unit']} remained lowest at {values['lowest']:.1f}%."
    if chart_key == "profit_margin":
        return f"{values['highest_unit']} led {values['period']} profit margin at {values['highest']:.1f}%, {values['variance']:+.1f} percentage points versus target."
    if chart_key == "revenue_trend":
        return f"{values['unit']} recorded {values['latest']:.1f}M in {values['period']}, a {values['change']:+.1f}% change from the prior month."
    if chart_key == "revenue_variance":
        return f"Revenue finished ${abs(values['variance']):.1f}M {'above' if values['variance'] >= 0 else 'below'} target, led by {values['driver']}."
    if chart_key == "expense_variance":
        return f"{values['unit']} had the largest unfavorable expense variance at ${values['variance']:.1f}M above budget."
    if chart_key == "npl_ratio":
        direction = "increased" if values["latest"] > values["prior"] else "decreased"
        return f"CRE NPL ratio {direction} to {values['latest']:.2f}% and remained below the illustrative 1.50% review threshold."
    if chart_key == "loan_to_deposit":
        direction = "increased" if values["latest"] > values["prior"] else "decreased"
        return f"The ratio {direction} to {values['latest']:.1f}% in {values['period']}, indicating loan growth continued to outpace deposit funding."
    return f"{values['largest_component']} was the largest {values['period']} Capital Markets fee component at {values['largest_share']:.1f}%."


@st.cache_data(ttl=3600, show_spinner=False)
def generate_chart_summary(chart_key: str, summary_dict: dict) -> str:
    """Summarize precomputed KPI values without exposing source-level data."""
    fallback = local_chart_summary(chart_key, summary_dict)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return fallback
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-5-mini"),
            instructions="Write one concise, factual sentence for a CFO briefing. Do not add facts or recommendations.",
            input=f"Chart: {chart_key}. Precomputed KPI summary: {summary_dict}",
        )
        return response.output_text.strip() or fallback
    except Exception:
        return fallback


def base_layout(fig: go.Figure, title: str, subtitle: str) -> go.Figure:
    """Apply one consistent visual system to every Plotly chart."""
    fig.update_layout(
        title={"text": f"<b>{title}</b><br><span style='font-size:20px;color:#64748B'>{subtitle}</span>", "x": 0.025, "xanchor": "left", "y": 0.94, "yanchor": "top", "font": {"family": "Times New Roman, Times, serif", "size": 32, "color": "#0B2E6F"}},
        margin=dict(l=92, r=48, t=174, b=78),
        height=550,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Times New Roman, Times, serif", size=19, color="#334155"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=0.98,
            font=dict(size=18, color="#334155"), bgcolor="rgba(255,255,255,.96)",
        ),
        hovermode="closest",
        hoverdistance=100,
        hoverlabel=dict(
            bgcolor="white", bordercolor="#0B2E6F",
            font=dict(family="Times New Roman, Times, serif", size=25, color="#0B2E6F"),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1", tickfont=dict(size=18, color="#64748B"), title_font=dict(size=20), automargin=True)
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#94A3B8", tickfont=dict(size=18, color="#64748B"), title_font=dict(size=20), automargin=True)
    fig.update_traces(textfont=dict(family="Times New Roman, Times, serif", size=21, color="#0B2E6F"))
    return fig


def add_chart_summary(fig: go.Figure, summary: str) -> go.Figure:
    """Add a visible conclusion as a dedicated third title line."""
    current_title = fig.layout.title.text or ""
    summary_lines = textwrap.wrap(summary, width=62, break_long_words=False, break_on_hyphens=False)
    summary_html = "<br>".join(escape(line) for line in summary_lines[:2])
    fig.update_layout(
        title_text=(
            f"{current_title}<br>"
            f"<span style='font-size:21px;color:#334155'><b>★ Summary:</b> {summary_html}</span>"
        ),
        margin=dict(t=226 if len(summary_lines) > 1 else 198),
    )
    return fig


def ordered_units(selected_units: list[str]) -> list[str]:
    """Return one stable order across every dashboard visual."""
    return [unit for unit in BUSINESS_UNIT_ORDER if unit in selected_units]


def trend_chart(kpis: pd.DataFrame, selected_units: list[str]) -> go.Figure:
    """Show actual revenue over the trailing monthly window."""
    fig = go.Figure()
    for unit in ordered_units(selected_units):
        frame = kpis.loc[kpis["business_unit"] == unit].sort_values("period")
        fig.add_trace(
            go.Scatter(
                x=frame["period"].dt.strftime("%b %Y"),
                y=frame["revenue_actual"],
                name=unit,
                mode="lines+markers",
                line=dict(color=UNIT_COLORS[unit], width=4),
                marker=dict(size=11),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Y: $%{y:.1f}M<extra></extra>",
            )
        )
    fig.update_yaxes(title="Revenue ($M)")
    fig.update_layout(showlegend=True)
    return base_layout(fig, "Revenue Trend", "Monthly revenue trend, synthetic data in USD millions")


def cost_income_trend_chart(kpis: pd.DataFrame, selected_units: list[str]) -> go.Figure:
    """Compare operating efficiency across business units."""
    fig = go.Figure()
    for unit in ordered_units(selected_units):
        frame = kpis.loc[kpis["business_unit"] == unit].sort_values("period")
        fig.add_trace(go.Scatter(
            x=frame["period"].dt.strftime("%b %Y"), y=frame["cost_to_income_ratio_actual"], name=unit,
            mode="lines+markers", line=dict(color=UNIT_COLORS[unit], width=4), marker=dict(size=11),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Y: %{y:.1%}<extra></extra>",
        ))
    fig.update_yaxes(title="Cost / Income", tickformat=".0%", rangemode="tozero")
    fig.update_layout(showlegend=True)
    return base_layout(fig, "Cost to Income Ratio", "Lower ratios indicate greater operating efficiency")


def profit_margin_comparison_chart(current_month: pd.DataFrame, period_label: str) -> go.Figure:
    """Compare latest actual and target profit margins by business unit."""
    frame = current_month.copy()
    frame["business_unit"] = pd.Categorical(frame["business_unit"], BUSINESS_UNIT_ORDER, ordered=True)
    frame = frame.sort_values("business_unit")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual", x=frame["business_unit"], y=frame["profit_margin_actual"], width=0.30,
        marker_color="#2563EB",
        hovertemplate="<b>%{x}</b><br>Actual Margin: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Target", x=frame["business_unit"], y=frame["profit_margin_budget"], width=0.30,
        marker_color="#F4B400",
        hovertemplate="<b>%{x}</b><br>Target Margin: %{y:.1%}<extra></extra>",
    ))
    upper = max(frame["profit_margin_actual"].max(), frame["profit_margin_budget"].max()) * 1.22
    fig.update_yaxes(title="Profit Margin", tickformat=".0%", range=[0, upper])
    fig.update_layout(barmode="group", bargap=0.42, bargroupgap=0.16)
    return base_layout(fig, "Profit Margin by Business Unit", f"Actual versus target – {period_label}")


def single_ratio_trend_chart(frame: pd.DataFrame, column: str, title: str, subtitle: str, threshold: float | None = None) -> go.Figure:
    """Render a business-specific ratio trend."""
    frame = frame.sort_values("period")
    fig = go.Figure(go.Scatter(
        x=frame["period"].dt.strftime("%b %Y"), y=frame[column], name=title, mode="lines+markers",
        line=dict(color="#2563EB", width=4), marker=dict(size=11),
        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Y: %{y:.2%}<extra></extra>",
    ))
    if threshold is not None:
        fig.add_hline(
            y=threshold, line_color="#FF3B30", line_dash="dash", line_width=2,
            annotation_text="Illustration threshold: 1.5%", annotation_position="top",
            annotation_font=dict(size=18, color="#B42318"),
        )
        fig.update_yaxes(title="Ratio", tickformat=".1%", range=[0, max(threshold * 1.18, frame[column].max() * 1.18)])
    elif column == "loan_to_deposit_ratio_actual":
        fig.update_yaxes(title="Ratio", tickformat=".1%", range=[1.00, 1.25])
    else:
        low, high = frame[column].min(), frame[column].max()
        padding = max((high - low) * 0.65, high * 0.025)
        fig.update_yaxes(title="Ratio", tickformat=".1%", range=[max(0, low - padding), high + padding])
    fig.update_layout(showlegend=False)
    return base_layout(fig, title, subtitle)


def fee_revenue_mix_chart(frame: pd.DataFrame) -> go.Figure:
    """Show the last six months of Capital Markets fee composition."""
    components = [
        ("Advisory", "advisory_mix_actual"), ("Underwriting", "underwriting_mix_actual"),
        ("Trading", "trading_mix_actual"), ("Structuring", "structuring_mix_actual"),
        ("Syndication", "syndication_mix_actual"),
    ]
    frame = frame.sort_values("period").tail(6)
    fig = go.Figure()
    for (label, column), color in zip(components, MIX_COLORS):
        fig.add_trace(go.Bar(
            name=label, x=frame["period"].dt.strftime("%b %Y"), y=frame[column], marker_color=color,
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Y: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_yaxes(title="Revenue Mix", tickformat=".0%", range=[0, 1])
    fig.update_layout(barmode="stack", bargap=0.35)
    fig = base_layout(fig, "Capital Markets Fee Revenue Mix", "Share of monthly fee revenue, latest six months")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5, font=dict(size=17)),
        margin=dict(l=92, r=48, t=174, b=108),
    )
    return fig


def waterfall_chart(detail: pd.DataFrame, period_label: str) -> go.Figure:
    """Bridge total budget to actual using driver-level variances."""
    frame = detail.loc[detail["metric_type"] == "Revenue"].copy()
    frame = frame.groupby("management_category", as_index=False)[["actual", "budget"]].sum()
    frame["variance"] = frame["actual"] - frame["budget"]
    budget = frame["budget"].sum()
    actual = frame["actual"].sum()
    short_labels = frame["management_category"].replace({"Fee / Noninterest Income": "Fee Income"}).tolist()
    labels = ["Target"] + short_labels + ["Actual"]
    values = [budget] + frame["variance"].tolist() + [actual]
    measures = ["absolute"] + ["relative"] * len(frame) + ["total"]
    display_text = [f"${budget:.1f}M"] + ["" if abs(value) < 0.05 else f"${value:+.1f}M" for value in frame["variance"]] + [f"${actual:.1f}M"]
    bar_widths = [0.52] + [0.42] * len(frame) + [0.52]
    fig = go.Figure(
        go.Waterfall(
            name="Target to Actual",
            showlegend=False,
            width=bar_widths,
            x=labels,
            y=values,
            measure=measures,
            text=display_text,
            textposition="outside",
            connector={"line": {"color": "#7C8A9A", "width": 1.5, "dash": "dot"}},
            increasing={"marker": {"color": "#2563EB"}},
            decreasing={"marker": {"color": "#FF3B30"}},
            totals={"marker": {"color": "#001E79"}},
            hovertemplate="<b>%{x}</b><br>Y: $%{y:.1f}M<extra></extra>",
        )
    )
    fig.update_traces(textfont=dict(size=22, color="#001E79"))
    fig.update_xaxes(tickangle=0, tickfont=dict(size=18))
    fig.update_yaxes(title="Revenue ($M)", rangemode="tozero", gridcolor="#EEF1F7")
    fig.update_yaxes(range=[0, max(budget, actual) * 1.22])
    fig.update_layout(showlegend=False, waterfallgap=0.20)
    return base_layout(fig, "Revenue Variance Bridge", f"Target to Actual – {period_label}")


def performance_colors(values: pd.Series) -> list[str]:
    """Rank the displayed values from red (lowest) to blue (highest)."""
    if len(values) == 1:
        return [PERFORMANCE_COLORS["positive"]]
    ranks = values.rank(method="first", pct=True)
    return [
        PERFORMANCE_COLORS["negative"] if rank <= 1 / 3 else PERFORMANCE_COLORS["watch"] if rank <= 2 / 3 else PERFORMANCE_COLORS["positive"]
        for rank in ranks
    ]


def revenue_bar_chart(current: pd.DataFrame, period_label: str) -> go.Figure:
    """Compare actual revenue across selected business units."""
    frame = current.sort_values("revenue_actual", ascending=False)
    fig = go.Figure()
    colors = performance_colors(frame["revenue_vs_budget"])
    for (_, row), color in zip(frame.iterrows(), colors):
        fig.add_trace(
            go.Bar(
                name=row["business_unit"], x=[row["business_unit"]], y=[row["revenue_actual"]], width=0.42,
                marker_color=color, text=[f"${row['revenue_actual']:.1f}M"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:.1f}M<extra></extra>", showlegend=True,
            )
        )
    fig.update_yaxes(title="Revenue ($M)")
    fig.update_layout(bargap=0.48)
    return base_layout(fig, "Revenue by Business Unit", f"Actual revenue – {period_label}")


def profit_bar_chart(current: pd.DataFrame, period_label: str) -> go.Figure:
    """Compare adjusted profit across selected business units."""
    frame = current.sort_values("adjusted_profit_actual", ascending=False)
    fig = go.Figure()
    colors = performance_colors(frame["adjusted_profit_vs_budget"])
    for (_, row), color in zip(frame.iterrows(), colors):
        fig.add_trace(
            go.Bar(
                name=row["business_unit"], x=[row["business_unit"]], y=[row["adjusted_profit_actual"]], width=0.42,
                marker_color=color, text=[f"${row['adjusted_profit_actual']:.1f}M"], textposition="outside",
                hovertemplate="<b>%{x}</b><br>Adjusted Profit: $%{y:.1f}M<extra></extra>", showlegend=True,
            )
        )
    fig.update_yaxes(title="Adjusted Profit ($M)")
    fig.update_layout(bargap=0.48)
    return base_layout(fig, "Adjusted Profit by Business Unit", f"Actual adjusted profit – {period_label}")


def expense_comparison_chart(current: pd.DataFrame, period_label: str) -> go.Figure:
    """Compare actual operating expense with budget by business unit."""
    frame = current.copy()
    frame["business_unit"] = pd.Categorical(frame["business_unit"], BUSINESS_UNIT_ORDER, ordered=True)
    frame = frame.sort_values("business_unit")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Actual",
            x=frame["business_unit"],
            y=frame["operating_expense_actual"],
            width=0.32,
            marker_color="#2563EB",
            hovertemplate="<b>%{x}</b><br>Actual Expense: $%{y:.1f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Budget",
            x=frame["business_unit"],
            y=frame["operating_expense_budget"],
            width=0.32,
            marker_color="#F4B400",
            hovertemplate="<b>%{x}</b><br>Budget Expense: $%{y:.1f}M<extra></extra>",
        )
    )
    upper = max(frame["operating_expense_actual"].max(), frame["operating_expense_budget"].max()) * 1.24
    fig.update_yaxes(title="Operating Expense ($M)", range=[0, upper])
    fig.update_layout(barmode="group", bargap=0.42, bargroupgap=0.16)
    return base_layout(fig, "Operating Expense vs. Budget", f"Actual versus budget – {period_label}")


def aggregate_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate additive measures and recompute ratios for the selected time window."""
    measure_prefixes = ("revenue_", "operating_expense_", "credit_provision_", "adjusted_profit_")
    additive_columns = [
        column for column in frame.columns
        if column.startswith(measure_prefixes)
        and not column.startswith(("revenue_vs_", "operating_expense_vs_", "credit_provision_vs_", "adjusted_profit_vs_"))
        and not column.startswith("revenue_mom")
    ]
    additive_columns = [column for column in additive_columns if not column.startswith("profit_margin_")]
    result = frame.groupby("business_unit", as_index=False)[additive_columns].sum()
    result["revenue_vs_budget"] = result["revenue_actual"] / result["revenue_budget"] - 1
    result["revenue_yoy"] = result["revenue_actual"] / result["revenue_prior_year"] - 1
    result["adjusted_profit_vs_budget"] = result["adjusted_profit_actual"] / result["adjusted_profit_budget"] - 1
    result["operating_expense_vs_budget"] = result["operating_expense_actual"] / result["operating_expense_budget"] - 1
    result["cost_to_income_ratio_actual"] = result["operating_expense_actual"] / result["revenue_actual"]
    result["profit_margin_actual"] = result["adjusted_profit_actual"] / result["revenue_actual"]
    result["forecast_accuracy"] = 1 - (result["revenue_actual"] - result["revenue_forecast"]).abs() / result["revenue_forecast"]
    ending_values = frame.sort_values("period").groupby("business_unit", as_index=False).tail(1).set_index("business_unit")
    for ratio in ["loan_to_deposit_ratio_actual", "npl_ratio_actual"]:
        if ratio in ending_values:
            result[ratio] = result["business_unit"].map(ending_values[ratio])
    for mix in ["advisory_mix_actual", "underwriting_mix_actual", "trading_mix_actual", "structuring_mix_actual", "syndication_mix_actual"]:
        if mix in frame:
            weighted = (frame[mix] * frame["revenue_actual"]).groupby(frame["business_unit"]).sum(min_count=1)
            revenue = frame.groupby("business_unit")["revenue_actual"].sum()
            result[mix] = result["business_unit"].map(weighted / revenue)
    return result


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
    header = "<tr><th>Business Unit</th><th>Status</th><th>Revenue</th><th>Revenue vs Target</th><th>YoY</th><th>Profit</th><th>Margin</th><th>Actual vs Forecast</th></tr>"
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


def render_business_model_metrics(current: pd.DataFrame) -> None:
    """Show common efficiency ratios and one business-specific metric set."""
    rows = []
    for _, row in current.sort_values("business_unit").iterrows():
        unit = row["business_unit"]
        if unit == "Commercial Banking":
            specialized = f"Loan-to-Deposit Ratio: {row['loan_to_deposit_ratio_actual']:.1%}"
        elif unit == "Commercial Real Estate":
            specialized = f"NPL Ratio: {row['npl_ratio_actual']:.2%}"
        else:
            mix_items = [
                ("Advisory", row["advisory_mix_actual"]),
                ("Underwriting", row["underwriting_mix_actual"]),
                ("Trading", row["trading_mix_actual"]),
                ("Structuring", row["structuring_mix_actual"]),
                ("Syndication", row["syndication_mix_actual"]),
            ]
            specialized = "Revenue Mix: " + ", ".join(f"{label} {value:.1%}" for label, value in mix_items)
        rows.append(
            "<tr>"
            f"<td>{escape(unit)}</td><td>{row['cost_to_income_ratio_actual']:.1%}</td>"
            f"<td>{row['profit_margin_actual']:.1%}</td><td>{escape(specialized)}</td></tr>"
        )
    header = "<tr><th>Business Unit</th><th>Cost-to-Income Ratio</th><th>Profit Margin</th><th>Other Metrics</th></tr>"
    st.markdown(f"<div class='ratio-wrap'><table class='ratio-table'>{header}{''.join(rows)}</table></div>", unsafe_allow_html=True)


st.set_page_config(page_title="Automating Three Businesses LOB Financial Review", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      html, body, .stApp, .stApp * { font-family:"Times New Roman", Times, serif !important; box-sizing:border-box; }
      html, body, .stApp { background:#F5F7FA; max-width:100%; overflow-x:hidden; }
      .block-container { width:calc(100% - 4rem); max-width:100%; padding-top:2.5rem; padding-bottom:4rem; overflow-x:hidden; }
      [data-testid="stHorizontalBlock"] { width:100%; max-width:100%; gap:16px; }
      [data-testid="stColumn"] { min-width:0; }
      h1, h2, h3 { color:#0B2E6F; letter-spacing:0; font-family:"Times New Roman", Times, serif !important; }
      h1 { font-size:3rem !important; font-weight:800 !important; }
      h2 { font-size:2.35rem !important; font-weight:800 !important; margin-top:1.7rem !important; }
      h3 { font-size:2.35rem !important; font-weight:800 !important; margin-top:2.2rem !important; margin-bottom:1rem !important; line-height:1.2 !important; }
      p, .stCaption { color:#334155; font-size:1.3rem !important; line-height:1.5 !important; }
      .kpi-spacer { height:1.2rem; }
      [data-testid="stMetric"] { display:grid; grid-template-rows:auto auto auto; align-content:start; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:20px; box-shadow:0 2px 8px rgba(15,23,42,.04); min-height:172px; }
      [data-testid="stMetricLabel"] { margin:0 0 .45rem !important; }
      [data-testid="stMetricLabel"] p { color:#334155 !important; font-size:1.8rem !important; font-weight:700 !important; line-height:1.15 !important; margin:0 !important; }
      [data-testid="stMetricValue"] { color:#0B2E6F !important; font-size:2.45rem !important; font-weight:800 !important; line-height:1.05 !important; margin:0 0 .5rem !important; }
      [data-testid="stMetricDelta"] { color:#2563EB !important; font-size:1.25rem !important; font-weight:800 !important; line-height:1.15 !important; margin:0 !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"] { grid-template-rows:auto 1fr; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetricValue"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) [data-testid="stMetricValue"] { align-self:center; margin:0 !important; }
      [data-testid="stMetricDelta"] svg { fill:#2563EB !important; color:#2563EB !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricDelta"] { color:#FF3B30 !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricDelta"] svg { fill:#FF3B30 !important; color:#FF3B30 !important; }
      .brief { background:#FFFFFF; border:1px solid #E2E8F0; border-left:7px solid #0B2E6F; padding:24px 28px; border-radius:10px; box-shadow:0 3px 12px rgba(15,23,42,.05); color:#334155; }
      .creator-credit { text-align:right; color:#334155; font-size:1.35rem; font-weight:700; margin:0 0 .45rem; }
      .creator-credit a { color:#0B2E6F; font-weight:800; text-decoration:underline; text-underline-offset:3px; }
      .creator-credit a:hover { color:#2563EB; }
      .brief strong { display:block; color:#0B2E6F; font-size:2.05rem; font-weight:800; line-height:1.15; margin-bottom:.7rem; }
      .summary-text { display:block; font-size:1.75rem; font-weight:700; line-height:1.45; }
      .attention-heading { display:flex; align-items:center; flex-wrap:wrap; gap:28px; margin:1.7rem 0 .8rem; }
      .attention-heading h2 { margin:0 !important; }
      .alert-legend { display:flex; align-items:center; gap:20px; color:#465269; font-size:1.25rem; font-weight:700; }
      .legend-item { display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }
      .legend-dot { width:13px; height:13px; border-radius:50%; display:inline-block; }
      .dot-red { background:#FF3B30; } .dot-yellow { background:#F4B400; } .dot-green { background:#2563EB; }
      .alert-group { display:grid; grid-template-columns:300px 1fr; min-height:92px; background:white; border:1px solid #DCE2F3; border-radius:11px; margin-bottom:12px; overflow:hidden; }
      .alert-business { display:flex; align-items:center; justify-content:center; min-height:92px; padding:16px 20px; color:#001E79; background:#F1F4FA; text-align:center; font-size:1.5rem; font-weight:800; line-height:1.25; }
      .alert-list { display:flex; flex-direction:column; min-width:0; min-height:92px; }
      .alert { display:flex; align-items:center; flex:1; border:0; border-bottom:1px solid #DCE2F3; border-radius:0; padding:16px 22px; margin:0; font-size:1.4rem; line-height:1.35; }
      .alert:last-child { border-bottom:0; }
      .alert-red { border-left:7px solid #FF3B30; } .alert-yellow { border-left:7px solid #F4B400; } .alert-green { border-left:7px solid #2563EB; }
      .alert-rule { color:#001E79; font-weight:800; }
      .score-wrap { overflow-x:auto; background:white; border-radius:10px; border:1px solid #DCE2F3; }
      .scorecard { width:100%; min-width:1200px; border-collapse:collapse; margin:0 !important; font-size:1.45rem; }
      .scorecard th { background:#001E79; color:white; border:2px solid white; padding:16px; text-align:center; font-size:1.5rem; }
      .scorecard th:first-child, .scorecard td:first-child { text-align:left; }
      .scorecard td { padding:16px; border-bottom:1px solid #DCE2F3; text-align:center; color:#26384D; }
      .scorecard tr:nth-child(even) { background:#F5F7FC; }
      .scorecard .unit { font-weight:800; color:#001E79; }
      .status { color:white; font-size:1.15rem; font-weight:800; padding:5px 10px; border-radius:14px; }
      .ratio-wrap { overflow-x:auto; background:white; border-radius:10px; border:1px solid #DCE2F3; margin-bottom:1.4rem; }
      .ratio-table { width:100%; min-width:1200px; border-collapse:collapse; margin:0 !important; font-size:1.4rem; }
      .ratio-table th { background:#001E79; color:white; border:2px solid white; padding:16px; text-align:center; font-size:1.5rem; }
      .ratio-table td { padding:16px; border-bottom:1px solid #DCE2F3; text-align:center; color:#26384D; }
      .ratio-table td:first-child { text-align:left; color:#001E79; font-weight:800; }
      .ratio-table td:last-child { text-align:left; }
      .ratio-table tr:nth-child(even) { background:#F5F7FC; }
      .disclaimer { color:#687386; font-size:1.05rem; padding-top:18px; }
      [data-testid="stSidebar"] { min-width:500px; max-width:500px; }
      [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a span { font-size:2rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebarNav"] li { margin-bottom:.7rem !important; }
      [data-testid="stSidebarNav"] a { min-height:3.8rem !important; padding:.55rem .8rem !important; border-radius:10px !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { min-height:3.8rem !important; height:auto !important; padding:.65rem .8rem !important; align-items:flex-start !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p { font-size:2rem !important; font-weight:800 !important; line-height:1.15 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere !important; }
      [data-testid="stSidebar"] h3 { font-size:2rem !important; font-weight:800 !important; margin-bottom:1.15rem !important; }
      [data-testid="stSidebar"] label p { font-size:1.8rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { font-size:1.65rem !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div { min-height:4rem !important; align-items:center !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] input { line-height:2rem !important; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size:1.8rem !important; }
      [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button,
      [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button p { font-size:1.5rem !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] span { font-size:1.35rem !important; white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; color:#001E79 !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] { display:flex !important; width:calc(100% - 3.2rem) !important; max-width:none !important; min-height:2.75rem !important; height:auto !important; justify-content:space-between !important; margin:.18rem 0 !important; background:#E9EEF8 !important; border:1px solid #C9D4E8 !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] svg { color:#001E79 !important; fill:#001E79 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div:has([data-baseweb="tag"]) { min-height:10.5rem !important; align-content:flex-start !important; }
      [data-testid="stSidebar"] .stCaption { font-size:1.15rem !important; }
      [data-testid="stPlotlyChart"] { width:100%; max-width:100%; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0; margin-bottom:16px; box-shadow:0 2px 8px rgba(15,23,42,.035); overflow:visible !important; }
      [data-testid="stPlotlyChart"] > div { width:100% !important; max-width:100% !important; overflow:visible !important; }
      [data-testid="stPlotlyChart"] .scatterlayer .point,
      [data-testid="stPlotlyChart"] .barlayer .point,
      [data-testid="stPlotlyChart"] .waterfalllayer .point,
      [data-testid="stPlotlyChart"] .hoverlayer { cursor:pointer !important; }
      [data-testid="stExpander"] { background:#FFFFFF; border:0 !important; border-top:1px solid #E2E8F0 !important; border-radius:0 !important; overflow:visible !important; margin-top:1.35rem !important; margin-bottom:1.35rem !important; }
      [data-testid="stExpander"] details, [data-testid="stExpander"] summary { overflow:visible !important; }
      [data-testid="stExpander"] summary { padding-top:.75rem !important; padding-bottom:.75rem !important; }
      [data-testid="stExpander"] summary p { font-size:2.35rem !important; font-weight:800 !important; color:#0B2E6F !important; line-height:1.2 !important; }
      @media (max-width:1100px) { .alert-group { grid-template-columns:220px 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    kpis, alerts, detail = load_data()
except FileNotFoundError:
    st.error("Processed data is missing. Run `python -m src.pipeline` from the project root.")
    st.stop()

period_options = sorted(pd.Timestamp(value) for value in kpis["period"].unique())
unit_options = sorted(kpis["business_unit"].unique())
with st.sidebar:
    st.page_link("App.py", label="Dashboard Overview")
    st.page_link("pages/1_Source_Data.py", label="Source Data for Three Businesses")
    st.markdown("---")
    st.markdown("### Review Controls")
    time_mode = st.segmented_control("Time view", ["Month", "Quarter", "Year", "Custom Range"], default="Month")
    if time_mode == "Month":
        selected_end = pd.Timestamp(st.selectbox("Reporting month", list(reversed(period_options)), format_func=lambda value: value.strftime("%B %Y")))
        selected_start = selected_end
        selection_label = selected_end.strftime("%B %Y")
        title_prefix = "Monthly"
    elif time_mode == "Quarter":
        quarter_options = sorted({period.to_period("Q") for period in period_options}, reverse=True)
        selected_quarter = st.selectbox("Reporting quarter", quarter_options, format_func=lambda value: f"Q{value.quarter} {value.year}")
        selected_start = pd.Timestamp(selected_quarter.start_time)
        selected_end = pd.Timestamp(selected_quarter.end_time).normalize()
        selection_label = f"Q{selected_quarter.quarter} {selected_quarter.year}"
        title_prefix = "Quarterly"
    elif time_mode == "Year":
        year_options = sorted({period.year for period in period_options}, reverse=True)
        selected_year = int(st.selectbox("Reporting year", year_options))
        selected_start = pd.Timestamp(selected_year, 1, 1)
        selected_end = pd.Timestamp(selected_year, 12, 31)
        selection_label = str(selected_year)
        title_prefix = "Annual"
    else:
        chosen_range = st.date_input(
            "Reporting range",
            value=(period_options[0].date(), period_options[-1].date()),
            min_value=period_options[0].date(),
            max_value=period_options[-1].date(),
        )
        if isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 2:
            selected_start, selected_end = map(pd.Timestamp, chosen_range)
        else:
            selected_start = selected_end = pd.Timestamp(chosen_range)
        selection_label = f"{selected_start:%b %Y} – {selected_end:%b %Y}"
        title_prefix = "Custom Period"
    selected_units = st.multiselect("Business units", unit_options, default=unit_options)
    st.markdown("---")

if not selected_units:
    st.warning("Select at least one business unit.")
    st.stop()

period_kpis = kpis.loc[kpis["period"].between(selected_start, selected_end) & kpis["business_unit"].isin(selected_units)]
current = aggregate_kpis(period_kpis)
current_alerts = alerts.loc[alerts["period"].between(selected_start, selected_end) & alerts["business_unit"].isin(selected_units)]

st.markdown(
    "<div class='creator-credit'>Ziqi (Ankit) Cao &nbsp;·&nbsp; "
    "<a href='https://www.linkedin.com/in/ziqi-ankit-cao' target='_blank' rel='noopener noreferrer'>LinkedIn</a></div>",
    unsafe_allow_html=True,
)
st.image(str(LOGO_PATH), width=220)
st.title(f"Automating Three Businesses LOB Financial Review – {selection_label}")

summary, attention_items = generate_executive_summary(period_kpis, current_alerts, selection_label)
st.markdown(
    f"<div class='brief'><strong>Executive Summary</strong><span class='summary-text'>{escape(summary)}</span></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='kpi-spacer'></div>", unsafe_allow_html=True)

total_revenue = current["revenue_actual"].sum()
total_budget = current["revenue_budget"].sum()
total_profit = current["adjusted_profit_actual"].sum()
total_expense = current["operating_expense_actual"].sum()
total_forecast = current["revenue_forecast"].sum()
cards = st.columns(5, gap="medium")
cards[0].metric("Revenue", fmt_money(total_revenue), f"{(total_revenue / total_budget - 1):+.1%} vs target")
cards[1].metric("Adjusted Profit", fmt_money(total_profit), f"{(total_profit / current['adjusted_profit_budget'].sum() - 1):+.1%} vs target")
cards[2].metric("Operating Expense", fmt_money(total_expense), f"{(total_expense / current['operating_expense_budget'].sum() - 1):+.1%} vs budget", delta_color="inverse")
cards[3].metric("Target Attainment", f"{total_revenue / total_budget:.1%}")
cards[4].metric("Actual vs Forecast", f"{1 - abs(total_revenue - total_forecast) / total_forecast:.1%}")

st.markdown(
    """
    <div class="attention-heading">
      <h2>Items Requiring Attention</h2>
      <div class="alert-legend">
        <span class="legend-item"><i class="legend-dot dot-red"></i>Critical: Action Required</span>
        <span class="legend-item"><i class="legend-dot dot-yellow"></i>Caution: Needs Review</span>
        <span class="legend-item"><i class="legend-dot dot-green"></i>Positive: Exceeding Target</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
severity_rank = {"Red": 0, "Yellow": 1, "Green": 2}
grouped_alerts = current_alerts.assign(rank=current_alerts["severity"].map(severity_rank)).sort_values(["rank", "period"])
for business_unit in selected_units:
    business_alerts = grouped_alerts.loc[grouped_alerts["business_unit"] == business_unit]
    alert_rows = []
    if business_alerts.empty:
        alert_rows.append(
            "<div class='alert alert-green'><span class='alert-rule'>No exceptions</span> · "
            "No configured alerts were triggered in the selected period.</div>"
        )
    else:
        for _, alert in business_alerts.iterrows():
            message = str(alert["message"])
            if message.startswith(f"{business_unit} "):
                message = message[len(business_unit) + 1 :]
                message = message[:1].upper() + message[1:]
            alert_rows.append(
                f"<div class='alert alert-{escape(str(alert['severity']).lower())}'>"
                f"<span class='alert-rule'>{escape(str(alert['rule']))}</span> · {escape(message)}</div>"
            )
    st.markdown(
        f"<div class='alert-group'><div class='alert-business'>{escape(str(business_unit))}</div>"
        f"<div class='alert-list'>{''.join(alert_rows)}</div></div>",
        unsafe_allow_html=True,
    )

st.subheader("Business Unit Scorecard")
render_scorecard(scorecard_rows(current))
st.subheader("Business Model Metrics")
render_business_model_metrics(current)

trend_start = selected_end - pd.DateOffset(months=5) if time_mode == "Month" else selected_start
trend_data = kpis.loc[kpis["period"].between(trend_start, selected_end) & kpis["business_unit"].isin(selected_units)]
selected_detail = detail.loc[detail["period"].between(selected_start, selected_end) & detail["business_unit"].isin(selected_units)]
chart_period_label = selection_label
latest_period = trend_data["period"].max()
latest_month = trend_data.loc[trend_data["period"] == latest_period].copy()
efficiency_latest = latest_month.sort_values("cost_to_income_ratio_actual", ascending=False).iloc[0]
efficiency_lowest = latest_month.sort_values("cost_to_income_ratio_actual", ascending=True).iloc[0]
margin_latest = latest_month.sort_values("profit_margin_actual", ascending=False).iloc[0]
margin_target = float(margin_latest["profit_margin_budget"])

revenue_latest = latest_month.sort_values("revenue_actual", ascending=False).iloc[0]
revenue_history = trend_data.loc[trend_data["business_unit"] == revenue_latest["business_unit"]].sort_values("period")
prior_revenue = float(revenue_history.iloc[-2]["revenue_actual"]) if len(revenue_history) > 1 else float(revenue_latest["revenue_actual"])
revenue_summary = generate_chart_summary("revenue_trend", {
    "_version": 3,
    "unit": revenue_latest["business_unit"], "latest": round(float(revenue_latest["revenue_actual"]), 1),
    "change": round((float(revenue_latest["revenue_actual"]) / prior_revenue - 1) * 100, 1),
    "period": latest_period.strftime("%B %Y"),
})

efficiency_summary = generate_chart_summary("cost_to_income", {
    "_version": 3,
    "highest_unit": efficiency_latest["business_unit"],
    "highest": round(float(efficiency_latest["cost_to_income_ratio_actual"] * 100), 1),
    "lowest_unit": efficiency_lowest["business_unit"],
    "lowest": round(float(efficiency_lowest["cost_to_income_ratio_actual"] * 100), 1),
    "period": latest_period.strftime("%B %Y"),
})
margin_summary = generate_chart_summary("profit_margin", {
    "_version": 3,
    "highest_unit": margin_latest["business_unit"],
    "highest": round(float(margin_latest["profit_margin_actual"] * 100), 1),
    "variance": round((float(margin_latest["profit_margin_actual"]) - margin_target) * 100, 1),
    "period": latest_period.strftime("%B %Y"),
})

variance_frame = selected_detail.loc[selected_detail["metric_type"] == "Revenue"].groupby("management_category", as_index=False)[["actual", "budget"]].sum()
variance_frame["variance"] = variance_frame["actual"] - variance_frame["budget"]
largest_driver = variance_frame.loc[variance_frame["variance"].abs().idxmax(), "management_category"]
variance_summary = generate_chart_summary("revenue_variance", {
    "_version": 3,
    "variance": round(float(variance_frame["variance"].sum()), 1), "driver": str(largest_driver),
})
expense_frame = current.copy()
expense_frame["variance"] = expense_frame["operating_expense_actual"] - expense_frame["operating_expense_budget"]
largest_expense = expense_frame.sort_values("variance", ascending=False).iloc[0]
expense_summary = generate_chart_summary("expense_variance", {
    "_version": 3,
    "unit": largest_expense["business_unit"], "variance": round(float(max(largest_expense["variance"], 0)), 1),
})

with st.expander("Total Trend", expanded=True):
    revenue_trend_view, efficiency_trend_view = st.columns(2)
    with revenue_trend_view:
        st.plotly_chart(
            add_chart_summary(trend_chart(trend_data, selected_units), revenue_summary).update_layout(height=600),
            use_container_width=True, config={"displayModeBar": False},
        )
    with efficiency_trend_view:
        st.plotly_chart(
            add_chart_summary(cost_income_trend_chart(trend_data, selected_units), efficiency_summary).update_layout(height=600),
            use_container_width=True, config={"displayModeBar": False},
        )

    variance_view, expense_view = st.columns(2)
    with variance_view:
        st.plotly_chart(
            add_chart_summary(waterfall_chart(selected_detail, chart_period_label), variance_summary).update_layout(height=560),
            use_container_width=True, config={"displayModeBar": False},
        )
    with expense_view:
        st.plotly_chart(
            add_chart_summary(expense_comparison_chart(current, chart_period_label), expense_summary).update_layout(height=560),
            use_container_width=True, config={"displayModeBar": False},
        )

with st.expander("Key Business Metrics", expanded=True):
    specialized_figures = [(
        margin_summary,
        profit_margin_comparison_chart(latest_month, latest_period.strftime("%B %Y")).update_layout(height=580),
    )]
    if "Commercial Real Estate" in selected_units:
        cre = kpis.loc[(kpis["business_unit"] == "Commercial Real Estate") & kpis["period"].between(trend_start, selected_end)].sort_values("period")
        if not cre.empty:
            values = cre["npl_ratio_actual"].dropna()
            summary_text = generate_chart_summary("npl_ratio", {
                "_version": 3,
                "business_unit": "Commercial Real Estate", "latest": round(float(values.iloc[-1] * 100), 2),
                "prior": round(float(values.iloc[-2] * 100), 2) if len(values) > 1 else round(float(values.iloc[-1] * 100), 2),
                "months": int(len(values)),
            })
            specialized_figures.append((summary_text, single_ratio_trend_chart(
                cre, "npl_ratio_actual", "Commercial Real Estate NPL Ratio",
                "NPL Proxy / CRE Loan Balance", threshold=0.015,
            ).update_layout(height=580)))
    if "Commercial Banking" in selected_units:
        banking = kpis.loc[(kpis["business_unit"] == "Commercial Banking") & kpis["period"].between(trend_start, selected_end)].sort_values("period")
        if not banking.empty:
            values = banking["loan_to_deposit_ratio_actual"].dropna()
            summary_text = generate_chart_summary("loan_to_deposit", {
                "_version": 3,
                "business_unit": "Commercial Banking", "latest": round(float(values.iloc[-1] * 100), 1),
                "prior": round(float(values.iloc[-2] * 100), 1) if len(values) > 1 else round(float(values.iloc[-1] * 100), 1),
                "months": int(len(values)), "period": banking.iloc[-1]["period"].strftime("%B %Y"),
            })
            specialized_figures.append((summary_text, single_ratio_trend_chart(
                banking, "loan_to_deposit_ratio_actual", "Commercial Banking Loan to Deposit Ratio",
                "Loan Balance / Deposit Balance",
            ).update_layout(height=580)))
    if "Capital Markets" in selected_units:
        markets = kpis.loc[(kpis["business_unit"] == "Capital Markets") & (kpis["period"] <= selected_end)].sort_values("period").tail(6)
        if not markets.empty:
            mix_columns = {
                "Advisory": "advisory_mix_actual", "Underwriting": "underwriting_mix_actual",
                "Trading": "trading_mix_actual", "Structuring": "structuring_mix_actual", "Syndication": "syndication_mix_actual",
            }
            latest_mix = markets.iloc[-1]
            largest_component = max(mix_columns, key=lambda label: latest_mix[mix_columns[label]])
            summary_text = generate_chart_summary("fee_revenue_mix", {
                "_version": 3,
                "business_unit": "Capital Markets", "largest_component": largest_component,
                "largest_share": round(float(latest_mix[mix_columns[largest_component]] * 100), 1), "months": int(len(markets)),
                "period": latest_mix["period"].strftime("%B %Y"),
            })
            specialized_figures.append((summary_text, fee_revenue_mix_chart(markets).update_layout(height=580, margin=dict(b=112))))
    if specialized_figures:
        left_column, right_column = st.columns(2)
        midpoint = (len(specialized_figures) + 1) // 2
        for column, figures in ((left_column, specialized_figures[:midpoint]), (right_column, specialized_figures[midpoint:])):
            with column:
                for summary_text, figure in figures:
                    st.plotly_chart(add_chart_summary(figure, summary_text), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Select a business unit to view its specialized metric.")
