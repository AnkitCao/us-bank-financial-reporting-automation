"""Streamlit entry point for the Automated Three-Business Performance Dashboard."""

from __future__ import annotations

import base64
import json
import re
import sys
import textwrap
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executive_summary import generate_executive_summary
from src.paths import PROCESSED_DIR, RAW_DIR

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "us-bank-logo.svg"
LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")

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


@st.cache_data
def load_raw_source_evidence(start: pd.Timestamp, end: pd.Timestamp, units: tuple[str, ...]) -> list[dict]:
    """Return selected source-workbook rows for LLM evidence; dashboard calculations still use clean data."""
    rows: list[dict] = []
    if "Commercial Banking" in units:
        frame = pd.read_excel(RAW_DIR / "commercial_banking_monthly.xlsx", sheet_name="Monthly Detail")
        dates = pd.to_datetime(frame["Reporting Date "], errors="coerce")
        frame = frame.loc[dates.between(start, end)].copy()
        frame = frame.loc[frame["Plan Type"].astype(str).str.strip().str.lower() != "forecast"]
        frame.insert(0, "Business Unit", "Commercial Banking")
        rows.extend(frame.to_dict("records"))
    if "Commercial Real Estate" in units:
        frame = pd.read_excel(RAW_DIR / "commercial_real_estate_monthly.xlsx", sheet_name="CRE Monthly")
        dates = pd.to_datetime(frame["Report Month"], errors="coerce") + pd.offsets.MonthEnd(0)
        frame = frame.loc[dates.between(start, end)].copy()
        frame = frame.loc[frame["Plan Case"].astype(str).str.strip().str.lower() != "forecast"]
        frame.insert(0, "Business Unit", "Commercial Real Estate")
        rows.extend(frame.to_dict("records"))
    if "Capital Markets" in units:
        path = RAW_DIR / "capital_markets_monthly.xlsx"
        for sheet in pd.ExcelFile(path).sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            frame = frame.loc[frame["Plan Scenario"].astype(str).str.strip().str.lower() != "forecast"]
            selected_columns = ["Plan Scenario"]
            for column in frame.columns[1:]:
                parsed = pd.to_datetime(str(column).replace("_", "-"), format="%b-%y", errors="coerce")
                if pd.notna(parsed) and start.to_period("M") <= parsed.to_period("M") <= end.to_period("M"):
                    selected_columns.append(column)
            if len(selected_columns) > 1:
                subset = frame[selected_columns].copy()
                subset.insert(0, "Source Metric", sheet)
                subset.insert(0, "Business Unit", "Capital Markets")
                rows.extend(subset.to_dict("records"))
    return rows


def attach_evidence(metrics: dict, raw_rows: list[dict], calculation_rows: pd.DataFrame | None = None) -> dict:
    """Attach compact numerical evidence so the LLM analyzes data instead of paraphrasing prose."""
    evidence = dict(metrics)
    evidence["raw_source_records"] = raw_rows
    if calculation_rows is not None and not calculation_rows.empty:
        columns = [
            column for column in (
                "business_unit", "period", "source_metric", "management_category", "metric_type",
                "actual", "budget", "prior_year", "variance_to_budget",
            ) if column in calculation_rows.columns
        ]
        prepared = calculation_rows[columns].copy()
        if "period" in prepared:
            prepared["period"] = prepared["period"].dt.strftime("%Y-%m-%d")
        evidence["calculation_records"] = prepared.to_dict("records")
    return evidence


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a filtered analytical frame into JSON-safe records for LLM review."""
    prepared = frame.copy()
    for column in prepared.select_dtypes(include=["datetime", "datetimetz"]).columns:
        prepared[column] = prepared[column].dt.strftime("%Y-%m-%d")
    return prepared.where(pd.notna(prepared), None).to_dict("records")


def sanitize_llm_evidence(value):
    """Remove every forecast field and forecast record before an LLM call."""
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if isinstance(item, dict):
                scenario = next(
                    (
                        item.get(key)
                        for key in ("scenario", "Plan Type", "Plan Case", "Plan Scenario")
                        if key in item
                    ),
                    None,
                )
                if str(scenario).strip().lower() == "forecast":
                    continue
                narrative_values = [
                    item.get(key) for key in ("message", "rule", "metric", "label") if key in item
                ]
                if any("forecast" in str(text).lower() for text in narrative_values):
                    continue
            sanitized_item = sanitize_llm_evidence(item)
            if sanitized_item is not None:
                cleaned.append(sanitized_item)
        return cleaned
    if isinstance(value, dict):
        return {
            key: sanitize_llm_evidence(item)
            for key, item in value.items()
            if "forecast" not in str(key).lower()
        }
    if isinstance(value, str) and "forecast" in value.lower():
        return None
    return value


def normalize_llm_sentence(text: str) -> str:
    """Normalize one LLM sentence and reject forbidden forecast language."""
    normalized = re.sub(r"&#x?20;|&nbsp;", " ", str(text), flags=re.IGNORECASE)
    normalized = normalized.replace("：", ":").replace("—", ", ").replace("–", ", ").replace("·", ": ").replace("•", "")
    field_names = {
        "revenue_actual": "Actual Revenue",
        "revenue_budget": "Budget Revenue",
        "adjusted_profit_actual": "Actual Adjusted Profit",
        "operating_expense_actual": "Actual Operating Expense",
        "operating_expense_budget": "Budget Operating Expense",
        "credit_provision_actual": "Actual Credit Provision",
        "Revenue actual": "Actual Revenue",
        "Operating Expense actual": "Actual Operating Expense",
        "Credit Provision actual": "Actual Credit Provision",
    }
    for source_name, display_name in field_names.items():
        normalized = normalized.replace(source_name, display_name)
    normalized = re.sub(r"(?<=\d)\s+(?:percentage\s+)?points?\b", "%", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?<![\w.])-?\d+\.\d{2,}(?!\w)", lambda match: f"{float(match.group()):.1f}", normalized)
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"\s+([,:;.])", r"\1", normalized)
    if "forecast" in normalized.lower():
        return ""
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def has_quantitative_evidence(text: str) -> bool:
    """Require a measurable value, not merely a calendar year, in an LLM conclusion."""
    patterns = (
        r"\$\s*[+-]?\d",                         # money
        r"[+-]?\d+(?:\.\d+)?\s*%",            # percentage or ratio
        r"\b[+-]?\d+(?:\.\d+)?\s*(?:M|B)\b", # scaled amount
        r"\b\d+(?:\.\d+)?\s+months?\b",      # numeric duration/count
        r"\b[+-]?\d+\.\d+\b",                 # other decimal metric
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def every_sentence_has_quantitative_evidence(text: str) -> bool:
    """Ensure each generated conclusion sentence contains auditable numerical evidence."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return bool(sentences) and all(has_quantitative_evidence(sentence) for sentence in sentences)


def normalize_insight_label(text: str) -> str:
    """Keep insight headings short and visually consistent."""
    cleaned = re.sub(r"&#x?20;|&nbsp;", " ", str(text), flags=re.IGNORECASE)
    label = " ".join(cleaned.replace("：", ":").replace(":", "").split())
    label = re.sub(r"\band\b", "&", label, flags=re.IGNORECASE)
    return label[:1].upper() + label[1:] if label else ""


def format_insight_text_html(text: str) -> str:
    """Escape insight text and bold only measurable amounts and percentages."""
    safe_text = escape(text)
    return re.sub(
        r"(\$[+-]?\d[\d,]*(?:\.\d+)?[MB]?|[+-]?\d+(?:\.\d+)?%)",
        r"<strong>\1</strong>",
        safe_text,
    )


def fmt_money(value: float) -> str:
    """Format a USD millions value for executive display."""
    return f"${value:,.1f}M"


def overall_performance_sentence(overall: dict) -> str:
    """Build the mandatory first executive-summary sentence from shared KPI totals."""
    revenue_direction = "above target" if overall["revenue_variance"] >= 0 else "below target"
    expense_direction = "above budget" if overall["expense_variance"] >= 0 else "below budget"
    subject = "the three businesses" if overall["business_unit_count"] == 3 else "the selected businesses"
    return (
        f"Overall, {subject} generated {fmt_money(overall['revenue_actual'])}, "
        f"{abs(overall['revenue_variance']):.1%} {revenue_direction}; expenses were "
        f"{fmt_money(overall['operating_expense_actual'])}, {abs(overall['expense_variance']):.1%} {expense_direction}."
    )


def department_performance_sentence(row: pd.Series) -> str:
    """Create one concise, auditable department line for the executive summary."""
    direction = "above target" if row["revenue_vs_budget"] >= 0 else "below target"
    return (
        f"{row['business_unit']} generated {fmt_money(row['revenue_actual'])} in revenue, "
        f"{abs(row['revenue_vs_budget']):.1%} {direction}."
    )


def format_chart_period_range(frame: pd.DataFrame) -> str:
    """Return a concise month or month range for chart subtitles."""
    periods = pd.to_datetime(frame["period"], errors="coerce").dropna()
    if periods.empty:
        return ""
    start, end = periods.min(), periods.max()
    if start.to_period("M") == end.to_period("M"):
        return end.strftime("%B %Y")
    return f"{start.strftime('%b %Y')} - {end.strftime('%b %Y')}"


def local_chart_summary(chart_key: str, values: dict) -> str:
    """Return a concise deterministic summary when an API key is unavailable."""
    if chart_key == "cost_to_income":
        gap = values["highest"] - values["lowest"]
        return f"{values['highest_unit']} had the highest cost-to-income ratio ({values['highest']:.1f}%), {gap:.1f}% above {values['lowest_unit']}."
    if chart_key == "profit_margin":
        return f"{values['highest_unit']} led profit margins ({values['highest']:.1f}%), {values['variance']:+.1f}% versus target."
    if chart_key == "revenue_trend":
        return f"{values['unit']} revenue reached ${values['latest']:.1f}M, changing {values['change']:+.1f}% from the prior month."
    if chart_key == "revenue_variance":
        return f"Revenue finished ${abs(values['variance']):.1f}M {'above' if values['variance'] >= 0 else 'below'} target, led by {values['driver']}."
    if chart_key == "expense_variance":
        return f"{values['unit']} expense was ${values['variance']:.1f}M above budget, the largest unfavorable variance."
    if chart_key == "npl_ratio":
        direction = "increased" if values["latest"] > values["prior"] else "decreased"
        return f"CRE NPL ratio {direction} to {values['latest']:.1f}%, remaining below the illustrative 1.5% threshold."
    if chart_key == "loan_to_deposit":
        direction = "increased" if values["latest"] > values["prior"] else "decreased"
        return f"Commercial Banking loan-to-deposit ratio {direction} to {values['latest']:.1f}%, reflecting loans relative to deposit funding."
    return f"{values['largest_component']} was Capital Markets' largest fee component ({values['largest_share']:.1f}%)."


def enforce_chart_summary(text: str, fallback: str) -> str:
    """Accept only a short, quantitative chart sentence; otherwise use the fallback."""
    candidate = normalize_llm_sentence(text)
    words = re.findall(r"\b[\w$%+.']+(?:-[\w$%+.']+)*\b", candidate)
    sentence_marks = re.findall(r"[.!?]", candidate)
    has_numeric_evidence = every_sentence_has_quantitative_evidence(candidate)
    if candidate and has_numeric_evidence and len(words) <= 22 and len(candidate) <= 150 and len(sentence_marks) == 1:
        return candidate
    return normalize_llm_sentence(fallback)


@st.cache_data(ttl=3600, show_spinner=False)
def generate_chart_summary(chart_key: str, summary_dict: dict) -> str:
    """Choose the most decision-relevant finding from the chart's filtered data."""
    fallback = local_chart_summary(chart_key, summary_dict)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return enforce_chart_summary(fallback, fallback)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-5-mini"),
            instructions=(
                "Analyze all supplied records for this chart and choose only its most decision-relevant exception, trend, or implication. Treat "
                "calculated clean metrics as authoritative when raw rows contain deliberate quality issues. Return only one complete, factual, "
                "executive-friendly sentence with no label, quotation marks, explanation, heading, bullet, colon, or introductory phrase. The final "
                "sentence must contain no more than 22 English words and no more than 150 characters including spaces. Mention each business unit "
                "no more than once and normally mention only the most relevant unit unless comparison is essential. Include no more than two "
                "numeric values. Every sentence must include at least one Arabic-numeral quantitative value from the supplied data, such as an "
                "amount, percentage, ratio, variance, or month count. Express every percentage difference with the % symbol; never use point, "
                "points, percentage point, or percentage points. Format every decimal to exactly one decimal place. Never return a qualitative "
                "trend statement without a number. Prioritize "
                "the variance instead of repeating Actual, Budget, and Prior Year values. After naming a metric once, "
                "use concise phrases such as above budget, below budget, above May, or above prior year. Do not repeatedly write long names such "
                "as Actual Revenue, Budget Revenue, Actual Operating Expense, or Budget Operating Expense. Compare levels, period changes, target "
                "or budget gaps, and business differences only where material. "
                "Never mention, analyze, compare, or output Forecast. Use only Actual, Budget, Target, and Prior Year facts. Whenever a metric "
                "name and its value appear together, write natural business language. Translate snake_case fields and synthesize figures rather "
                "than mechanically listing fields. Format every displayed number to exactly one decimal place. End with a period. Do not use an "
                "em dash or en dash. Preserve supplied facts; never invent causes, explanations, thresholds, recommendations, or missing values. "
                "Before returning, count the words and characters and rewrite until both limits are satisfied."
            ),
            input=json.dumps(sanitize_llm_evidence({"chart": chart_key, "filtered_data": summary_dict}), default=str),
            store=False,
        )
        return enforce_chart_summary(response.output_text, fallback)
    except Exception:
        return enforce_chart_summary(fallback, fallback)


@st.cache_data(ttl=3600, show_spinner=False)
def generate_ai_period_review(facts: dict, fallback_summary: str, fallback_reviews: list[dict]) -> dict:
    """Let the LLM select period-specific executive insights from all filtered evidence."""
    fallback = {"executive_summary": fallback_summary, "business_reviews": fallback_reviews}
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return fallback
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=st.secrets.get("OPENAI_MODEL", "gpt-5-mini"),
            instructions=(
                "Act as a CFO reviewing one selected reporting period. Analyze every supplied raw record, clean monthly KPI record, calculated "
                "metric record, and deterministic rule alert. Decide what is genuinely most material for this period; do not always choose the "
                "same metrics or reuse a fixed template. Consider target gaps, expense pressure, trend reversals, margin, cost-to-income, NPL, "
                "loan-to-deposit, fee concentration, and alert persistence. Return JSON only with: "
                '{"executive_summary":"one Overall sentence followed by one sentence per supplied business unit","business_reviews":['
                '{"business_unit":"allowed supplied name","status":"Critical|Caution|Positive","insights":['
                '{"title":"plain-text headline of no more than four words","text":"one quantitative factual sentence"}]}]}. '
                "The first executive-summary sentence must begin with Overall, and quantify the combined performance of all supplied business "
                "units. It must include exactly four overall figures from overall_performance: combined actual revenue; revenue variance versus "
                "target as a percentage; combined actual operating expense; and operating-expense variance versus budget as a percentage. Use "
                "this concise structure with supplied values: Overall, the three businesses generated $31.5M, 6.9% above target; expenses were "
                "$10.4M, 4.2% above budget. Replace every example value with overall_performance values. Use below "
                "target or below budget for negative variances. Never add another number to the first sentence. Never describe overall performance "
                "without these four figures. After the Overall sentence, write exactly one separate sentence for each supplied business unit, in "
                "the supplied order. For each department, evaluate every metric and every change inside the selected reporting period, including "
                "revenue, expense, profit, margin, cost-to-income, loans, deposits, NPL, fee mix, target or budget variance, and time trend. Independently "
                "choose the one or two findings that are most material and worthy of executive attention; do not default to revenue or expense. Combine "
                "those findings into one complete sentence that starts with the business-unit name, uses only figures supported by the selected-period "
                "records, contains no more than 32 words and four numeric values, and states a clear comparison, change, high, low, or exception rather "
                "than merely listing metric names and values. A month-specific finding is allowed inside a quarter or year only when it explains a "
                "material change within that selected period. Avoid repetitive phrases such as Actual Revenue and Budget Revenue when revenue and its "
                "variance communicate the same fact. Return exactly one review for each supplied "
                "business unit. If a unit has no material concern, use Positive and state its most useful favorable or stable fact. "
                "Return one or two insights per business unit. Every executive-summary sentence and every insight text must include at least one measurable Arabic-numeral value "
                "from the supplied data, such as an amount, percentage, ratio, variance, or numeric month count. A calendar year alone does not "
                "satisfy this requirement. Never return a qualitative conclusion without quantitative evidence. Express percentage differences "
                "with the % symbol and never use point, points, percentage point, or percentage points. Preserve all numbers and periods "
                "exactly; never invent thresholds, causes, or recommendations. Never mention, analyze, compare, or output "
                "Forecast. Use only Actual, Budget, Target, and Prior Year facts. Whenever a metric name and value appear together, use Metric "
                "Name (value), using natural adjective-first names such as Actual Revenue ($11.3M), Budget Revenue ($9.5M), Actual "
                "Operating Expense ($4.0M), and Budget Operating Expense ($3.7M). Never write Revenue actual or Expense actual. Give every insight "
                "its own short decision-oriented title. Return plain text only inside JSON: no HTML, Markdown, entities such as &#x20;, bullets, or "
                "bold markup. Do not put a colon in title or text; the interface adds an English colon and all visual formatting. State the result first, "
                "then the comparison. When Actual and Budget are available, calculate and state the variance, using natural language such as "
                "Revenue reached $11.3M, exceeding budget by $1.8M. Use above budget, below budget, or in line with budget consistently. Avoid "
                "repeating Actual Revenue and Budget Revenue when Revenue and the variance say the same thing. Translate technical field names "
                "into natural finance language: revenue_actual becomes Actual Revenue, "
                "adjusted_profit_actual becomes Actual Adjusted Profit, operating_expense_actual becomes Actual Operating Expense, and "
                "credit_provision_actual becomes Actual Credit Provision. Do not expose snake_case field names or mechanically list fields. "
                "Synthesize the figures into fluent executive language. Preserve every source fact, but never invent causes, thresholds, values, "
                "or recommendations. Format every displayed number to exactly one decimal place, including money, percentages, ratios, and "
                "variances. Keep each review near 30 words when possible. Write complete sentences ending with periods. Do not use em dashes or "
                "en dashes."
            ),
            input=json.dumps(sanitize_llm_evidence(facts), default=str),
            store=False,
        )
        parsed = json.loads(response.output_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        allowed_units = set(facts["selected_business_units"])
        reviews = []
        seen = set()
        for item in parsed.get("business_reviews", []):
            unit = str(item.get("business_unit", ""))
            status = str(item.get("status", ""))
            if unit not in allowed_units or unit in seen or status not in {"Critical", "Caution", "Positive"}:
                continue
            insights = []
            for insight in item.get("insights", []):
                title = normalize_insight_label(str(insight.get("title", "")))
                message = normalize_llm_sentence(str(insight.get("text", "")))
                if title and message and "forecast" not in title.lower() and every_sentence_has_quantitative_evidence(message):
                    insights.append({"title": title, "text": message})
            if insights:
                reviews.append({"business_unit": unit, "status": status, "insights": insights[:2]})
                seen.add(unit)
        summary = normalize_llm_sentence(str(parsed.get("executive_summary", "")))
        if not summary or not every_sentence_has_quantitative_evidence(summary) or seen != allowed_units:
            return fallback
        generated_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", summary)
            if sentence.strip()
        ][1:]
        canonical_first_sentence = overall_performance_sentence(facts["overall_performance"])
        department_lines = []
        conclusion_verbs = re.compile(
            r"\b(?:was|were|is|are|had|reached|exceeded|beat|grew|rose|increased|declined|fell|decreased|"
            r"improved|deteriorated|led|lagged|recorded|delivered|generated|posted|remained|moved)\b",
            re.IGNORECASE,
        )
        for unit in facts["selected_business_units"]:
            fallback_line = facts["department_summary_fallbacks"][unit]
            generated_line = next(
                (sentence for sentence in generated_sentences if sentence.startswith(unit)),
                fallback_line,
            )
            word_count = len(re.findall(r"\b[\w$%+.']+(?:-[\w$%+.']+)*\b", generated_line))
            if (
                not every_sentence_has_quantitative_evidence(generated_line)
                or not conclusion_verbs.search(generated_line)
                or word_count > 32
            ):
                generated_line = fallback_line
            department_lines.append(generated_line)
        summary = " ".join([canonical_first_sentence, *department_lines])
        return {"executive_summary": summary, "business_reviews": reviews}
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
    # Plotly titles do not wrap automatically. These charts render two per row;
    # 72 characters lets one more short word fit while retaining a safe right margin.
    summary_lines = textwrap.wrap(summary, width=72, break_long_words=False, break_on_hyphens=False)
    summary_html = "<br>".join(escape(line) for line in summary_lines[:3])
    title_margin = 198 + max(0, len(summary_lines[:3]) - 1) * 34
    fig.update_layout(
        title_text=(
            f"{current_title}<br>"
            f"<span style='font-size:25px;color:#334155'><b>★</b> {summary_html}</span>"
        ),
        margin=dict(t=title_margin),
    )
    return fig


def render_plotly_chart(fig: go.Figure, height: int) -> None:
    """Render at the pre-scale card width so the 50% page transform stays crisp."""
    # Streamlit measures Plotly after the app's 50% transform, which otherwise
    # compresses a chart into half of its card. Render at double resolution inside
    # an iframe, then scale that canvas once inside the iframe and once with the app.
    chart_width = 1760
    chart_height = round(height * 2.36)
    frame_width = chart_width // 2
    frame_height = chart_height // 2

    def doubled(value: object) -> object:
        return value * 2 if isinstance(value, (int, float)) else value

    title_text = fig.layout.title.text or ""
    title_text = re.sub(
        r"font-size:(\d+(?:\.\d+)?)px",
        lambda match: f"font-size:{float(match.group(1)) * 2:g}px",
        title_text,
    )
    margin = fig.layout.margin
    fig.update_layout(
        width=chart_width,
        height=chart_height,
        autosize=False,
        title_text=title_text,
        title_font_size=doubled(fig.layout.title.font.size),
        font_size=doubled(fig.layout.font.size),
        legend_font_size=doubled(fig.layout.legend.font.size),
        hoverlabel_font_size=doubled(fig.layout.hoverlabel.font.size),
        margin=dict(
            l=doubled(margin.l), r=doubled(margin.r),
            t=doubled(margin.t), b=doubled(margin.b),
        ),
    )
    for axis in [*fig.select_xaxes(), *fig.select_yaxes()]:
        axis.tickfont.size = doubled(axis.tickfont.size)
        axis.title.font.size = doubled(axis.title.font.size)
    for annotation in fig.layout.annotations or ():
        annotation.font.size = doubled(annotation.font.size)
    for trace in fig.data:
        if getattr(trace, "textfont", None) is not None:
            trace.textfont.size = doubled(trace.textfont.size)
        if getattr(trace, "line", None) is not None:
            trace.line.width = doubled(trace.line.width)
        if getattr(trace, "marker", None) is not None:
            trace.marker.line.width = doubled(trace.marker.line.width)

    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": False},
    )
    components.html(
        "<style>html,body{margin:0;overflow:hidden;background:#fff;}"
        ".scaled-chart{width:1760px;height:auto;transform:scale(.5);transform-origin:top left;}"
        f"</style><div class='scaled-chart'>{chart_html}</div>",
        height=frame_height,
        width=frame_width,
        scrolling=False,
    )


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
    return base_layout(fig, "Revenue Trends", f"= Sum of Revenue Metrics, {format_chart_period_range(kpis)}")


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
    return base_layout(fig, "Cost-to-Income Ratios", f"= Operating Expense ÷ Total Revenue, {format_chart_period_range(kpis)}")


def profit_margin_comparison_chart(period_frame: pd.DataFrame, period_label: str) -> go.Figure:
    """Compare actual and target profit margins for the selected period."""
    frame = period_frame.copy()
    frame["business_unit"] = pd.Categorical(frame["business_unit"], BUSINESS_UNIT_ORDER, ordered=True)
    frame = frame.sort_values("business_unit")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual", x=frame["business_unit"], y=frame["profit_margin_actual"], width=0.24,
        offsetgroup="actual",
        marker_color="#2563EB",
        hovertemplate="<b>%{x}</b><br>Actual Margin: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Target", x=frame["business_unit"], y=frame["profit_margin_budget"], width=0.24,
        offsetgroup="target",
        marker_color="#F4B400",
        hovertemplate="<b>%{x}</b><br>Target Margin: %{y:.1%}<extra></extra>",
    ))
    upper = max(frame["profit_margin_actual"].max(), frame["profit_margin_budget"].max()) * 1.22
    fig.update_yaxes(title="Profit Margin", tickformat=".0%", range=[0, upper])
    fig.update_layout(barmode="group", bargap=0.46, bargroupgap=0.12)
    return base_layout(fig, "Profit Margins by Business Units", f"= Adjusted Profit ÷ Total Revenue, {period_label}")


def single_ratio_trend_chart(frame: pd.DataFrame, column: str, title: str, subtitle: str, threshold: float | None = None) -> go.Figure:
    """Render a business-specific ratio trend."""
    frame = frame.sort_values("period")
    fig = go.Figure(go.Scatter(
        x=frame["period"].dt.strftime("%b %Y"), y=frame[column], name=title, mode="lines+markers",
        line=dict(color="#2563EB", width=4), marker=dict(size=11),
        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Y: %{y:.1%}<extra></extra>",
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
    return base_layout(fig, title, f"{subtitle}, {format_chart_period_range(frame)}")


def fee_revenue_mix_chart(frame: pd.DataFrame) -> go.Figure:
    """Show Capital Markets fee composition for the selected period."""
    components = [
        ("Advisory", "advisory_mix_actual"), ("Underwriting", "underwriting_mix_actual"),
        ("Trading", "trading_mix_actual"), ("Structuring", "structuring_mix_actual"),
        ("Syndication", "syndication_mix_actual"),
    ]
    frame = frame.sort_values("period")
    fig = go.Figure()
    for (label, column), color in zip(components, MIX_COLORS):
        fig.add_trace(go.Bar(
            name=label, x=frame["period"].dt.strftime("%b %Y"), y=frame[column], marker_color=color,
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Y: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_yaxes(title="Revenue Mix", tickformat=".0%", range=[0, 1])
    fig.update_layout(barmode="stack", bargap=0.35)
    fig = base_layout(fig, "Fee Revenue Shares (Capital Markets)", f"= Fee Component ÷ Total Fee Revenue, {format_chart_period_range(frame)}")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5, font=dict(size=17)),
        margin=dict(l=92, r=48, t=174, b=108),
    )
    return fig


def waterfall_chart(detail: pd.DataFrame, period_label: str) -> go.Figure:
    """Explain target-to-actual revenue variance with readable driver contribution bars."""
    frame = detail.loc[detail["metric_type"] == "Revenue"].copy()
    frame = frame.groupby("management_category", as_index=False)[["actual", "budget"]].sum()
    frame["variance"] = frame["actual"] - frame["budget"]
    budget = frame["budget"].sum()
    actual = frame["actual"].sum()
    frame["Driver"] = frame["management_category"].replace({"Fee / Noninterest Income": "Fee / Noninterest Income"})
    frame = frame.sort_values("variance", key=lambda values: values.abs(), ascending=True)
    colors = ["#2563EB" if value >= 0 else "#FF3B30" for value in frame["variance"]]
    text_values = ["$0.0M" if abs(value) < 0.05 else f"${value:+.1f}M" for value in frame["variance"]]
    hover_data = [
        [f"${budget_value:.1f}M", f"${actual_value:.1f}M", f"${variance_value:+.1f}M"]
        for budget_value, actual_value, variance_value in zip(frame["budget"], frame["actual"], frame["variance"])
    ]
    fig = go.Figure(go.Bar(
        x=frame["variance"], y=frame["Driver"], orientation="h",
        marker_color=colors, width=0.48, text=text_values, textposition="outside",
        textfont=dict(size=22, color="#001E79"),
        customdata=hover_data,
        hovertemplate=(
            "<b>%{y}</b><br>Contribution: %{customdata[2]}"
            "<br>Target: %{customdata[0]}<br>Actual: %{customdata[1]}<extra></extra>"
        ),
    ))
    net_change = actual - budget
    net_color = "#2563EB" if net_change >= 0 else "#FF3B30"
    fig.add_annotation(
        x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False, align="center",
        text=(
            f"<span style='color:{net_color}'><b>Total Net Changes ${net_change:+.1f}M</b></span>"
            f" &nbsp;&nbsp; = &nbsp;&nbsp; <b>Actual ${actual:.1f}M</b>"
            f" &nbsp;&nbsp; − &nbsp;&nbsp; <b>Target ${budget:.1f}M</b>"
        ),
        font=dict(family="Times New Roman, Times, serif", size=23, color="#001E79"),
    )
    minimum_variance = float(frame["variance"].min())
    maximum_variance = float(frame["variance"].max())
    max_abs = max(float(frame["variance"].abs().max()), 0.5)
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1.5)
    if minimum_variance >= 0:
        x_range = [0, max(maximum_variance * 1.28, 0.5)]
    elif maximum_variance <= 0:
        x_range = [min(minimum_variance * 1.28, -0.5), 0]
    else:
        x_range = [-max_abs * 1.25, max_abs * 1.25]
    fig.update_xaxes(title="Contribution to Revenue Variance ($M)", range=x_range, zeroline=False)
    fig.update_yaxes(title="", tickfont=dict(size=19), ticklabelstandoff=28, automargin=True)
    fig.update_layout(showlegend=False, bargap=0.38)
    return base_layout(fig, "Revenue Results vs. Targets", f"= Actual − Target, {period_label}")


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
    return base_layout(fig, "Revenues by Business Units", f"= Sum of Actual Revenue Metrics, {period_label}")


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
    return base_layout(fig, "Adjusted Profits by Business Units", f"= Revenue − Operating Expense − Credit Provision (where applicable), {period_label}")


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
            width=0.24,
            offsetgroup="actual",
            alignmentgroup="expense",
            marker_color="#2563EB",
            hovertemplate="<b>%{x}</b><br>Actual Expense: $%{y:.1f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Budget",
            x=frame["business_unit"],
            y=frame["operating_expense_budget"],
            width=0.24,
            offsetgroup="budget",
            alignmentgroup="expense",
            marker_color="#F4B400",
            hovertemplate="<b>%{x}</b><br>Budget Expense: $%{y:.1f}M<extra></extra>",
        )
    )
    upper = max(frame["operating_expense_actual"].max(), frame["operating_expense_budget"].max()) * 1.24
    fig.update_yaxes(title="Operating Expense ($M)", range=[0, upper])
    fig.update_layout(barmode="group", bargap=0.50, bargroupgap=0.28)
    return base_layout(fig, "Operating Expenses vs. Budgets", f"= Actual Operating Expense − Budget Operating Expense, {period_label}")


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
    result["profit_margin_budget"] = result["adjusted_profit_budget"] / result["revenue_budget"]
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
    result = current[["business_unit", "revenue_actual", "revenue_vs_budget", "revenue_yoy", "adjusted_profit_actual", "profit_margin_actual"]].copy()
    result["status"] = result.apply(
        lambda row: "Green" if row["revenue_vs_budget"] >= 0 and row["profit_margin_actual"] >= 0.35 else ("Yellow" if row["revenue_vs_budget"] >= -0.05 else "Red"),
        axis=1,
    )
    return result


def render_scorecard(scorecard: pd.DataFrame) -> None:
    """Render a bounded HTML scorecard with clear status indicators."""
    header = "<tr><th>Business Units</th><th>Statuses</th><th>Revenues</th><th>Revenues vs. Targets</th><th>YoY</th><th>Profits</th><th>Margins</th></tr>"
    rows = []
    status_colors = {"Green": "#2563EB", "Yellow": "#C89B3C", "Red": "#B74242"}
    status_labels = {"Green": "Positive", "Yellow": "Caution", "Red": "Critical"}
    for _, row in scorecard.iterrows():
        badge = f"<span class='status' style='background:{status_colors[row['status']]}'>{status_labels[row['status']]}</span>"
        rows.append(
            "<tr>"
            f"<td class='unit'>{row['business_unit']}</td><td>{badge}</td>"
            f"<td>{fmt_money(row['revenue_actual'])}</td><td>{row['revenue_vs_budget']:+.1%}</td>"
            f"<td>{row['revenue_yoy']:+.1%}</td><td>{fmt_money(row['adjusted_profit_actual'])}</td>"
            f"<td>{row['profit_margin_actual']:.1%}</td></tr>"
        )
    st.markdown(f"<div class='score-wrap'><table class='scorecard'>{header}{''.join(rows)}</table></div>", unsafe_allow_html=True)


def render_business_model_metrics(current: pd.DataFrame) -> None:
    """Show one department per row with detailed subrows for specialized metrics."""
    rows = []
    ordered = current.copy()
    ordered["business_unit"] = pd.Categorical(ordered["business_unit"], BUSINESS_UNIT_ORDER, ordered=True)
    ordered = ordered.sort_values("business_unit")
    for _, row in ordered.iterrows():
        unit = row["business_unit"]
        if unit == "Commercial Banking":
            specialized = [(
                "Loan-to-Deposit Ratios",
                "= Loan Balance ÷ Deposit Balance",
                f"{row['loan_to_deposit_ratio_actual']:.1%}",
            )]
        elif unit == "Commercial Real Estate":
            specialized = [(
                "NPL Ratios",
                "= NPL Proxy ÷ CRE Loan Balance",
                f"{row['npl_ratio_actual']:.1%}",
            )]
        else:
            specialized = [
                ("Advisory Share", "= Advisory Fee ÷ Total Fee Revenue", f"{row['advisory_mix_actual']:.1%}"),
                ("Underwriting Share", "= Underwriting Revenue ÷ Total Fee Revenue", f"{row['underwriting_mix_actual']:.1%}"),
                ("Trading Share", "= Trading Revenue ÷ Total Fee Revenue", f"{row['trading_mix_actual']:.1%}"),
                ("Structuring Share", "= Structuring Fee ÷ Total Fee Revenue", f"{row['structuring_mix_actual']:.1%}"),
                ("Syndication Share", "= Syndication Fee ÷ Total Fee Revenue", f"{row['syndication_mix_actual']:.1%}"),
            ]
        specialized_html = "".join(
            "<div class='other-metric-line'>"
            f"<span class='other-name'>{escape(metric)}</span>"
            f"<span class='other-formula'>{escape(calculation)}</span>"
            f"<strong class='other-value'>{escape(value)}</strong></div>"
            for metric, calculation, value in specialized
        )
        rows.append(
            "<tr>"
            f"<td>{escape(str(unit))}</td>"
            f"<td>{row['profit_margin_actual']:.1%}</td>"
            f"<td>{row['cost_to_income_ratio_actual']:.1%}</td>"
            f"<td class='other-metrics-cell'>{specialized_html}</td></tr>"
        )
    header = "<tr><th>Business Units</th><th>Profit Margins</th><th>Cost-to-Income Ratios</th><th>Other Metrics</th></tr>"
    st.markdown(f"<div class='ratio-wrap'><table class='ratio-table'>{header}{''.join(rows)}</table></div>", unsafe_allow_html=True)


st.set_page_config(page_title="Automated Three-Business Performance Dashboard", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      .stApp { width:200%; min-height:200vh; max-width:none !important; transform:scale(.5); transform-origin:top left; }
      html, body, .stApp, .stApp * { font-family:"Times New Roman", Times, serif !important; box-sizing:border-box; }
      html, body, .stApp { background:#F5F7FA; overflow-x:hidden; }
      html, body { max-width:100%; }
      .block-container { width:calc(100% - 4rem); max-width:100%; padding-top:2.5rem; padding-bottom:4rem; overflow-x:hidden; }
      [data-testid="stHorizontalBlock"] { width:100%; max-width:100%; gap:16px; }
      [data-testid="stColumn"] { min-width:0; }
      h1, h2, h3 { color:#0B2E6F; letter-spacing:0; font-family:"Times New Roman", Times, serif !important; }
      h1 { font-size:3rem !important; font-weight:800 !important; }
      h2 { font-size:2.35rem !important; font-weight:800 !important; margin-top:1.7rem !important; }
      h3 { font-size:2.35rem !important; font-weight:800 !important; margin-top:2.2rem !important; margin-bottom:1rem !important; line-height:1.2 !important; }
      p, .stCaption { color:#334155; font-size:1.3rem !important; line-height:1.5 !important; }
      .kpi-spacer { height:1.2rem; }
      [data-testid="stMetric"] { display:grid; grid-template-rows:auto auto auto; align-content:start; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:22px 20px; box-shadow:0 2px 8px rgba(15,23,42,.04); min-height:172px; }
      [data-testid="stMetricLabel"] { margin:0 0 .72rem !important; }
      [data-testid="stMetricLabel"] p { color:#334155 !important; font-size:1.9rem !important; font-weight:700 !important; line-height:1.15 !important; margin:0 !important; }
      [data-testid="stMetricValue"] { color:#0B2E6F !important; font-size:2.2rem !important; font-weight:800 !important; line-height:1.08 !important; margin:0 0 .82rem !important; }
      [data-testid="stMetricDelta"] { color:#2563EB !important; font-size:1.25rem !important; font-weight:800 !important; line-height:1.15 !important; margin:0 !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"] { grid-template-rows:auto 1fr; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(4) [data-testid="stMetricValue"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(5) [data-testid="stMetricValue"] { align-self:center; margin:0 !important; }
      [data-testid="stMetricDelta"] svg { fill:#2563EB !important; color:#2563EB !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricDelta"] { color:#FF3B30 !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) [data-testid="stMetricDelta"] svg { fill:#FF3B30 !important; color:#FF3B30 !important; }
      .brief { background:#FFFFFF; border:1px solid #E2E8F0; border-left:7px solid #0B2E6F; padding:24px 28px; border-radius:10px; box-shadow:0 3px 12px rgba(15,23,42,.05); color:#334155; }
      .brand-row { display:flex; align-items:center; justify-content:space-between; gap:24px; width:100%; margin:0 0 1.15rem; }
      .brand-logo { display:block; width:220px; height:auto; flex:0 0 auto; }
      .creator-credit { text-align:right; color:#334155; font-size:1.6rem; font-weight:700; margin:0; white-space:nowrap; }
      .creator-credit a { color:#0B2E6F; font-weight:800; text-decoration:underline; text-underline-offset:3px; }
      .creator-credit a:hover { color:#2563EB; }
      .brief strong { display:block; color:#0B2E6F; font-size:2.05rem; font-weight:800; line-height:1.15; margin-bottom:.7rem; }
      .summary-text { display:block; font-size:1.75rem; font-weight:700; line-height:1.45; }
      .summary-line { display:block; }
      .summary-line + .summary-line { margin-top:.28rem; }
      .attention-heading { display:flex; align-items:center; flex-wrap:wrap; gap:28px; margin:1.7rem 0 .8rem; }
      .attention-heading h2 { margin:0 !important; }
      .alert-legend { display:flex; align-items:center; gap:20px; color:#465269; font-size:1.25rem; font-weight:700; }
      .legend-item { display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }
      .legend-dot { width:13px; height:13px; border-radius:50%; display:inline-block; }
      .dot-red { background:#FF3B30; } .dot-yellow { background:#F4B400; } .dot-green { background:#2563EB; }
      .alert-group { display:grid; grid-template-columns:325px 1fr; column-gap:10px; background:transparent; border:0; border-left:8px solid var(--alert-color); border-radius:11px 0 0 11px; margin-bottom:12px; overflow:visible; }
      .alert-group-red { --alert-color:#FF3B30; --alert-title:#C62828; }
      .alert-group-yellow { --alert-color:#E5B400; --alert-title:#D99F00; }
      .alert-group-green { --alert-color:#2563EB; --alert-title:#2563EB; }
      .alert-business { display:flex; align-items:center; min-height:96px; padding:14px 22px; color:#001E79; background:#F1F4FA; border:1px solid #DCE2F3; border-left:0; border-radius:0 10px 10px 0; text-align:left; font-size:1.7rem; font-weight:800; line-height:1.25; }
      .alert-list { display:flex; flex-direction:column; min-width:0; }
      .alert { display:flex; flex-direction:column; align-items:stretch; justify-content:center; gap:10px; flex:1; border:0; padding:0; margin:0; font-size:1.55rem; line-height:1.35; }
      .alert-insight { display:flex; align-items:center; flex:1 1 96px; min-height:96px; color:#334155; background:#FFFFFF; border:1px solid #DCE2F3; border-radius:10px; padding:14px 18px; font-weight:400; }
      .alert-insight-copy { display:inline; }
      .alert-insight-title { color:var(--alert-title); font-weight:800; }
      .alert-insight strong { color:inherit; font-weight:800; }
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
      .ratio-table th:nth-child(1) { width:18%; } .ratio-table th:nth-child(2) { width:13%; }
      .ratio-table th:nth-child(3) { width:14%; } .ratio-table th:nth-child(4) { width:55%; }
      .ratio-table td { padding:16px; border-bottom:1px solid #DCE2F3; text-align:center; color:#26384D; }
      .ratio-table td:first-child { text-align:center; vertical-align:middle; color:#001E79; font-weight:800; }
      .ratio-table .other-metrics-cell { padding:0; text-align:left; }
      .other-metric-line { display:grid; grid-template-columns:26% 56% 18%; align-items:center; min-height:56px; border-bottom:1px solid #DCE2F3; }
      .other-metric-line:last-child { border-bottom:0; }
      .other-name, .other-formula, .other-value { padding:12px 14px; }
      .other-name { color:#001E79; font-weight:800; }
      .other-formula { border-left:1px solid #DCE2F3; border-right:1px solid #DCE2F3; }
      .other-value { color:#001E79; text-align:center; }
      .ratio-table tr:nth-child(even) { background:#F5F7FC; }
      .disclaimer { color:#687386; font-size:1.05rem; padding-top:18px; }
      [data-testid="stSidebar"] { min-width:500px; max-width:500px; }
      [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] { transform:translateZ(0); backface-visibility:hidden; }
      [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a span { font-size:2rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebarNav"] li { margin-bottom:.7rem !important; }
      [data-testid="stSidebarNav"] a { min-height:3.8rem !important; padding:.55rem .8rem !important; border-radius:10px !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { min-height:3.8rem !important; height:auto !important; padding:.65rem .8rem !important; align-items:flex-start !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p { font-size:2rem !important; font-weight:800 !important; line-height:1.15 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; overflow-wrap:anywhere !important; }
      [data-testid="stSidebar"] h3 { font-size:2rem !important; font-weight:800 !important; margin-top:0 !important; margin-bottom:1.15rem !important; }
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-left:.8rem !important; }
      [data-testid="stSidebar"] label p { font-size:1.4rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] * { font-size:1.65rem !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div { min-height:4rem !important; align-items:center !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] input { line-height:2rem !important; }
      /* Select menus are rendered in an unscaled portal outside .stApp.
         Scale the portal once, instead of competing with BaseWeb's computed widths. */
      [data-baseweb="popover"]:has([role="option"]) {
        zoom:.5 !important;
        margin-left:-42px !important;
      }
      [data-baseweb="popover"] [role="option"],
      [data-baseweb="popover"] [role="option"] * {
        font-family:"Times New Roman", Times, serif !important;
      }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size:1.4rem !important; }
      [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button,
      [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button p { font-size:1.5rem !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] span { font-size:1.35rem !important; white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important; color:#001E79 !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] { display:flex !important; width:calc(100% - 3.2rem) !important; max-width:none !important; min-height:2.75rem !important; height:auto !important; justify-content:space-between !important; margin:.18rem 0 !important; background:#E9EEF8 !important; border:1px solid #C9D4E8 !important; }
      [data-testid="stSidebar"] [data-baseweb="tag"] svg { color:#001E79 !important; fill:#001E79 !important; }
      [data-testid="stSidebar"] [data-baseweb="select"] > div:has([data-baseweb="tag"]) { min-height:10.5rem !important; align-content:flex-start !important; }
      [data-testid="stSidebar"] .stCaption { font-size:1.15rem !important; }
      .sidebar-divider { border-top:1px solid #CBD5E1; margin:1.1rem 0 1.35rem; }
      .nav-divider { border-top:1px solid #CBD5E1; margin:.15rem 0 -1rem; }
      .overview-toc { margin-top:.2rem; padding:0 .8rem .5rem; }
      .overview-toc-title { color:#0B2E6F; font-size:2rem; font-weight:800; margin:0 0 .8rem; }
      .overview-toc a { display:block; color:#26384D !important; font-size:1.45rem; font-weight:700; line-height:1.3; padding:.42rem .35rem; text-decoration:none !important; border-radius:6px; }
      .overview-toc a:hover { color:#001E79 !important; background:#E9EEF8; }
      .section-anchor { position:relative; top:-75px; visibility:hidden; }
      /* The page is intentionally shown at 50%. Plotly figures are rendered at
         a fixed logical width in Python so their visible width still fills each card. */
      [data-testid="stPlotlyChart"] { width:100% !important; max-width:100% !important; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0; margin-bottom:16px; box-shadow:0 2px 8px rgba(15,23,42,.035); overflow:hidden !important; }
      [data-testid="stPlotlyChart"] > div { width:100% !important; max-width:100% !important; overflow:hidden !important; }
      [data-testid="stPlotlyChart"] .scatterlayer .point,
      [data-testid="stPlotlyChart"] .barlayer .point,
      [data-testid="stPlotlyChart"] .waterfalllayer .point,
      [data-testid="stPlotlyChart"] .hoverlayer { cursor:pointer !important; }
      [data-testid="stExpander"] { background:#FFFFFF; border:0 !important; border-top:1px solid #E2E8F0 !important; border-radius:0 !important; overflow:visible !important; margin-top:1.35rem !important; margin-bottom:1.35rem !important; }
      [data-testid="stExpander"] details, [data-testid="stExpander"] summary { overflow:visible !important; }
      [data-testid="stExpander"] summary { padding-top:.75rem !important; padding-bottom:.75rem !important; }
      [data-testid="stExpander"] summary p { font-size:2.35rem !important; font-weight:800 !important; color:#0B2E6F !important; line-height:1.2 !important; }
      @media (max-width:1100px) { .alert-group { grid-template-columns:250px 1fr; } }
      @media (max-width:700px) {
        .brand-row { align-items:flex-start; flex-direction:column; gap:12px; }
        .brand-logo { width:190px; }
        .creator-credit { font-size:1.4rem; white-space:normal; text-align:left; }
      }
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
    st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Review Controls")
    time_mode = st.segmented_control("Time view", ["Month", "Quarter", "Year", "Custom Range"], default="Month", key="time_view")
    if time_mode == "Month":
        selected_end = pd.Timestamp(st.selectbox("Reporting month", list(reversed(period_options)), format_func=lambda value: value.strftime("%B %Y"), key="reporting_month"))
        selected_start = selected_end
        selection_label = selected_end.strftime("%B %Y")
        title_prefix = "Monthly"
    elif time_mode == "Quarter":
        quarter_options = sorted({period.to_period("Q") for period in period_options}, reverse=True)
        selected_quarter = st.selectbox("Reporting quarter", quarter_options, format_func=lambda value: f"Q{value.quarter} {value.year}", key="reporting_quarter")
        selected_start = pd.Timestamp(selected_quarter.start_time)
        selected_end = pd.Timestamp(selected_quarter.end_time).normalize()
        selection_label = f"Q{selected_quarter.quarter} {selected_quarter.year}"
        title_prefix = "Quarterly"
    elif time_mode == "Year":
        year_options = sorted({period.year for period in period_options}, reverse=True)
        selected_year = int(st.selectbox("Reporting year", year_options, key="reporting_year"))
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
            key="reporting_range",
        )
        if isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 2:
            selected_start, selected_end = map(pd.Timestamp, chosen_range)
        else:
            selected_start = selected_end = pd.Timestamp(chosen_range)
        selection_label = f"{selected_start:%b %Y} – {selected_end:%b %Y}"
        title_prefix = "Custom Period"
    selected_units = st.multiselect("Business units", unit_options, default=unit_options, key="business_units")
    st.markdown(
        """
        <div class="sidebar-divider"></div>
        <nav class="overview-toc" aria-label="Dashboard contents">
          <div class="overview-toc-title">Contents</div>
          <a href="#executive-summary">1. AI Executive Summary</a>
          <a href="#attention">2. Items Requiring Attention</a>
          <a href="#scorecard">3. Business Unit Scorecards</a>
          <a href="#business-model-metrics">4. Defined Metrics</a>
          <a href="#total-trend">5. Total Trends</a>
          <a href="#key-business-metrics">6. Key Business Metrics</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )

if not selected_units:
    st.warning("Select at least one business unit.")
    st.stop()

period_kpis = kpis.loc[kpis["period"].between(selected_start, selected_end) & kpis["business_unit"].isin(selected_units)]
current = aggregate_kpis(period_kpis)
total_revenue = current["revenue_actual"].sum()
total_budget = current["revenue_budget"].sum()
total_profit = current["adjusted_profit_actual"].sum()
total_expense = current["operating_expense_actual"].sum()
total_expense_budget = current["operating_expense_budget"].sum()
overall_performance = {
    "business_unit_count": len(selected_units),
    "revenue_actual": float(total_revenue),
    "revenue_variance": float(total_revenue / total_budget - 1),
    "operating_expense_actual": float(total_expense),
    "expense_variance": float(total_expense / total_expense_budget - 1),
}
current_alerts = alerts.loc[alerts["period"].between(selected_start, selected_end) & alerts["business_unit"].isin(selected_units)]
raw_period_evidence = load_raw_source_evidence(selected_start, selected_end, tuple(selected_units))
period_calculation_evidence = detail.loc[
    detail["period"].between(selected_start, selected_end) & detail["business_unit"].isin(selected_units)
]

st.markdown("<div id='executive-summary' class='section-anchor'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='brand-row'>"
    f"<img class='brand-logo' src='{LOGO_DATA_URI}' alt='U.S. Bank logo'>"
    "<div class='creator-credit'>Ziqi (Ankit) Cao &nbsp;·&nbsp; "
    "<a href='https://www.linkedin.com/in/ziqi-ankit-cao' target='_blank' rel='noopener noreferrer'>LinkedIn</a></div>"
    "</div>",
    unsafe_allow_html=True,
)
st.title(f"Automated Three-Business Performance Dashboard – {selection_label}")

_, attention_items = generate_executive_summary(period_kpis, current_alerts, selection_label)
department_summary_fallbacks = {
    str(row["business_unit"]): department_performance_sentence(row)
    for _, row in current.iterrows()
}
summary_fallback = " ".join([
    overall_performance_sentence(overall_performance),
    *(department_summary_fallbacks[unit] for unit in selected_units),
])
severity_to_status = {"Red": "Critical", "Yellow": "Caution", "Green": "Positive"}
severity_rank = {"Red": 0, "Yellow": 1, "Green": 2}
fallback_reviews = []
for business_unit in selected_units:
    unit_alerts = current_alerts.loc[current_alerts["business_unit"].eq(business_unit)].copy()
    if unit_alerts.empty:
        unit_metrics = current.loc[current["business_unit"].eq(business_unit)].iloc[0]
        fallback_reviews.append({
            "business_unit": business_unit,
            "status": "Positive",
            "insights": [{
                "title": "Revenue on track",
                "text": (
                    f"Revenue reached ${unit_metrics['revenue_actual']:.1f}M, "
                    f"{unit_metrics['revenue_vs_budget']:+.1%} versus target."
                ),
            }],
        })
    else:
        unit_alerts["rank"] = unit_alerts["severity"].map(severity_rank)
        alert = unit_alerts.sort_values(["rank", "period"], ascending=[True, False]).iloc[0]
        message = str(alert["message"])
        if message.startswith(f"{business_unit} "):
            message = message[len(business_unit) + 1 :]
            message = message[:1].upper() + message[1:]
        fallback_reviews.append({
            "business_unit": business_unit,
            "status": severity_to_status.get(str(alert["severity"]), "Caution"),
            "insights": [{"title": str(alert["rule"]), "text": message}],
        })

period_review = generate_ai_period_review({
    "version": 6,
    "reporting_period": selection_label,
    "selected_business_units": list(selected_units),
    "overall_performance": overall_performance,
    "department_summary_fallbacks": department_summary_fallbacks,
    "raw_source_records": raw_period_evidence,
    "clean_monthly_kpis": dataframe_records(period_kpis),
    "calculated_metric_records": dataframe_records(period_calculation_evidence),
    "deterministic_rule_alerts": dataframe_records(current_alerts),
}, summary_fallback, fallback_reviews)
summary = period_review["executive_summary"]
summary_lines = [line.strip() for line in re.split(r"(?<=[.!?])\s+", summary) if line.strip()]
summary_html = "".join(f"<span class='summary-line'>{escape(line)}</span>" for line in summary_lines)
st.markdown(
    f"<div class='brief'><strong>AI Executive Summary:</strong><span class='summary-text'>{summary_html}</span></div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='kpi-spacer'></div>", unsafe_allow_html=True)

cards = st.columns(4, gap="medium")
cards[0].metric("Revenues", fmt_money(total_revenue), f"{overall_performance['revenue_variance']:+.1%} vs. target")
cards[1].metric("Adjusted Profits", fmt_money(total_profit), f"{(total_profit / current['adjusted_profit_budget'].sum() - 1):+.1%} vs. target")
cards[2].metric("Operating Expenses", fmt_money(total_expense), f"{overall_performance['expense_variance']:+.1%} vs. budget", delta_color="inverse")
cards[3].metric("Targets Achieved", f"{1 + overall_performance['revenue_variance']:.1%}")

st.markdown("<div id='attention' class='section-anchor'></div>", unsafe_allow_html=True)
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
review_by_unit = {item["business_unit"]: item for item in period_review["business_reviews"]}
status_css = {"Critical": "red", "Caution": "yellow", "Positive": "green"}
for business_unit in selected_units:
    review = review_by_unit[business_unit]
    has_alert = not current_alerts.loc[current_alerts["business_unit"].eq(business_unit)].empty
    if has_alert:
        alert_items = [
            "<div class='alert-insight'>"
            "<span class='alert-insight-copy'>"
            f"<span class='alert-insight-title'>{escape(normalize_insight_label(insight['title']))}:</span> "
            f"{format_insight_text_html(normalize_llm_sentence(insight['text']))}</span></div>"
            for insight in review["insights"]
        ]
        alert_status_class = status_css[review["status"]]
    else:
        alert_items = ["<div class='alert-insight'><span class='alert-insight-title'>No alert.</span></div>"]
        alert_status_class = "green"
    alert_rows = [
        f"<div class='alert'>{''.join(alert_items)}</div>"
    ]
    st.markdown(
        f"<div class='alert-group alert-group-{alert_status_class}'><div class='alert-business'>{escape(str(business_unit))}</div>"
        f"<div class='alert-list'>{''.join(alert_rows)}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div id='scorecard' class='section-anchor'></div>", unsafe_allow_html=True)
st.subheader("Business Unit Scorecards")
render_scorecard(scorecard_rows(current))
st.markdown("<div id='business-model-metrics' class='section-anchor'></div>", unsafe_allow_html=True)
st.subheader("Defined Metrics")
render_business_model_metrics(current)

# Trend visuals always show a consistent rolling 12-month history ending at the
# selected period. All non-trend visuals below continue to use selected_start.
available_start = pd.Timestamp(kpis["period"].min())
trend_start = max(available_start, selected_end - pd.DateOffset(months=11))
trend_data = kpis.loc[kpis["period"].between(trend_start, selected_end) & kpis["business_unit"].isin(selected_units)]
selected_detail = detail.loc[detail["period"].between(selected_start, selected_end) & detail["business_unit"].isin(selected_units)]
trend_detail = detail.loc[detail["period"].between(trend_start, selected_end) & detail["business_unit"].isin(selected_units)]
raw_trend_evidence = load_raw_source_evidence(trend_start, selected_end, tuple(selected_units))
chart_period_label = selection_label
latest_period = trend_data["period"].max()
latest_month = trend_data.loc[trend_data["period"] == latest_period].copy()
efficiency_latest = latest_month.sort_values("cost_to_income_ratio_actual", ascending=False).iloc[0]
efficiency_lowest = latest_month.sort_values("cost_to_income_ratio_actual", ascending=True).iloc[0]
margin_period = current.sort_values("profit_margin_actual", ascending=False).iloc[0]
margin_target = float(margin_period["profit_margin_budget"])

revenue_latest = latest_month.sort_values("revenue_actual", ascending=False).iloc[0]
revenue_history = trend_data.loc[trend_data["business_unit"] == revenue_latest["business_unit"]].sort_values("period")
prior_revenue = float(revenue_history.iloc[-2]["revenue_actual"]) if len(revenue_history) > 1 else float(revenue_latest["revenue_actual"])
revenue_summary = generate_chart_summary("revenue_trend", attach_evidence({
    "_version": 5,
    "unit": revenue_latest["business_unit"], "latest": round(float(revenue_latest["revenue_actual"]), 1),
    "change": round((float(revenue_latest["revenue_actual"]) / prior_revenue - 1) * 100, 1),
    "period": latest_period.strftime("%B %Y"),
}, raw_trend_evidence, trend_detail.loc[trend_detail["metric_type"] == "Revenue"]))

efficiency_summary = generate_chart_summary("cost_to_income", attach_evidence({
    "_version": 5,
    "highest_unit": efficiency_latest["business_unit"],
    "highest": round(float(efficiency_latest["cost_to_income_ratio_actual"] * 100), 1),
    "lowest_unit": efficiency_lowest["business_unit"],
    "lowest": round(float(efficiency_lowest["cost_to_income_ratio_actual"] * 100), 1),
    "period": latest_period.strftime("%B %Y"),
}, raw_trend_evidence, trend_detail))
margin_summary = generate_chart_summary("profit_margin", attach_evidence({
    "_version": 5,
    "highest_unit": margin_period["business_unit"],
    "highest": round(float(margin_period["profit_margin_actual"] * 100), 1),
    "variance": round((float(margin_period["profit_margin_actual"]) - margin_target) * 100, 1),
    "period": selection_label,
}, raw_period_evidence, period_calculation_evidence))

variance_frame = selected_detail.loc[selected_detail["metric_type"] == "Revenue"].groupby("management_category", as_index=False)[["actual", "budget"]].sum()
variance_frame["variance"] = variance_frame["actual"] - variance_frame["budget"]
largest_driver = variance_frame.loc[variance_frame["variance"].abs().idxmax(), "management_category"]
variance_summary = generate_chart_summary("revenue_variance", attach_evidence({
    "_version": 5,
    "variance": round(float(variance_frame["variance"].sum()), 1), "driver": str(largest_driver),
}, raw_period_evidence, selected_detail.loc[selected_detail["metric_type"] == "Revenue"]))
expense_frame = current.copy()
expense_frame["variance"] = expense_frame["operating_expense_actual"] - expense_frame["operating_expense_budget"]
largest_expense = expense_frame.sort_values("variance", ascending=False).iloc[0]
expense_summary = generate_chart_summary("expense_variance", attach_evidence({
    "_version": 5,
    "unit": largest_expense["business_unit"], "variance": round(float(max(largest_expense["variance"], 0)), 1),
}, raw_period_evidence, selected_detail.loc[selected_detail["metric_type"] == "Expense"]))

st.markdown("<div id='total-trend' class='section-anchor'></div>", unsafe_allow_html=True)
with st.expander("Total Trends", expanded=True):
    revenue_trend_view, efficiency_trend_view = st.columns(2)
    with revenue_trend_view:
        render_plotly_chart(add_chart_summary(trend_chart(trend_data, selected_units), revenue_summary), 600)
    with efficiency_trend_view:
        render_plotly_chart(add_chart_summary(cost_income_trend_chart(trend_data, selected_units), efficiency_summary), 600)

    variance_view, expense_view = st.columns(2)
    with variance_view:
        render_plotly_chart(add_chart_summary(waterfall_chart(selected_detail, chart_period_label), variance_summary), 560)
    with expense_view:
        render_plotly_chart(add_chart_summary(expense_comparison_chart(current, chart_period_label), expense_summary), 560)

st.markdown("<div id='key-business-metrics' class='section-anchor'></div>", unsafe_allow_html=True)
with st.expander("Key Business Metrics", expanded=True):
    specialized_figures = [(
        margin_summary,
        profit_margin_comparison_chart(current, selection_label).update_layout(height=580),
    )]
    if "Commercial Real Estate" in selected_units:
        cre = kpis.loc[(kpis["business_unit"] == "Commercial Real Estate") & kpis["period"].between(trend_start, selected_end)].sort_values("period")
        if not cre.empty:
            values = cre["npl_ratio_actual"].dropna()
            summary_text = generate_chart_summary("npl_ratio", attach_evidence({
                "_version": 5,
                "business_unit": "Commercial Real Estate", "latest": round(float(values.iloc[-1] * 100), 1),
                "prior": round(float(values.iloc[-2] * 100), 1) if len(values) > 1 else round(float(values.iloc[-1] * 100), 1),
                "months": int(len(values)),
            }, [row for row in raw_trend_evidence if row.get("Business Unit") == "Commercial Real Estate"],
                trend_detail.loc[trend_detail["business_unit"] == "Commercial Real Estate"]))
            specialized_figures.append((summary_text, single_ratio_trend_chart(
                cre, "npl_ratio_actual", "NPL Ratios (Commercial Real Estate)",
                "= NPL Proxy ÷ CRE Loan Balance", threshold=0.015,
            ).update_layout(height=580)))
    if "Commercial Banking" in selected_units:
        banking = kpis.loc[(kpis["business_unit"] == "Commercial Banking") & kpis["period"].between(trend_start, selected_end)].sort_values("period")
        if not banking.empty:
            values = banking["loan_to_deposit_ratio_actual"].dropna()
            summary_text = generate_chart_summary("loan_to_deposit", attach_evidence({
                "_version": 5,
                "business_unit": "Commercial Banking", "latest": round(float(values.iloc[-1] * 100), 1),
                "prior": round(float(values.iloc[-2] * 100), 1) if len(values) > 1 else round(float(values.iloc[-1] * 100), 1),
                "months": int(len(values)), "period": banking.iloc[-1]["period"].strftime("%B %Y"),
            }, [row for row in raw_trend_evidence if row.get("Business Unit") == "Commercial Banking"],
                trend_detail.loc[trend_detail["business_unit"] == "Commercial Banking"]))
            specialized_figures.append((summary_text, single_ratio_trend_chart(
                banking, "loan_to_deposit_ratio_actual", "Loan-to-Deposit Ratios (Commercial Banking)",
                "= Loan Balance ÷ Deposit Balance",
            ).update_layout(height=580)))
    if "Capital Markets" in selected_units:
        markets = kpis.loc[
            (kpis["business_unit"] == "Capital Markets")
            & kpis["period"].between(selected_start, selected_end)
        ].sort_values("period")
        if not markets.empty:
            mix_columns = {
                "Advisory": "advisory_mix_actual", "Underwriting": "underwriting_mix_actual",
                "Trading": "trading_mix_actual", "Structuring": "structuring_mix_actual", "Syndication": "syndication_mix_actual",
            }
            latest_mix = markets.iloc[-1]
            largest_component = max(mix_columns, key=lambda label: latest_mix[mix_columns[label]])
            summary_text = generate_chart_summary("fee_revenue_mix", attach_evidence({
                "_version": 5,
                "business_unit": "Capital Markets", "largest_component": largest_component,
                "largest_share": round(float(latest_mix[mix_columns[largest_component]] * 100), 1), "months": int(len(markets)),
                "period": latest_mix["period"].strftime("%B %Y"),
            }, [row for row in raw_period_evidence if row.get("Business Unit") == "Capital Markets"],
                period_calculation_evidence.loc[(period_calculation_evidence["business_unit"] == "Capital Markets") & (period_calculation_evidence["metric_type"] == "Revenue")]))
            specialized_figures.append((summary_text, fee_revenue_mix_chart(markets).update_layout(height=580, margin=dict(b=112))))
    if specialized_figures:
        left_column, right_column = st.columns(2)
        midpoint = (len(specialized_figures) + 1) // 2
        for column, figures in ((left_column, specialized_figures[:midpoint]), (right_column, specialized_figures[midpoint:])):
            with column:
                for summary_text, figure in figures:
                    render_plotly_chart(add_chart_summary(figure, summary_text), int(figure.layout.height or 580))
    else:
        st.info("Select a business unit to view its specialized metric.")
