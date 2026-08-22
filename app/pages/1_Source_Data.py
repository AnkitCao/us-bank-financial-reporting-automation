"""Display the three complete simulated source datasets."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean_data import profile_raw_sources

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "us-bank-logo.svg"


@st.cache_data
def load_source_tables() -> list[tuple[str, str, pd.DataFrame]]:
    """Load every row and column from the three raw source workbooks."""
    commercial_banking = pd.read_excel(RAW_DIR / "commercial_banking_monthly.xlsx", sheet_name="Monthly Detail")
    commercial_real_estate = pd.read_excel(RAW_DIR / "commercial_real_estate_monthly.xlsx", sheet_name="CRE Monthly")

    capital_path = RAW_DIR / "capital_markets_monthly.xlsx"
    capital_workbook = pd.ExcelFile(capital_path)
    capital_markets = pd.concat(
        [
            pd.read_excel(capital_path, sheet_name=sheet).assign(**{"Source Metric": sheet})
            for sheet in capital_workbook.sheet_names
        ],
        ignore_index=True,
    )
    capital_markets.insert(0, "Source Metric", capital_markets.pop("Source Metric"))

    return [
        ("Commercial Banking", "commercial_banking_monthly.xlsx", commercial_banking),
        ("Commercial Real Estate", "commercial_real_estate_monthly.xlsx", commercial_real_estate),
        ("Capital Markets", "capital_markets_monthly.xlsx", capital_markets),
    ]


SOURCE_PROFILES = {
    "Commercial Banking": {
        "structure": "Long table",
        "grain": "One row for each month",
        "coverage": "Apr 2025 - Jun 2026",
        "note": "Text dates, inconsistent labels, missing values, a duplicate record and an extreme value require cleaning.",
        "method": "Map approved column aliases, trim labels, parse month-end dates, remove exact duplicate rows, flag same-group median/MAD outliers, and interpolate only within the same metric and scenario.",
        "fields": [
            ("Key", "Reporting Date", "Month covered by the record."),
            ("Key", "Plan Type", "Comparison case: Actual, Budget, Forecast or Prior Year."),
            ("Revenue", "Loan Interest", "Interest income from commercial loans."),
            ("Revenue", "Treasury Fee", "Treasury-service fees."),
            ("Revenue", "Deposit Fee", "Deposit-service fees."),
            ("Revenue", "Merchant Fee", "Merchant-payment fees."),
            ("Revenue", "FX Fee", "Foreign-exchange fees."),
            ("Cost", "Operating Expense", "Operating costs deducted from revenue."),
            ("Balance", "Loan Balance", "Commercial loan portfolio size."),
            ("Balance", "Deposit Balance", "Commercial deposit funding size."),
        ],
        "revenue_metrics": "Loan Interest; Treasury Fee; Deposit Fee; Merchant Fee; FX Fee",
        "cost_metrics": "Operating Expense",
        "operating_profit": "Total Revenue − Operating Expense",
        "guardrails": "Loan Balance and Deposit Balance show operating scale; they are monitored but excluded from revenue and profit sums.",
        "guardrail_metrics": "Loan Balance; Deposit Balance",
        "issue_evidence": [
            ("Missing Values", "Value_MM (financial value)", "8 cells", "Fill by month within the same measure and plan type."),
            ("Duplicate Rows", "All columns; key: Date + Measure + Plan", "12 rows", "Remove identical rows; reject conflicting keys."),
            ("Inconsistent Labels", "Plan Type and Measure Name", "9 labels", "Trim spaces and map labels to approved names."),
            ("Extreme Values", "Value_MM (financial value)", "5 values", "Flag outside the same-group robust range, then interpolate."),
            ("Column Aliases", "All source headers", "5 headers", "Rename source headers to standard dashboard fields."),
        ],
    },
    "Commercial Real Estate": {
        "structure": "Wide table",
        "grain": "One row for each month and reporting case",
        "coverage": "Apr 2025 - Jun 2026",
        "note": "Wide structure, missing values, a duplicate row, an extreme cost value and inconsistent case labels require cleaning.",
        "method": "Map approved aliases, standardize cases, unpivot metric columns, remove exact duplicates, flag same-group median/MAD outliers, and interpolate only within the same metric and scenario.",
        "fields": [
            ("Key", "Report Month", "Month covered by the record."),
            ("Key", "Plan Case", "Comparison case: Actual, Budget, Forecast or Prior Year."),
            ("Revenue", "Net Interest Income", "Interest income from CRE loans."),
            ("Revenue", "Origination Fees", "New-loan origination fees."),
            ("Revenue", "Prepayment Fees", "Early-repayment fees."),
            ("Revenue", "Servicing Fee", "Loan-servicing fees."),
            ("Revenue", "Other Fee", "Other noninterest fees."),
            ("Cost", "Operating Costs", "Operating costs deducted from revenue."),
            ("Credit", "Credit Loss Provision", "Expected credit-loss cost."),
            ("Credit", "NPL Proxy ($MM)", "Asset-quality risk indicator."),
            ("Balance", "CRE Loan Bal ($MM)", "CRE portfolio size."),
        ],
        "revenue_metrics": "Net Interest Income; Origination Fees; Prepayment Fees; Servicing Fee; Other Fee",
        "cost_metrics": "Operating Costs; Credit Loss Provision shown separately as credit cost",
        "operating_profit": "Total Revenue − Operating Costs − Credit Loss Provision",
        "guardrails": "CRE Loan Balance and NPL Proxy explain scale and credit quality, but are balance measures—not revenue.",
        "guardrail_metrics": "CRE Loan Balance; NPL Proxy",
        "issue_evidence": [
            ("Missing Values", "Prepayment Fees and Net Interest Income", "2 cells", "Fill by month within the same metric and plan case."),
            ("Duplicate Rows", "All columns; key: Month + Plan Case", "3 rows", "Remove identical rows; reject conflicting keys."),
            ("Inconsistent Labels", "Plan Case", "2 labels", "Trim spaces and map labels to approved names."),
            ("Extreme Values", "Operating Costs and Origination Fees", "2 values", "Flag outside the same-group robust range, then interpolate."),
            ("Column Aliases", "Source headers", "6 headers", "Rename source headers to standard dashboard fields."),
        ],
    },
    "Capital Markets": {
        "structure": "Cross tab wide workbook",
        "grain": "One row for each source metric and scenario",
        "coverage": "Apr 2025 - Jun 2026",
        "note": "A missing value, duplicate scenario, extreme value, inconsistent label and month columns require cleaning.",
        "method": "Combine six sheets, standardize scenarios and month headers, unpivot months, remove exact duplicates, flag same-group median/MAD outliers, and interpolate only within the same metric and scenario.",
        "fields": [
            ("Key", "Plan Scenario", "Comparison case: Actual, Budget, Forecast or Prior Year."),
            ("Revenue", "Advisory Fee", "Advisory transaction fees."),
            ("Revenue", "Underwriting Revenue", "Securities-underwriting income."),
            ("Revenue", "Trading Revenue", "Client and market trading income."),
            ("Revenue", "Structuring Fee", "Financing-structure fees."),
            ("Revenue", "Syndication Fee", "Financing-syndication fees."),
            ("Cost", "Operating Expense", "Operating costs deducted from revenue."),
        ],
        "revenue_metrics": "Advisory Fee; Underwriting Revenue; Trading Revenue; Structuring Fee; Syndication Fee",
        "cost_metrics": "Operating Expense",
        "operating_profit": "Total Revenue − Operating Expense",
        "guardrails": "Fee and trading worksheets map only to Revenue; the expense worksheet maps only to Expense.",
        "guardrail_metrics": "Fee and Trading Revenue; Operating Expense",
        "issue_evidence": [
            ("Missing Values", "Advisory Fee (May-25); Underwriting Revenue (Oct-25)", "2 cells", "Fill by month within the same metric and scenario."),
            ("Duplicate Rows", "Key: Metric + Scenario + Month", "2 rows", "Remove identical rows; reject conflicting keys."),
            ("Inconsistent Labels", "Plan Scenario", "2 labels", "Trim spaces and map labels to approved names."),
            ("Extreme Values", "Trading Revenue (Sep-25); Operating Expense (Dec-25)", "2 values", "Flag outside the same-group robust range, then interpolate."),
            ("Column Aliases", "Plan Scenario and month headers", "2 headers", "Rename standard fields, then convert months into rows."),
        ],
    },
}


def _field_range_and_stats(table: pd.DataFrame, field: str) -> tuple[str, str]:
    """Describe raw content and calculate raw numeric statistics for a source field."""
    if field in {"Plan Type", "Plan Case", "Plan Scenario"}:
        return "-", "N/A"
    if field in table.columns:
        series = table[field]
    else:
        matching_columns = [column for column in table.columns if str(column).strip() == field.strip()]
        if matching_columns:
            series = table[matching_columns[0]]
        elif "Measure Name" in table.columns and table["Measure Name"].astype(str).str.strip().eq(field).any():
            series = table.loc[table["Measure Name"].astype(str).str.strip().eq(field), "Value_MM"]
        elif "Source Metric" in table.columns and table["Source Metric"].astype(str).str.strip().eq(field).any():
            metric_rows = table.loc[table["Source Metric"].astype(str).str.strip().eq(field)]
            series = metric_rows.drop(columns=["Source Metric", "Plan Scenario"], errors="ignore").stack(dropna=False)
        else:
            return "N/A", "N/A"
    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce")
    if len(non_null) and numeric.notna().mean() >= 0.8:
        return (
            f"({numeric.min():,.3f}, {numeric.max():,.3f})",
            f"{numeric.mean():,.3f}",
        )
    parsed_dates = pd.to_datetime(non_null, errors="coerce")
    if len(non_null) and parsed_dates.notna().mean() >= 0.8:
        return f"{parsed_dates.min():%b %Y} - {parsed_dates.max():%b %Y}", "N/A"
    values = [str(value) for value in non_null.astype(str).unique()]
    preview = ", ".join(values[:6]) + (", …" if len(values) > 6 else "")
    return preview, "N/A"


def source_profile_html(business_unit: str, table: pd.DataFrame, quality: pd.Series) -> str:
    """Build the four-part interview narrative for one source."""
    profile = SOURCE_PROFILES[business_unit]
    category_order = list(dict.fromkeys(category for category, _, _ in profile["fields"]))
    ordered_fields = [item for category in category_order for item in profile["fields"] if item[0] == category]
    field_rows = []
    category_counts = pd.Series([category for category, _, _ in ordered_fields]).value_counts().to_dict()
    rendered_categories: set[str] = set()
    for category, field, description in ordered_fields:
        content_range, statistics = _field_range_and_stats(table, field)
        category_cell = ""
        if category not in rendered_categories:
            category_cell = f"<td rowspan='{category_counts[category]}' class='field-category'><strong>{escape(category)}</strong></td>"
            rendered_categories.add(category)
        field_rows.append(
            f"<tr>{category_cell}<td class='field-metric'><strong>{escape(field)}</strong></td><td>{escape(description)}</td>"
            f"<td>{escape(content_range)}</td><td>{escape(statistics)}</td></tr>"
        )
    issue_rows = "".join(
        f"<tr><td><strong>{escape(issue)}</strong></td><td>{escape(columns)}</td>"
        f"<td>{escape(count)}</td><td>{escape(resolution)}</td></tr>"
        for issue, columns, count, resolution in profile["issue_evidence"]
    )
    def metric_lines(value: str) -> str:
        return "<br>".join(escape(item.strip()) for item in value.split(";"))

    revenue_fields = [item.strip() for item in profile["revenue_metrics"].split(";")]
    revenue_formula = "= " + " + ".join(revenue_fields)
    if business_unit == "Commercial Real Estate":
        cost_defined = "Operating Cost<br>Credit Cost"
        cost_formula = "= Operating Costs + Credit Loss Provision"
        profit_formula = "= " + " + ".join(revenue_fields) + " − Operating Costs − Credit Loss Provision"
        efficiency_cost = "Operating Costs"
        specialized_metric = (
            "NPL Ratio", "= NPL Proxy ($MM) ÷ CRE Loan Bal ($MM)",
            "Nonperforming exposure as a share of CRE loans.", "Monitor CRE asset quality across periods.",
        )
    else:
        cost_defined = "Operating Cost"
        cost_formula = "= Operating Expense"
        profit_formula = "= " + " + ".join(revenue_fields) + " − Operating Expense"
        efficiency_cost = "Operating Expense"
        if business_unit == "Commercial Banking":
            specialized_metric = (
                "Loan-to-Deposit Ratio", "= Loan Balance ÷ Deposit Balance",
                "Loans funded per dollar of deposits.", "Monitor funding use and liquidity balance.",
            )
        else:
            specialized_metric = (
                "Fee Revenue Mix", "= Each revenue field ÷ Total Revenue",
                "Share of revenue from each Capital Markets activity.", "Monitor income concentration and diversification.",
            )
    metric_rows = [
        ("Total Revenue", revenue_formula, "Income from all revenue fields.", "Track trends and compare Target, Forecast and Prior Year."),
        (cost_defined, cost_formula, "Costs deducted from revenue.", "Track spending and budget variance."),
        ("Operating Profit", profit_formula, "Revenue remaining after costs.", "Track profit margin and business performance."),
        ("Cost-to-Income Ratio", f"= {efficiency_cost} ÷ Total Revenue", "Operating cost per dollar of revenue.", "Compare efficiency across businesses and periods."),
        ("Profit Margin", "= Operating Profit ÷ Total Revenue", "Profit retained per dollar of revenue.", "Compare profitability across businesses and periods."),
        specialized_metric,
        (metric_lines(profile["guardrail_metrics"]), "= 0 in Revenue and Profit", "Balance or risk measures; not income.", "Monitor scale or risk without overstating profit."),
    ]
    metric_table_rows = "".join(
        f"<tr><td>{metrics}</td><td>{escape(formula)}</td>"
        f"<td>{escape(meaning)}</td><td>{escape(used_for)}</td></tr>"
        for metrics, formula, meaning, used_for in metric_rows
    )
    section_slug = business_unit.lower().replace(" ", "-")
    return f"""
    <div id="{section_slug}-raw-data-issues" class="story-step section-anchor"><span>1</span><div><strong>Raw Data Issues</strong></div></div>
    <div class="profile-panel">
      <div class="profile-grid">
        <div class="profile-stat"><strong>{profile['structure'].title()}</strong></div>
        <div class="profile-stat"><strong>{profile['coverage'].title()}</strong></div>
        <div class="profile-stat"><strong>{profile['grain'].title()}</strong></div>
        <div class="profile-stat"><strong>{int(quality['problem_records'])} Issues Over {int(quality['rows'])} Rows ({quality['problem_rate']:.1%})</strong></div>
      </div>
      <div class="issue-title">Issues Table</div>
      <div class="issue-table-wrap"><table class="issue-table"><thead><tr><th>Issue</th><th>Columns</th><th>Count</th><th>Fix</th></tr></thead><tbody>{issue_rows}</tbody></table></div>
    </div>
    <div id="{section_slug}-important-columns" class="story-step section-anchor"><span>2</span><div><strong>Important Columns</strong></div></div>
    <div class="profile-panel compact-panel field-table-wrap"><table class="field-table"><thead><tr><th>Category</th><th>Field</th><th>Meaning</th><th>Range</th><th>Average</th></tr></thead><tbody>{''.join(field_rows)}</tbody></table></div>
    <div id="{section_slug}-defined-metrics" class="story-step section-anchor"><span>3</span><div><strong>Defined Metrics</strong></div></div>
    <div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>Metric</th><th>Calculation</th><th>Meaning</th><th>Use</th></tr></thead><tbody>{metric_table_rows}</tbody></table></div>
    """


st.set_page_config(page_title="Source Data for Three Businesses", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
      html, body, .stApp, .stApp * { font-family:"Times New Roman", Times, serif !important; }
      .stApp { background:#F7F8FC; }
      .block-container { width:calc(100% - 4rem); max-width:none; padding-top:2.5rem; padding-bottom:4rem; }
      h1, h2, h3 { color:#001E79; font-family:"Times New Roman", Times, serif !important; }
      h1 { font-size:3rem !important; }
      h2 { font-size:2.75rem !important; font-weight:800 !important; margin:1.2rem 0 .55rem !important; padding:0 !important; line-height:1.08 !important; }
      p, .stCaption { font-size:1.18rem !important; line-height:1.6 !important; }
      .simulated-notice { background:#FFF1F2; border:2px solid #CF2A36; border-left:9px solid #CF2A36; border-radius:10px; padding:22px 26px; margin:1.5rem 0 .5rem; color:#701821; font-size:1.55rem; line-height:1.6; }
      .source-meta { color:#465269; font-size:1.45rem; line-height:1.3; margin:.75rem 0 .65rem; }
      .story-step { display:flex; align-items:center; gap:14px; margin:2.1rem 0 1.45rem; color:#001E79; }
      .story-step > span { display:flex; align-items:center; justify-content:center; flex:0 0 50px; height:50px; border-radius:50%; background:#001E79; color:white; font-size:1.6rem; font-weight:800; }
      .story-step strong { display:block; font-size:1.95rem; line-height:1.2; }
      .story-step small { display:block; color:#59657A; font-size:1.05rem; margin-top:4px; }
      .profile-panel { background:white; border:1px solid #DCE2F3; border-radius:12px; padding:24px; margin:1rem 0 1.5rem; box-shadow:0 3px 14px rgba(0,30,121,.05); }
      .compact-panel { margin-top:.6rem; }
      .profile-title { color:#001E79; font-size:1.7rem; font-weight:800; margin-bottom:1rem; }
      .profile-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; overflow:hidden; background:white; }
      .profile-stat { display:flex; align-items:center; justify-content:center; min-height:92px; background:#F1F4FA; border:0; padding:14px 16px; text-align:center; }
      .profile-stat span { display:block; color:#59657A; font-size:1.05rem; font-weight:700; }
      .profile-stat strong { display:block; color:#12213A; font-size:1.65rem; line-height:1.3; }
      .quality-note { background:#FFF8E8; border-left:5px solid #B76E00; padding:14px 16px; margin:14px 0; font-size:1.15rem; line-height:1.55; }
      .cleaning-note { background:#EDF8F4; border-left:5px solid #00A878; padding:14px 16px; margin:14px 0; font-size:1.15rem; line-height:1.55; }
      .issue-title { color:#001E79; font-size:2.15rem; font-weight:800; line-height:1.15; margin:1.3rem 0 .8rem; }
      .issue-table-wrap { overflow-x:auto; border:1px solid #F2CED1; border-radius:8px; }
      .issue-table { width:100%; min-width:1250px; table-layout:fixed; border-collapse:collapse; margin:0 !important; color:#12213A; font-size:1.75rem; }
      .issue-table th { box-sizing:border-box; background:#FF3B30; color:white; border:2px solid white; padding:18px 20px; text-align:center; font-size:1.9rem; white-space:nowrap; }
      .issue-table td { box-sizing:border-box; padding:18px 20px; border-bottom:1px solid #F2CED1; text-align:left; vertical-align:middle; line-height:1.4; }
      .issue-table tbody tr:nth-child(odd) { background:#FFF7F7; }
      .issue-table tbody tr:nth-child(even) { background:#FFFDFD; }
      .issue-table tbody tr:last-child td { border-bottom:0; }
      .issue-table td:first-child { color:#12213A; font-weight:800; white-space:nowrap; }
      .issue-table td:nth-child(3) { color:#12213A; font-weight:400; text-align:center; white-space:nowrap; }
      .issue-table td:last-child { color:#12213A; font-weight:800; }
      .issue-table th:nth-child(1), .issue-table td:nth-child(1) { width:21%; }
      .issue-table th:nth-child(2), .issue-table td:nth-child(2) { width:25%; }
      .issue-table th:nth-child(3), .issue-table td:nth-child(3) { width:12%; }
      .issue-table th:nth-child(4), .issue-table td:nth-child(4) { width:42%; }
      .field-title { color:#001E79; font-size:1.35rem; font-weight:800; margin:1rem 0 .7rem; }
      .field-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; }
      .field-item { border:1px solid #DCE2F3; border-radius:8px; padding:12px 14px; }
      .field-item strong { display:block; color:#001E79; font-size:1.12rem; }
      .field-item span { display:block; color:#39465E; font-size:1.02rem; line-height:1.4; margin-top:4px; }
      .field-table-wrap { overflow-x:auto; padding:0; }
      .field-table { width:100%; min-width:1250px; table-layout:fixed; border-collapse:collapse; margin:0 !important; color:#26384D; font-size:1.85rem; }
      .field-table th { background:#001E79; color:white; border:2px solid white; padding:18px 20px; text-align:left; font-size:2rem; white-space:nowrap; }
      .field-table td { padding:18px 20px; border-bottom:1px solid #DCE2F3; text-align:left; vertical-align:middle; line-height:1.4; }
      .field-table tbody tr:nth-child(even) { background:#F5F7FC; }
      .field-table tbody tr:last-child td { border-bottom:0; }
      .field-category { color:#001E79; white-space:nowrap; vertical-align:middle !important; background:#EEF2FA; border-right:1px solid #DCE2F3; }
      .field-metric { color:#001E79; white-space:nowrap; }
      .field-table th:nth-child(4), .field-table th:nth-child(5), .field-table td:nth-last-child(2), .field-table td:last-child { text-align:center; }
      .field-table th, .field-table td { box-sizing:border-box; }
      .field-table th:nth-child(1) { width:12%; }
      .field-table th:nth-child(2) { width:21%; }
      .field-table th:nth-child(3) { width:31%; }
      .field-table th:nth-child(4) { width:22%; }
      .field-table th:nth-child(5) { width:14%; }
      .metric-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; margin:.6rem 0 2rem; }
      .metric-card { background:white; border:1px solid #DCE2F3; border-top:6px solid #001E79; border-radius:10px; padding:20px; }
      .metric-card > span { display:block; color:#59657A; font-size:1.05rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
      .metric-card > strong { display:block; color:#001E79; font-size:1.22rem; line-height:1.45; margin:.55rem 0; }
      .metric-card p { color:#39465E; font-size:1.05rem !important; line-height:1.45 !important; margin:.45rem 0 0; }
      .revenue-card { border-top-color:#2563EB; } .cost-card { border-top-color:#CF2A36; }
      .profit-card { border-top-color:#00A878; } .guardrail-card { border-top-color:#B76E00; }
      .metric-table-wrap { overflow-x:auto; background:white; border:1px solid #DCE2F3; border-radius:10px; margin:.6rem 0 2rem; }
      .metric-table { width:100%; min-width:1250px; table-layout:fixed; border-collapse:collapse; margin:0 !important; color:#26384D; font-size:1.75rem; }
      .metric-table th { background:#001E79; color:white; border:2px solid white; padding:18px 20px; text-align:left; font-size:1.9rem; white-space:nowrap; }
      .metric-table td { padding:18px 20px; border-bottom:1px solid #DCE2F3; text-align:left; vertical-align:middle; line-height:1.4; }
      .metric-table tbody tr:nth-child(even) { background:#F5F7FC; }
      .metric-table tbody tr:last-child td { border-bottom:0; }
      .metric-table td:first-child { color:#001E79; font-weight:700; white-space:nowrap; }
      .metric-table th, .metric-table td { box-sizing:border-box; }
      .metric-table th:nth-child(1), .metric-table td:nth-child(1) { width:18%; }
      .metric-table th:nth-child(2), .metric-table td:nth-child(2) { width:39%; }
      .metric-table th:nth-child(3), .metric-table td:nth-child(3) { width:17%; }
      .metric-table th:nth-child(4), .metric-table td:nth-child(4) { width:26%; }
      .raw-table-title { color:#001E79; font-size:1.55rem; font-weight:800; margin:1.5rem 0 .7rem; }
      .raw-table-scroll { height:1200px; overflow:scroll; scrollbar-gutter:stable both-edges; background:white; border:2px solid #DCE2F3; border-radius:12px; margin-bottom:4rem; }
      .raw-table-scroll::-webkit-scrollbar { width:18px; height:18px; }
      .raw-table-scroll::-webkit-scrollbar-track { background:#E8ECF5; }
      .raw-table-scroll::-webkit-scrollbar-thumb { background:#3152A4; border:3px solid #E8ECF5; border-radius:10px; }
      .raw-table { width:100%; min-width:900px; border-collapse:collapse; margin:0 !important; font-size:1.7rem; line-height:1.45; color:#12213A; }
      .raw-table thead th { position:sticky; top:0; z-index:2; box-sizing:border-box; background:#001E79; color:white; border:2px solid white; padding:18px 20px; text-align:center; vertical-align:middle; white-space:nowrap; }
      .raw-table tbody td { box-sizing:border-box; padding:16px 20px; border:1px solid #DCE2F3; text-align:center; vertical-align:middle; white-space:nowrap; }
      .raw-table tbody td:nth-child(2) { text-align:center; }
      .raw-table tbody tr:nth-child(even) { background:#F5F7FC; }
      @media (max-width:1200px) { .profile-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
      @media (max-width:900px) { .profile-grid, .field-grid, .metric-grid { grid-template-columns:1fr; } }
      [data-testid="stSidebar"] { min-width:500px; max-width:500px; }
      [data-testid="stSidebarNav"] a, [data-testid="stSidebarNav"] a span { font-size:2rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebarNav"] li { margin-bottom:.7rem !important; }
      .source-toc { margin-top:1.6rem; padding-top:1.4rem; border-top:1px solid #CBD5E1; }
      .source-toc-title { color:#0B2E6F; font-size:1.8rem; font-weight:800; margin-bottom:.8rem; }
      .source-toc a { display:block; color:#0B2E6F !important; font-size:1.5rem; font-weight:700; line-height:1.25; padding:.7rem .8rem; margin:.25rem 0; border-radius:8px; text-decoration:none !important; }
      .source-toc a:hover { background:#E9EEF8; color:#001E79 !important; }
      .source-toc .toc-step { font-size:1.25rem; font-weight:600; color:#465269 !important; padding:.38rem .8rem .38rem 1.7rem; margin:.05rem 0; }
      .section-anchor { scroll-margin-top:6.75rem; }
      [data-testid="stSidebarNav"] a { min-height:3.8rem !important; padding:.55rem .8rem !important; border-radius:10px !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { min-height:3.8rem !important; padding:.55rem .8rem !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p { font-size:2rem !important; font-weight:800 !important; line-height:1.2 !important; }
      [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] p, [data-testid="stSidebarNav"] a { white-space:normal !important; overflow:visible !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.page_link("App.py", label="Dashboard Overview")
    st.page_link("pages/1_Source_Data.py", label="Source Data for Three Businesses")
    st.markdown(
        """
        <nav class="source-toc" aria-label="Source data contents">
          <div class="source-toc-title">Contents</div>
          <a href="#commercial-banking">I. Commercial Banking</a>
          <a class="toc-step" href="#commercial-banking-raw-data-issues">1. Raw Data Issues</a>
          <a class="toc-step" href="#commercial-banking-important-columns">2. Important Columns</a>
          <a class="toc-step" href="#commercial-banking-defined-metrics">3. Defined Metrics</a>
          <a class="toc-step" href="#commercial-banking-raw-source-data">4. Raw Source Data</a>
          <a href="#commercial-real-estate">II. Commercial Real Estate</a>
          <a class="toc-step" href="#commercial-real-estate-raw-data-issues">1. Raw Data Issues</a>
          <a class="toc-step" href="#commercial-real-estate-important-columns">2. Important Columns</a>
          <a class="toc-step" href="#commercial-real-estate-defined-metrics">3. Defined Metrics</a>
          <a class="toc-step" href="#commercial-real-estate-raw-source-data">4. Raw Source Data</a>
          <a href="#capital-markets">III. Capital Markets</a>
          <a class="toc-step" href="#capital-markets-raw-data-issues">1. Raw Data Issues</a>
          <a class="toc-step" href="#capital-markets-important-columns">2. Important Columns</a>
          <a class="toc-step" href="#capital-markets-defined-metrics">3. Defined Metrics</a>
          <a class="toc-step" href="#capital-markets-raw-source-data">4. Raw Source Data</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )

st.image(str(LOGO_PATH), width=220)
st.title("Source Data for Three Businesses")
st.markdown(
    """
    <div class="simulated-notice">
      <strong>Disclaimer: Data is synthetic and for demonstration purposes only, not actual U.S. Bank figures.</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
quality_profile = profile_raw_sources().set_index("business_unit")
roman_numerals = ("I", "II", "III")
for section_index, (business_unit, source_name, source_table) in enumerate(load_source_tables()):
    section_id = business_unit.lower().replace(" ", "-")
    st.markdown(f"<div id='{section_id}' class='section-anchor'></div>", unsafe_allow_html=True)
    st.header(f"{roman_numerals[section_index]}. {business_unit}")
    st.markdown(
        f"<div class='source-meta'><strong>File:</strong> {source_name} &nbsp;·&nbsp; "
        f"<strong>Size:</strong> {len(source_table):,} rows × {len(source_table.columns):,} columns</div>",
        unsafe_allow_html=True,
    )
    st.markdown(source_profile_html(business_unit, source_table, quality_profile.loc[business_unit]), unsafe_allow_html=True)
    st.markdown(
        f"<div id='{section_id}-raw-source-data' class='story-step section-anchor'><span>4</span><div><strong>Raw Source Data</strong></div></div>",
        unsafe_allow_html=True,
    )
    table_html = source_table.to_html(index=False, border=0, classes="raw-table", float_format=lambda value: f"{value:,.3f}")
    st.markdown(f"<div class='raw-table-scroll'>{table_html}</div>", unsafe_allow_html=True)
