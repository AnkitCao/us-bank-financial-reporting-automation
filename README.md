# U.S. Bank Financial Reporting Automation

An interview-ready demonstration of a multi-business monthly finance reporting pipeline:

**Synthetic Finance Packages → Cleaning → Financial Taxonomy Mapping → KPI & Exception Detection → Streamlit CFO Review**

> **Disclaimer:** All monthly figures are synthetic and simplified for demonstration. Public U.S. Bancorp disclosures are used only to calibrate directional scenarios. This project does not contain or represent U.S. Bank internal data, accounting definitions, forecasts, or organizational structure.

## What this project demonstrates

- Three intentionally heterogeneous Excel inputs
- Business-specific cleaning logic instead of one fragile generic parser
- Controlled financial vocabulary mapping with an unmapped-metric stop check
- Auditable Actual, Budget, Forecast, Prior Year, YoY, MoM and T12M calculations
- Deterministic exception detection before narrative generation
- A summary-first Streamlit page designed for a monthly CFO/LOB review
- Clear separation between public evidence, analytical inference and synthetic assumptions

## Architecture

```text
data/raw/*.xlsx
      │
      ├── Commercial Banking: stacked long table
      ├── CRE: wide monthly table
      └── Capital Markets: month columns across metric sheets
      │
      ▼
src/clean_data.py
      ▼
config/financial_mapping.csv → src/map_financials.py
      ▼
src/calculate_kpis.py → src/detect_exceptions.py
      ▼
data/processed/*.csv
      ▼
app/App.py → Automated Three-Business Performance Dashboard
```

## Quick start

```bash
cd "/Users/ankit/Downloads/US Bank Financial Reporting Automation"
python -m src.pipeline
streamlit run app/App.py
```

Open the local URL printed by Streamlit. The default review month is June 2026.

## Run tests

```bash
pytest -q
```

Tests cover scenario completeness, mapping coverage, the Capital Markets reality-calibrated scenario, and the rule-engine alert path.

## Repository guide

```text
app/                         Streamlit presentation layer
config/                      Business rules and financial mapping
data/raw/                    Three synthetic heterogeneous Excel packages
data/processed/              Cleaned ledger, mapped data, KPIs and alerts
docs/                        Data dictionary, scenario rationale and interview script
scripts/                     Convenience entry points
src/                         Pipeline modules with one responsibility each
tests/                       Reconciliation and business-rule tests
```

## KPI model

All three business lines share management KPIs: Total Revenue, Operating Expense, Adjusted Profit, Profit Margin, Budget Attainment, Forecast Variance, YoY, MoM and T12M Revenue.

Business-specific drivers remain visible:

- **Commercial Banking:** loan interest, treasury/deposit/merchant/FX fees, loan and deposit balances
- **Commercial Real Estate:** interest income, origination/prepayment/servicing fees, credit provision, mortgage balance and NPL proxy
- **Capital Markets:** advisory, underwriting, trading, structuring and syndication revenue

## Exception rules

- Revenue YoY decline greater than 10% — Red
- Operating expense over budget greater than 5% — Yellow
- Credit provision over forecast greater than 5% — Yellow
- Lowest trailing-window profit margin — Yellow
- Three consecutive monthly revenue declines — Red
- Positive growth after three declines — Green

Thresholds are centralized in `config/business_rules.yaml`.

## Reality calibration

The latest scenario anchor is U.S. Bancorp Q2 2026. Public disclosures reported strong commercial loan growth, improving CRE origination balances with mixed localized credit signals, and a material increase in Capital Markets revenue associated with client activity and the BTIG acquisition.

Primary source: [U.S. Bancorp Second Quarter 2026 Results](https://www.sec.gov/Archives/edgar/data/36104/000003610426000039/a2q26earningsrelease.htm)

See `docs/scenario_design.md` for the fact/inference/assumption boundary.

## Narrative-control design

`src/executive_summary.py` generates the default briefing from validated outputs. It does not recalculate KPIs or inspect raw data. A production LLM can replace the wording layer later, but it should receive only reviewed KPI, variance and alert records.
