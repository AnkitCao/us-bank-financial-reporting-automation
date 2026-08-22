# Data Dictionary

All monetary values use synthetic USD millions unless a balance label states otherwise.

## Standard ledger

| Field | Definition |
|---|---|
| `business_unit` | Commercial Banking, Commercial Real Estate, or Capital Markets |
| `period` | Standardized month-end date |
| `scenario` | Actual, Budget, Forecast, or Prior Year benchmark |
| `source_metric` | Business-specific source label retained for traceability |
| `management_category` | Standard management reporting category |
| `metric_type` | Revenue, Expense, Credit, or Balance |
| `amount_millions` | Numeric value in USD millions |

## Management KPIs

| KPI | Definition |
|---|---|
| Total Revenue | Sum of records classified as Revenue |
| Pre-Provision Profit | Total Revenue minus Operating Expense |
| Adjusted Profit | Pre-Provision Profit minus Credit Provision |
| Profit Margin | Adjusted Profit divided by Total Revenue |
| vs Budget | `(Actual - Budget) / Budget` |
| vs Forecast | `(Actual - Forecast) / Forecast` |
| YoY | `(Actual - Prior Year benchmark) / Prior Year benchmark` |
| MoM | Percentage change in Actual Revenue from the previous month |
| T12M Revenue | Rolling sum of up to 12 monthly Actual Revenue values |
| Forecast Accuracy | `1 - abs(Actual - Forecast) / abs(Forecast)` |

## Controls

- Mapping is a controlled many-to-one join; unmapped metrics stop the pipeline.
- Amounts must be numeric after cleaning; missing values stop the pipeline.
- Balance metrics remain outside revenue and profit calculations.
- Exception rules run before narrative generation.

