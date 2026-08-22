# Three-Minute Interview Demo

## 0:00-0:35 — Frame the problem

“Finance receives three monthly packages with different structures: one stacked table, one wide table, and one workbook with months across columns. The goal is not only to rename columns; it is to translate business-specific financial vocabulary into a comparable management view.”

## 0:35-1:15 — Show the pipeline

Open `config/financial_mapping.csv`, then explain:

1. Raw source structures are normalized independently.
2. Source metrics are mapped into a controlled management taxonomy.
3. KPI calculations are deterministic and reconciled.
4. Rules detect exceptions before any narrative is generated.

## 1:15-2:25 — Show the dashboard

Use June 2026 and all business units.

- Commercial Banking shows stable balance-led performance.
- Capital Markets materially outperforms plan, calibrated to public Q2 momentum and the first month of BTIG contribution.
- CRE originations improve, while a synthetic local provision pressure alert preserves management focus on credit.
- The waterfall explains Budget-to-Actual movement instead of showing only totals.

## 2:25-3:00 — Close with controls

“The executive summary never calculates a number. It only reads validated KPI and alert outputs. Every chart can be traced to a processed table, every processed metric to a mapping row, and every mapping row to a source package.”

## Likely follow-up questions

**Why synthetic data?** Public disclosures do not provide monthly product-level P&Ls. Synthetic data protects confidentiality while preserving realistic relationships.

**Why not force all metrics into one taxonomy?** Business-specific drivers remain visible. Only comparable management categories are standardized.

**How would this productionize?** Replace Excel reads with governed source tables, add data-quality checks and orchestration, and publish certified metric definitions.

**What does the LLM do?** Interpretation only. Python owns calculations, comparisons, and exception decisions.

