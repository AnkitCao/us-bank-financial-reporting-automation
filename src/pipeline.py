"""Run the complete reporting pipeline and persist auditable outputs."""

from __future__ import annotations

from src.calculate_kpis import build_metric_detail, build_monthly_kpis
from src.clean_data import clean_all_sources
from src.detect_exceptions import detect_exceptions
from src.generate_data import generate_raw_packages
from src.map_financials import apply_financial_mapping
from src.paths import PROCESSED_DIR


def run_pipeline() -> None:
    """Generate raw packages, standardize data, calculate KPIs, and detect alerts."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    generate_raw_packages()
    cleaned = clean_all_sources()
    mapped = apply_financial_mapping(cleaned)
    kpis = build_monthly_kpis(mapped)
    detail = build_metric_detail(mapped)
    alerts = detect_exceptions(kpis)

    cleaned.to_csv(PROCESSED_DIR / "cleaned_ledger.csv", index=False)
    mapped.to_csv(PROCESSED_DIR / "mapped_financials.csv", index=False)
    kpis.to_csv(PROCESSED_DIR / "monthly_kpis.csv", index=False)
    detail.to_csv(PROCESSED_DIR / "metric_detail.csv", index=False)
    alerts.to_csv(PROCESSED_DIR / "executive_alerts.csv", index=False)

    print(f"Pipeline complete: {len(mapped):,} mapped records, {len(alerts):,} alerts.")


if __name__ == "__main__":
    run_pipeline()

