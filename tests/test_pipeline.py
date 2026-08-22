"""Core reconciliation and business-rule tests."""

import pandas as pd

from src.calculate_kpis import build_monthly_kpis
from src.clean_data import clean_all_sources
from src.detect_exceptions import detect_exceptions
from src.generate_data import generate_raw_packages
from src.map_financials import apply_financial_mapping


def test_pipeline_has_complete_scenario_coverage(tmp_path):
    """Every business unit, month, and metric must contain four scenarios."""
    generate_raw_packages(tmp_path)
    cleaned = clean_all_sources(tmp_path)
    counts = cleaned.groupby(["business_unit", "period", "source_metric"])["scenario"].nunique()
    assert counts.eq(4).all()
    assert cleaned["period"].nunique() == 15


def test_all_source_metrics_are_mapped(tmp_path):
    """The controlled vocabulary must cover every raw source metric."""
    generate_raw_packages(tmp_path)
    mapped = apply_financial_mapping(clean_all_sources(tmp_path))
    assert mapped["management_category"].notna().all()
    assert set(mapped["business_unit"].unique()) == {"Commercial Banking", "Commercial Real Estate", "Capital Markets"}


def test_latest_capital_markets_outperforms_budget(tmp_path):
    """The BTIG-calibrated June scenario should visibly beat plan."""
    generate_raw_packages(tmp_path)
    kpis = build_monthly_kpis(apply_financial_mapping(clean_all_sources(tmp_path)))
    row = kpis.loc[(kpis["business_unit"] == "Capital Markets") & (kpis["period"] == pd.Timestamp("2026-06-30"))].iloc[0]
    assert row["revenue_vs_budget"] > 0.15


def test_rule_engine_flags_latest_capital_markets_expense(tmp_path):
    """Integration-related expense pressure should trigger the configured alert."""
    generate_raw_packages(tmp_path)
    kpis = build_monthly_kpis(apply_financial_mapping(clean_all_sources(tmp_path)))
    alerts = detect_exceptions(kpis)
    matches = alerts.loc[
        (alerts["business_unit"] == "Capital Markets")
        & (alerts["period"] == pd.Timestamp("2026-06-30"))
        & (alerts["rule"] == "Expense over budget")
    ]
    assert not matches.empty

