"""Core reconciliation and business-rule tests."""

import pandas as pd
import pytest

from src.calculate_kpis import build_monthly_kpis
from src.clean_data import _finalize_ledger, _require_columns, clean_all_sources, profile_raw_sources, quality_rule_catalog
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


def test_business_model_ratios_use_matching_economic_drivers(tmp_path):
    """Common ratios apply to all units; specialized ratios apply only where the source model supports them."""
    generate_raw_packages(tmp_path)
    kpis = build_monthly_kpis(apply_financial_mapping(clean_all_sources(tmp_path)))
    latest = kpis.loc[kpis["period"] == pd.Timestamp("2026-06-30")].set_index("business_unit")
    assert latest["cost_to_income_ratio_actual"].notna().all()
    assert latest["profit_margin_actual"].notna().all()
    assert pd.notna(latest.loc["Commercial Banking", "loan_to_deposit_ratio_actual"])
    assert pd.isna(latest.loc["Commercial Real Estate", "loan_to_deposit_ratio_actual"])
    assert pd.notna(latest.loc["Commercial Real Estate", "npl_ratio_actual"])
    assert pd.isna(latest.loc["Capital Markets", "npl_ratio_actual"])
    mix_columns = ["advisory_mix_actual", "underwriting_mix_actual", "trading_mix_actual", "structuring_mix_actual", "syndication_mix_actual"]
    assert latest.loc["Capital Markets", mix_columns].sum() == pytest.approx(1.0)
    assert latest.loc["Commercial Banking", mix_columns].isna().all()


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


def test_raw_packages_contain_controlled_quality_issues(tmp_path):
    """The source layer should demonstrate realistic EDA and cleaning work."""
    generate_raw_packages(tmp_path)
    profile = profile_raw_sources(tmp_path)
    assert profile["missing_values"].gt(0).all()
    assert profile["duplicate_keys"].gt(0).all()
    assert profile["outliers"].gt(0).all()
    assert profile["column_aliases"].gt(0).all()
    assert profile["problem_rate"].ge(0.05).all()


def test_cleaning_repairs_issues_and_standardizes_schema(tmp_path):
    """The clean layer must be complete, unique and use one canonical schema."""
    generate_raw_packages(tmp_path)
    cleaned = clean_all_sources(tmp_path)
    key = ["business_unit", "period", "scenario", "source_metric"]
    assert cleaned.columns.tolist() == key + ["amount_millions"]
    assert not cleaned.isna().any().any()
    assert not cleaned.duplicated(key).any()
    assert set(cleaned["scenario"]) == {"Actual", "Budget", "Forecast", "Prior Year"}


def test_generation_is_reproducible(tmp_path):
    """Two runs must create identical source values and identical quality issues."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_raw_packages(first)
    generate_raw_packages(second)
    for filename, sheets in {
        "commercial_banking_monthly.xlsx": ["Monthly Detail"],
        "commercial_real_estate_monthly.xlsx": ["CRE Monthly"],
        "capital_markets_monthly.xlsx": None,
    }.items():
        left = pd.read_excel(first / filename, sheet_name=sheets)
        right = pd.read_excel(second / filename, sheet_name=sheets)
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for sheet in left:
                pd.testing.assert_frame_equal(left[sheet], right[sheet])
        else:
            pd.testing.assert_frame_equal(left, right)


def test_repairs_stay_within_metric_and_scenario_group():
    """Missing and extreme amounts use only their own ordered comparison group."""
    periods = pd.date_range("2026-01-31", periods=6, freq="ME")
    frame = pd.DataFrame({
        "business_unit": "Test Unit", "period": periods, "scenario": "Actual", "source_metric": "Fee Revenue",
        "amount_millions": [10.0, 11.0, None, 13.0, 200.0, 15.0],
    })
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    cleaned = _finalize_ledger(frame).sort_values("period")
    assert cleaned["amount_millions"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert len(cleaned) == 6


def test_conflicting_duplicate_keys_fail_instead_of_silent_selection():
    frame = pd.DataFrame({
        "business_unit": ["Test Unit", "Test Unit"],
        "period": [pd.Timestamp("2026-01-31")] * 2,
        "scenario": ["Actual", "Actual"],
        "source_metric": ["Fee Revenue", "Fee Revenue"],
        "amount_millions": [10.0, 99.0],
    })
    with pytest.raises(ValueError, match="Conflicting rows"):
        _finalize_ledger(frame)


def test_quality_rules_are_explicit_and_complete():
    rules = quality_rule_catalog().set_index("issue_type")
    assert {"Column aliases", "Missing amounts", "Exact duplicates", "Extreme values"}.issubset(rules.index)


def test_unresolved_required_fields_stop_cleaning():
    """Unknown schemas must be quarantined rather than silently guessed."""
    with pytest.raises(ValueError, match="Unresolved required fields"):
        _require_columns(pd.DataFrame({"mystery": [1]}), ["period"], "new_file.xlsx")
