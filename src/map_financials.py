"""Map business-specific labels to a consistent management taxonomy."""

from __future__ import annotations

import pandas as pd

from src.paths import CONFIG_DIR


def apply_financial_mapping(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Join the controlled vocabulary map and reject unmapped source metrics."""
    mapping = pd.read_csv(CONFIG_DIR / "financial_mapping.csv")
    mapped = cleaned.merge(mapping, on=["business_unit", "source_metric"], how="left", validate="many_to_one")
    missing = mapped.loc[mapped["management_category"].isna(), ["business_unit", "source_metric"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(f"Unmapped financial metrics:\n{missing.to_string(index=False)}")
    return mapped

