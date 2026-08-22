"""Map business-specific labels to a consistent management taxonomy."""

from __future__ import annotations

import pandas as pd

from src.llm_data_quality import review_semantic_mappings
from src.paths import CONFIG_DIR


def apply_financial_mapping(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Join the controlled vocabulary map and reject unmapped source metrics."""
    mapping = pd.read_csv(CONFIG_DIR / "financial_mapping.csv")
    mapped = cleaned.merge(mapping, on=["business_unit", "source_metric"], how="left", validate="many_to_one")
    missing = mapped.loc[mapped["management_category"].isna(), ["business_unit", "source_metric"]].drop_duplicates()
    if not missing.empty:
        repaired = cleaned.copy()
        changed = False
        for business_unit, rows in missing.groupby("business_unit"):
            allowed = mapping.loc[mapping["business_unit"].eq(business_unit), "source_metric"].tolist()
            reviewed = review_semantic_mappings(
                kind="financial metric labels",
                values=rows["source_metric"].astype(str).tolist(),
                allowed_targets=allowed,
                context=f"Map unfamiliar {business_unit} metric spellings to the existing controlled financial vocabulary.",
            )
            if reviewed:
                mask = repaired["business_unit"].eq(business_unit)
                repaired.loc[mask, "source_metric"] = repaired.loc[mask, "source_metric"].replace(reviewed)
                changed = True
        if changed:
            mapped = repaired.merge(mapping, on=["business_unit", "source_metric"], how="left", validate="many_to_one")
            missing = mapped.loc[mapped["management_category"].isna(), ["business_unit", "source_metric"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(f"Unmapped financial metrics:\n{missing.to_string(index=False)}")
    return mapped
