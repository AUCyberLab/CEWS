import pandas as pd
from typing import Any, Optional


def add_label_from_gt(
    sample_file: str,
    gt_file: str,
    output_file: Optional[str] = None,
    sample_sheet_name: Any = 0,
    gt_sheet_name: Any = 0,
    sample_cve_col: str = "CVE-ID",
    sample_capec_col: str = "CAPEC-ID",
    gt_cve_col: str = "CVE_ID",
    gt_capec_col: str = "CAPEC_ID",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Add a binary label column to the sample file based on (CVE, CAPEC) pairs
    present in the ground-truth file. Pairs found in GT get label 1, others 0.

    Args:
        sample_file:       Path to the Excel file to be labeled.
        gt_file:           Path to the ground-truth Excel file.
        output_file:       If provided, the labeled DataFrame is written here.
        sample_sheet_name: Sheet name or index for the sample file.
        gt_sheet_name:     Sheet name or index for the GT file.
        sample_cve_col:    CVE column name in the sample file.
        sample_capec_col:  CAPEC column name in the sample file.
        gt_cve_col:        CVE column name in the GT file.
        gt_capec_col:      CAPEC column name in the GT file.
        label_col:         Name of the output label column.

    Returns:
        The sample DataFrame with the label column added.
    """
    sample_df = pd.read_excel(sample_file, sheet_name=sample_sheet_name)
    gt_df = pd.read_excel(gt_file, sheet_name=gt_sheet_name)

    for col in (sample_cve_col, sample_capec_col):
        if col not in sample_df.columns:
            raise ValueError(f"Column not found in sample file: {col}")

    for col in (gt_cve_col, gt_capec_col):
        if col not in gt_df.columns:
            raise ValueError(f"Column not found in GT file: {col}")

    gt_pairs = set(zip(gt_df[gt_cve_col], gt_df[gt_capec_col]))

    # Vectorized membership check; faster than apply(lambda) for large datasets
    sample_pairs = list(zip(sample_df[sample_cve_col], sample_df[sample_capec_col]))
    sample_df[label_col] = [1 if pair in gt_pairs else 0 for pair in sample_pairs]

    if output_file is not None:
        sample_df.to_excel(output_file, index=False)

    return sample_df


if __name__ == "__main__":
    df = add_label_from_gt(
        sample_file="data/RankingEx/output/TL_predictions_5_score.xlsx",
        gt_file="data/RankingEx/ThreatLinker_cve-capec-mapping.xlsx",
        output_file="data/RankingEx/output/TL_predictions_5_score_with_label.xlsx",
    )