import pandas as pd
from pathlib import Path


def add_gt_label(
    pred_path: str,
    gt_path: str,
    output_path: str = None,
    pred_cve_col: str = "CVE-ID",
    pred_capec_col: str = "CAPEC-ID",
    gt_cve_col: str = "CVE-ID",
    gt_capec_col: str = "CAPEC-ID",
    gt_label_col: str = "Label",
    new_label_col: str = "Label",
) -> pd.DataFrame:
    """
    Compare the prediction file against the GT file and attach a GT label
    to each prediction based on the (CVE-ID, CAPEC-ID) combination.

    Parameters
    ----------
    pred_path : Path to the prediction file (.xlsx / .csv)
    gt_path   : Path to the ground truth file (.xlsx / .csv)
    output_path : Output file path; if None, the result is only returned and not saved
    pred_cve_col, pred_capec_col : Key column names in the prediction file
    gt_cve_col, gt_capec_col     : Key column names in the GT file
    gt_label_col : Name of the label column in the GT file
    new_label_col : Name of the new column added to the prediction file
                    (to avoid conflict with any existing column)

    Returns
    -------
    DataFrame : The prediction DataFrame with an additional `new_label_col` column.
                Samples present in the predictions but absent from the GT will have
                NaN as their label.
    """

    def _read(path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str)
        return pd.read_excel(path, dtype=str)

    # 1. Read files
    pred_df = _read(pred_path)
    gt_df = _read(gt_path)

    # 2. Check required columns
    for col, df_name, df in [
        (pred_cve_col, "pred", pred_df),
        (pred_capec_col, "pred", pred_df),
        (gt_cve_col, "gt", gt_df),
        (gt_capec_col, "gt", gt_df),
        (gt_label_col, "gt", gt_df),
    ]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' does not exist in the {df_name} file. "
                f"Actual columns in this file: {list(df.columns)}"
            )

    # 3. Build normalized matching keys (uppercase + strip to remove case and whitespace differences)
    def _norm(s):
        return s.astype(str).str.strip().str.upper()

    pred_df["_key_cve"] = _norm(pred_df[pred_cve_col])
    pred_df["_key_capec"] = _norm(pred_df[pred_capec_col])
    gt_df["_key_cve"] = _norm(gt_df[gt_cve_col])
    gt_df["_key_capec"] = _norm(gt_df[gt_capec_col])

    # 4. Deduplicate GT (if the same CVE+CAPEC pair appears multiple times in GT,
    #    keep the first occurrence's label)
    gt_unique = gt_df.drop_duplicates(subset=["_key_cve", "_key_capec"], keep="first")
    dup_in_gt = len(gt_df) - len(gt_unique)
    if dup_in_gt > 0:
        print(f"[Warning] {dup_in_gt} duplicate (CVE-ID, CAPEC-ID) pairs found in GT; first occurrence kept.")

    # 5. Left join: predictions are the base, GT label is attached
    gt_slim = gt_unique[["_key_cve", "_key_capec", gt_label_col]].rename(
        columns={gt_label_col: new_label_col}
    )
    merged = pred_df.merge(gt_slim, on=["_key_cve", "_key_capec"], how="left")

    # 6. Clean up temporary columns
    merged = merged.drop(columns=["_key_cve", "_key_capec"])

    # 7. Report
    total = len(merged)
    matched = merged[new_label_col].notna().sum()
    unmatched = total - matched
    print("============ Label Merge Report ============")
    print(f"Pred samples      : {total}")
    print(f"GT samples        : {len(gt_unique)} (unique after dedup)")
    print(f"Matched w/ label  : {matched}")
    print(f"Unmatched (NaN)   : {unmatched}")
    print("============================================")

    # 8. Save
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            merged.to_csv(out, index=False)
        else:
            merged.to_excel(out, index=False)
        print(f"[System] Labeled predictions saved to: {output_path}")

    return merged


if __name__ == "__main__":
    df = add_gt_label(
        pred_path="data/Test_dataset/output/Test_output/New_set_5_scores_hard.xlsx",
        gt_path="data/Test_dataset/Test_hard.xlsx",
        output_path="data/Test_dataset/output/Hard/New_test_5_scores_hard_with_label.xlsx",
    )
    # add structure score
    # df = add_gt_label(
    #     pred_path="data/Test_dataset/output/Hard/New_test_4_scores_hard.xlsx",
    #     gt_path="data/Test_dataset/output/Hard/Structure_prediction_newtest_hard.xlsx",
    #     output_path="data/Test_dataset/output/Hard/New_test_5_scores_hard.xlsx",
    # )