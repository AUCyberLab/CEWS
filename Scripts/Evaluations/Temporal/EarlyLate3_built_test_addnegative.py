import pandas as pd
import numpy as np


def read_table(file_path):
    return pd.read_excel(str(file_path))


def write_table(df, file_path):
    df.to_excel(str(file_path), index=False)


def is_positive_label(x):
    """Return True if x represents a positive label (1, 1.0, true, yes)."""
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "1.0", "true", "yes"}


def build_testset_with_negatives(
    positive_file,
    capec_info_file,
    output_file,
    seed=42,
    negatives_per_cve=2,
    label_col="Label",
    cve_id_col="CVE-ID",
    capec_id_col="CAPEC-ID",
):
    """
    Build a test set by sampling negative (CVE, CAPEC) pairs for every CVE in the
    positive file, then concatenating positives and negatives.

    Args:
        positive_file:     Excel file containing only positive samples.
        capec_info_file:   CAPEC info file; must include the CAPEC-ID column.
        output_file:       Destination Excel for the final test set.
        seed:              Random seed for negative sampling.
        negatives_per_cve: Number of negative CAPECs to draw per unique CVE.
        label_col:         Label column name.
        cve_id_col:        CVE-ID column name.
        capec_id_col:      CAPEC-ID column name.
    """
    positive_df = read_table(positive_file)
    capec_df = read_table(capec_info_file)

    # Strip whitespace from column names so accidental leading/trailing spaces don't break lookups
    positive_df.columns = [str(c).strip() for c in positive_df.columns]
    capec_df.columns = [str(c).strip() for c in capec_df.columns]

    required_pos_cols = {label_col, cve_id_col, capec_id_col}
    missing = required_pos_cols - set(positive_df.columns)
    if missing:
        raise ValueError(f"Positive file is missing required columns: {missing}")

    if capec_id_col not in capec_df.columns:
        raise ValueError(f"CAPEC info file is missing required column: {capec_id_col}")

    # Keep only valid positive rows
    positive_df = positive_df[positive_df[label_col].apply(is_positive_label)].copy()
    positive_df = positive_df[
        positive_df[cve_id_col].notna() & positive_df[capec_id_col].notna()
    ].copy()

    if positive_df.empty:
        raise ValueError("No usable positive samples in the positive file.")

    # Build the CAPEC pool used for negative sampling
    capec_df = capec_df[capec_df[capec_id_col].notna()].copy()
    capec_df = capec_df.drop_duplicates(subset=capec_id_col, keep="first")

    if capec_df.empty:
        raise ValueError("No usable CAPEC entries in the CAPEC info file.")

    capec_info_dict = capec_df.set_index(capec_id_col).to_dict(orient="index")
    all_capec_pool = sorted(capec_info_dict.keys())

    # Per-CVE ground-truth CAPEC sets, used to exclude positives from the negative pool
    gt_map = (
        positive_df.groupby(cve_id_col)[capec_id_col]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )

    # Use the first positive row of each CVE as the template (preserves CVE-side fields)
    cve_template_df = positive_df.drop_duplicates(subset=cve_id_col, keep="first").copy()

    rng = np.random.default_rng(seed)
    negative_rows = []

    for _, template_row in cve_template_df.iterrows():
        current_cve = template_row[cve_id_col]
        gt_capecs = gt_map.get(current_cve, set())

        candidate_pool = [capec for capec in all_capec_pool if capec not in gt_capecs]

        if len(candidate_pool) < negatives_per_cve:
            raise ValueError(
                f"{current_cve} has only {len(candidate_pool)} candidate negatives, "
                f"need {negatives_per_cve}."
            )

        sampled_neg_capecs = rng.choice(
            candidate_pool,
            size=negatives_per_cve,
            replace=False,
        )

        for neg_capec in sampled_neg_capecs:
            new_row = template_row.to_dict()

            new_row[cve_id_col] = current_cve
            new_row[label_col] = 0
            new_row[capec_id_col] = neg_capec

            # Overwrite CAPEC-side fields with the negative CAPEC's info
            capec_info = capec_info_dict[neg_capec]
            for col, value in capec_info.items():
                if col == capec_id_col:
                    continue
                new_row[col] = value

            negative_rows.append(new_row)

    negative_df = pd.DataFrame(negative_rows)

    testset_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Sort so each CVE's positives come before its negatives
    testset_df["_LABEL_SORT_"] = testset_df[label_col].apply(
        lambda x: 1 if is_positive_label(x) else 0
    )
    testset_df = (
        testset_df.sort_values(by=[cve_id_col, "_LABEL_SORT_"], ascending=[True, False])
        .drop(columns=["_LABEL_SORT_"])
        .reset_index(drop=True)
    )

    write_table(testset_df, output_file)

    print(f"Positive rows:       {len(positive_df)}")
    print(f"Unique CVEs:         {positive_df[cve_id_col].nunique()}")
    print(f"Negatives generated: {len(negative_df)}")
    print(f"Final testset size:  {len(testset_df)}")
    print(f"Output file:         {output_file}")


if __name__ == "__main__":
    build_testset_with_negatives(
        positive_file="data/Early_late_dataset/EarlyLate_test_positives.xlsx",
        capec_info_file="data/RankingEx/CAPEC_info/CAPEC_info_all.xlsx",
        output_file="data/Early_late_dataset/EarlyLate_testset1.xlsx",
        seed=42,
        negatives_per_cve=2,
        label_col="Label",
        cve_id_col="CVE-ID",
        capec_id_col="CAPEC-ID",
    )