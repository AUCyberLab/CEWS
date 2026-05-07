from pathlib import Path
import pandas as pd


def read_table(file_path):
    """Read an Excel or CSV table; for CSVs, try several encodings before giving up."""
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1", "gbk"]
        last_error = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError as e:
                last_error = e
        else:
            raise ValueError(f"Unable to read {file_path}. Last error: {last_error}")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def write_table(df, file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(file_path, index=False)
    elif file_path.suffix.lower() == ".csv":
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def extract_and_filter_by_score(
    cve_list_file,
    new_file,
    output_file,
    cve_col="CVE-ID",
    score_col="Final_Score",
    mode="topk",
    topk=10,
    threshold=0.5,
    threshold_inclusive=False,
    capec_col_in_cve=None,
    capec_col_in_new=None,
):
    """
    Extract rows from `new_file` whose CVE-ID appears in `cve_list_file`,
    then narrow the result down using one of four filtering strategies.

    Modes:
        topk                        - keep the top-k highest-scoring rows per CVE
        threshold                   - keep rows whose score passes the threshold
        topk_then_threshold         - take top-k per CVE, then apply threshold
        intersection_and_threshold  - keep rows whose CAPEC matches the
                                      related-CAPEC list in cve_list_file AND
                                      whose score passes the threshold

    For 'intersection_and_threshold', both `capec_col_in_cve` and
    `capec_col_in_new` must be supplied.
    """
    df_cve = read_table(cve_list_file)
    df_new = read_table(new_file)

    if cve_col not in df_cve.columns:
        raise KeyError(f"Column not found in cve_list_file: {cve_col}")
    if cve_col not in df_new.columns:
        raise KeyError(f"Column not found in new_file: {cve_col}")
    if score_col not in df_new.columns:
        raise KeyError(f"Column not found in new_file: {score_col}")

    df_cve[cve_col] = df_cve[cve_col].astype(str).str.strip()
    df_new[cve_col] = df_new[cve_col].astype(str).str.strip()

    cve_set = set(df_cve[cve_col].dropna().astype(str).str.strip().unique())

    extracted_df = df_new[df_new[cve_col].isin(cve_set)].copy()

    # Drop rows whose score isn't numeric so they don't affect ranking or threshold checks
    extracted_df[score_col] = pd.to_numeric(extracted_df[score_col], errors="coerce")
    extracted_df = extracted_df.dropna(subset=[score_col]).copy()

    extracted_df = extracted_df.sort_values(
        by=[cve_col, score_col],
        ascending=[True, False],
    ).copy()

    def apply_threshold(df):
        if threshold_inclusive:
            return df[df[score_col] >= threshold].copy()
        return df[df[score_col] > threshold].copy()

    if mode == "topk":
        filtered_df = extracted_df.groupby(cve_col, group_keys=False).head(topk).copy()

    elif mode == "threshold":
        filtered_df = apply_threshold(extracted_df)

    elif mode == "topk_then_threshold":
        topk_df = extracted_df.groupby(cve_col, group_keys=False).head(topk).copy()
        filtered_df = apply_threshold(topk_df)

    elif mode == "intersection_and_threshold":
        if not capec_col_in_cve or not capec_col_in_new:
            raise ValueError(
                "mode='intersection_and_threshold' requires both "
                "capec_col_in_cve and capec_col_in_new."
            )
        if capec_col_in_cve not in df_cve.columns:
            raise KeyError(f"Column not found in cve_list_file: {capec_col_in_cve}")
        if capec_col_in_new not in df_new.columns:
            raise KeyError(f"Column not found in new_file: {capec_col_in_new}")

        # Build the ground-truth (CVE, CAPEC) pair set, exploding multi-CAPEC cells
        df_truth = df_cve[[cve_col, capec_col_in_cve]].dropna().copy()
        df_truth[capec_col_in_cve] = (
            df_truth[capec_col_in_cve].astype(str).str.replace(';', ',').str.split(',')
        )
        df_truth = df_truth.explode(capec_col_in_cve)
        df_truth[capec_col_in_cve] = df_truth[capec_col_in_cve].str.strip()

        df_truth = df_truth[df_truth[capec_col_in_cve] != '']
        df_truth = df_truth[df_truth[capec_col_in_cve].str.lower() != 'nan']

        # Set membership gives O(1) lookup when we filter the prediction rows below
        valid_pairs = set(zip(df_truth[cve_col], df_truth[capec_col_in_cve]))

        mask = extracted_df.apply(
            lambda x: (str(x[cve_col]).strip(), str(x[capec_col_in_new]).strip()) in valid_pairs,
            axis=1,
        )
        intersected_df = extracted_df[mask].copy()

        filtered_df = apply_threshold(intersected_df)

    else:
        raise ValueError(
            "mode must be one of: 'topk', 'threshold', "
            "'topk_then_threshold', 'intersection_and_threshold'"
        )

    filtered_df = filtered_df.sort_values(
        by=[cve_col, score_col],
        ascending=[True, False],
    ).copy()

    write_table(filtered_df, output_file)

    print("========== Summary ==========")
    print(f"Unique CVEs in cve_list:       {len(cve_set)}")
    print(f"new_file total rows:           {len(df_new)}")
    print(f"Extracted rows:                {len(extracted_df)}")
    print(f"Extracted unique CVEs:         {extracted_df[cve_col].nunique()}")

    op = '>=' if threshold_inclusive else '>'
    if mode == "topk":
        print(f"Strategy: top-{topk} rows per {cve_col}")
    elif mode == "threshold":
        print(f"Strategy: {score_col} {op} {threshold}")
    elif mode == "topk_then_threshold":
        print(f"Strategy: top-{topk} per CVE, then {score_col} {op} {threshold}")
    elif mode == "intersection_and_threshold":
        print(
            f"Strategy: rows whose CAPEC is in related-capec ({capec_col_in_cve}) "
            f"AND {score_col} {op} {threshold}"
        )

    print(f"Filtered rows:                 {len(filtered_df)}")
    print(f"Filtered unique CVEs:          {filtered_df[cve_col].nunique()}")

    return filtered_df


if __name__ == "__main__":
    extract_and_filter_by_score(
        cve_list_file="data/Fan-out-reduction/Structure_prediction_TL_BRON_capec.xlsx",
        new_file="data/Fan-out-reduction/Ensemble_Weighted_Results_Ranking.xlsx",
        output_file="data/Fan-out-reduction/CEWS_filtered_intersection.xlsx",
        cve_col="CVE-ID",
        score_col="Final_Score",
        mode="intersection_and_threshold",
        threshold=0.5,
        threshold_inclusive=False,
        capec_col_in_cve="relatedcapec",
        capec_col_in_new="CAPEC-ID",
    )