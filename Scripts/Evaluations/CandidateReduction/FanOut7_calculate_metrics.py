import re
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


def write_excel(per_cve_df, summary_df, output_file):
    """Write per-CVE rows and the summary into two sheets of one Excel file."""
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        per_cve_df.to_excel(writer, index=False, sheet_name="per_cve_metrics")
        summary_df.to_excel(writer, index=False, sheet_name="summary")


def extract_capec_list(text):
    """Pull all CAPEC-N tokens out of a free-form text cell."""
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [x.upper() for x in re.findall(r"CAPEC-\d+", text, flags=re.IGNORECASE)]


def safe_numeric(series, default=0):
    """Coerce a column to numeric; non-numeric values become `default`."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def compute_metrics_from_filtered_file(
    input_file,
    output_file,
    cve_col="CVE-ID",
    relatedcapec_count_col="relatedcapec_count",
    gtcapec_col="GTcapec",
    bron_tp_col="BRON_TP_count",
    bron_fp_col="BRON_FP_count",
    cews_tp_col="CEWS_TP_count",
    cews_fp_col="CEWS_FP_count",
):
    """
    Compute per-CVE and aggregate fan-out reduction metrics comparing BRON
    candidates against CEWS-retained candidates. Writes a two-sheet Excel
    (per_cve_metrics, summary).
    """
    df = read_table(input_file).copy()

    required_cols = [
        cve_col,
        relatedcapec_count_col,
        gtcapec_col,
        bron_tp_col,
        bron_fp_col,
        cews_tp_col,
        cews_fp_col,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Coerce all count columns to numeric ints up front
    for col in (relatedcapec_count_col, bron_tp_col, bron_fp_col, cews_tp_col, cews_fp_col):
        df[col] = safe_numeric(df[col], default=0).astype(int)

    df["GT_count"] = df[gtcapec_col].apply(lambda x: len(extract_capec_list(x)))

    df["BRON_candidate_count"] = df[relatedcapec_count_col]
    df["CEWS_retained_count"] = df[cews_tp_col] + df[cews_fp_col]

    # ----- Per-CVE metrics -----
    df["candidate_reduction"] = (
        (df["BRON_candidate_count"] - df["CEWS_retained_count"]) / df["BRON_candidate_count"]
    )

    df["fp_reduction"] = pd.NA
    mask_fp_valid = df[bron_fp_col] > 0
    df.loc[mask_fp_valid, "fp_reduction"] = (
        (df.loc[mask_fp_valid, bron_fp_col] - df.loc[mask_fp_valid, cews_fp_col])
        / df.loc[mask_fp_valid, bron_fp_col]
    )

    df["gt_recall_cews"] = pd.NA
    mask_gt_valid = df["GT_count"] > 0
    df.loc[mask_gt_valid, "gt_recall_cews"] = (
        df.loc[mask_gt_valid, cews_tp_col] / df.loc[mask_gt_valid, "GT_count"]
    )

    df["tp_retention_over_bron"] = pd.NA
    mask_tp_valid = df[bron_tp_col] > 0
    df.loc[mask_tp_valid, "tp_retention_over_bron"] = (
        df.loc[mask_tp_valid, cews_tp_col] / df.loc[mask_tp_valid, bron_tp_col]
    )

    df["bron_precision"] = pd.NA
    mask_bron_nonzero = (df[bron_tp_col] + df[bron_fp_col]) > 0
    df.loc[mask_bron_nonzero, "bron_precision"] = (
        df.loc[mask_bron_nonzero, bron_tp_col]
        / (df.loc[mask_bron_nonzero, bron_tp_col] + df.loc[mask_bron_nonzero, bron_fp_col])
    )

    df["cews_precision"] = pd.NA
    mask_cews_nonzero = df["CEWS_retained_count"] > 0
    df.loc[mask_cews_nonzero, "cews_precision"] = (
        df.loc[mask_cews_nonzero, cews_tp_col] / df.loc[mask_cews_nonzero, "CEWS_retained_count"]
    )

    # ----- Aggregate summary metrics -----
    avg_bron_candidate_size = df["BRON_candidate_count"].mean()
    avg_cews_retained_size = df["CEWS_retained_count"].mean()

    macro_candidate_reduction = df["candidate_reduction"].mean()
    global_candidate_reduction = 1 - (
        df["CEWS_retained_count"].sum() / df["BRON_candidate_count"].sum()
    )

    macro_fp_reduction = (
        df.loc[mask_fp_valid, "fp_reduction"].astype(float).mean()
        if mask_fp_valid.any() else None
    )

    global_fp_reduction = (
        1 - (df[cews_fp_col].sum() / df[bron_fp_col].sum())
        if df[bron_fp_col].sum() > 0 else None
    )

    macro_gt_recall_cews = (
        df.loc[mask_gt_valid, "gt_recall_cews"].astype(float).mean()
        if mask_gt_valid.any() else None
    )

    micro_gt_recall_cews = (
        df[cews_tp_col].sum() / df["GT_count"].sum()
        if df["GT_count"].sum() > 0 else None
    )

    macro_tp_retention_over_bron = (
        df.loc[mask_tp_valid, "tp_retention_over_bron"].astype(float).mean()
        if mask_tp_valid.any() else None
    )

    micro_tp_retention_over_bron = (
        df[cews_tp_col].sum() / df[bron_tp_col].sum()
        if df[bron_tp_col].sum() > 0 else None
    )

    summary_df = pd.DataFrame([
        {"Metric": "num_cves", "Value": len(df)},
        {"Metric": "avg_bron_candidate_size", "Value": avg_bron_candidate_size},
        {"Metric": "avg_cews_retained_size", "Value": avg_cews_retained_size},
        {"Metric": "macro_candidate_reduction", "Value": macro_candidate_reduction},
        {"Metric": "global_candidate_reduction", "Value": global_candidate_reduction},
        {"Metric": "macro_fp_reduction", "Value": macro_fp_reduction},
        {"Metric": "global_fp_reduction", "Value": global_fp_reduction},
        {"Metric": "macro_gt_recall_cews", "Value": macro_gt_recall_cews},
        {"Metric": "micro_gt_recall_cews", "Value": micro_gt_recall_cews},
        {"Metric": "macro_tp_retention_over_bron", "Value": macro_tp_retention_over_bron},
        {"Metric": "micro_tp_retention_over_bron", "Value": micro_tp_retention_over_bron},
    ])

    per_cve_cols = [
        cve_col,
        "BRON_candidate_count",
        "CEWS_retained_count",
        "GT_count",
        bron_tp_col,
        bron_fp_col,
        cews_tp_col,
        cews_fp_col,
        "candidate_reduction",
        "fp_reduction",
        "gt_recall_cews",
        "tp_retention_over_bron",
        "bron_precision",
        "cews_precision",
    ]

    per_cve_df = df[per_cve_cols].copy()
    write_excel(per_cve_df, summary_df, output_file)

    print(f"Output saved to: {output_file}")
    print(summary_df)


if __name__ == "__main__":
    compute_metrics_from_filtered_file(
        input_file="data/Fan-out-reduction/BRON_CEWS_intersection_merged_GTinRelated.xlsx",
        output_file="data/Fan-out-reduction/BRON_CEWS_intersection_merged_GTinRelated_metrics.xlsx",
    )