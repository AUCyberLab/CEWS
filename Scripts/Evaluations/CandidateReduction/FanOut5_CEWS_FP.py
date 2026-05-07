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


def normalize_binary_label(x):
    """Coerce label values into 0/1 (defaults to 0 for anything unrecognized)."""
    if pd.isna(x):
        return 0

    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "yes"}:
        return 1
    if s in {"0", "0.0", "false", "no"}:
        return 0
    return 0


def summarize_tp_fp_per_cve_from_positive_predictions(
    input_file,
    output_file,
    cve_col="CVE-ID",
    label_col="Label",
    tp_col="TP_count",
    fp_col="FP_count",
):
    """
    Aggregate TP / FP counts per CVE for an input file that already contains
    only predicted-positive rows.

    For each CVE-ID:
      TP = number of rows with label=1
      FP = number of rows with label=0

    Output has one row per CVE-ID.
    """
    df = read_table(input_file)

    if cve_col not in df.columns:
        raise KeyError(f"Column not found in input file: {cve_col}")
    if label_col not in df.columns:
        raise KeyError(f"Column not found in input file: {label_col}")

    df[cve_col] = df[cve_col].astype(str).str.strip()
    df[label_col] = df[label_col].apply(normalize_binary_label)

    # Every row in the input is assumed to already be a predicted-positive
    df[tp_col] = (df[label_col] == 1).astype(int)
    df[fp_col] = (df[label_col] == 0).astype(int)

    summary_df = (
        df.groupby(cve_col, as_index=False)
        .agg(**{
            tp_col: (tp_col, "sum"),
            fp_col: (fp_col, "sum"),
        })
        .copy()
    )

    write_table(summary_df, output_file)

    print("========== Summary ==========")
    print(f"Input total rows:    {len(df)}")
    print(f"Unique CVEs:         {df[cve_col].nunique()}")
    print(f"Output total rows:   {len(summary_df)}")
    print(summary_df.head(10))

    return summary_df


if __name__ == "__main__":
    summarize_tp_fp_per_cve_from_positive_predictions(
        input_file="data/Fan-out-reduction/CEWS_filtered_intersection.xlsx",
        output_file="data/Fan-out-reduction/CEWS_intersection_tp_fp_per_cve.xlsx",
        cve_col="CVE-ID",
        label_col="Label",
    )