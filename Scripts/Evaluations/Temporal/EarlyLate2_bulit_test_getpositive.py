import pandas as pd


def read_table(file_path):
    return pd.read_excel(str(file_path))


def write_table(df, file_path):
    df.to_excel(str(file_path), index=False)


def is_positive_label(x):
    """Return True if x represents a positive label (1, 1.0, true, yes)."""
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "1.0", "true", "yes"}


def extract_positive_rows(
    input_file,
    output_file,
    label_col="Label",
    cve_id_col="CVE-ID",
    capec_id_col="CAPEC-ID",
):
    """
    Extract all positive-label rows from an input Excel file, sort them by
    (CVE-ID, CAPEC-ID), and write the result to a new file.

    Args:
        input_file:   Source Excel file.
        output_file:  Destination Excel file for the positive subset.
        label_col:    Name of the label column.
        cve_id_col:   Name of the CVE-ID column.
        capec_id_col: Name of the CAPEC-ID column.
    """
    df = read_table(input_file)

    # Strip whitespace from column names so accidental leading/trailing spaces don't break lookups
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {label_col, cve_id_col, capec_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    positive_df = df[df[label_col].apply(is_positive_label)].copy()

    if positive_df.empty:
        raise ValueError("No positive-label rows found in the input file.")

    positive_df = positive_df[
        positive_df[cve_id_col].notna() & positive_df[capec_id_col].notna()
    ].copy()

    if positive_df.empty:
        raise ValueError("Positive rows have no usable CVE-ID / CAPEC-ID values.")

    positive_df = positive_df.sort_values(
        by=[cve_id_col, capec_id_col],
        ascending=[True, True],
    ).reset_index(drop=True)

    write_table(positive_df, output_file)

    print(f"Total input rows:     {len(df)}")
    print(f"Positive rows kept:   {len(positive_df)}")
    print(f"Unique CVEs:          {positive_df[cve_id_col].nunique()}")
    print(f"Output file:          {output_file}")


if __name__ == "__main__":
    extract_positive_rows(
        input_file="data/Early_late_dataset/Description_test.xlsx",
        output_file="data/Early_late_dataset/Description_test_positives_only.xlsx",
        label_col="Label",
        cve_id_col="CVE-ID",
        capec_id_col="CAPEC-ID",
    )