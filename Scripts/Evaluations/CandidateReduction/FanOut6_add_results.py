import pandas as pd


def merge_cve_counts(file_a_path, file_b_path, output_path):
    """
    Merge TP_count / FP_count from file B into file A by CVE-ID, prefixing the
    new columns with CEWS_. File A's row count and order are preserved (left join).

    Args:
        file_a_path: Path to file A (Excel).
        file_b_path: Path to file B (Excel).
        output_path: Path where the merged result is written (CSV).
    """
    df_a = pd.read_excel(file_a_path)
    df_b = pd.read_excel(file_b_path)

    required_cols_b = ['CVE-ID', 'TP_count', 'FP_count']
    if 'CVE-ID' not in df_a.columns:
        raise ValueError("File A is missing the 'CVE-ID' column.")
    if not all(col in df_b.columns for col in required_cols_b):
        raise ValueError(f"File B is missing required columns: {required_cols_b}")

    df_b_subset = df_b[required_cols_b].copy().rename(columns={
        'TP_count': 'CEWS_TP_count',
        'FP_count': 'CEWS_FP_count',
    })

    # Left join: rows in A with no match in B get NaN in the new columns
    merged_df = pd.merge(df_a, df_b_subset, on='CVE-ID', how='left')

    merged_df.to_csv(output_path, index=False)
    print(f"Merge complete. Output saved to: {output_path}")

    return merged_df


if __name__ == "__main__":
    merge_cve_counts(
        file_a_path='data/Fan-out-reduction/BRON_tpfp.xlsx',
        file_b_path='data/Fan-out-reduction/CEWS_intersection_tp_fp_per_cve.xlsx',
        output_path='data/Fan-out-reduction/BRON_CEWS_intersection_merged_output.csv',
    )