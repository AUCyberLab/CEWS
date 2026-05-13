import pandas as pd
import os


def build_cross_dataset(dataset_a_path, dataset_b_path, output_path):
    """Cross-join CVE rows from dataset A with every CAPEC row in dataset B."""
    print("================ Building cross dataset ================")

    if not os.path.exists(dataset_a_path):
        print(f"[Error] Dataset A not found: {dataset_a_path}")
        return
    if not os.path.exists(dataset_b_path):
        print(f"[Error] Dataset B not found: {dataset_b_path}")
        return

    print("[System] Loading dataset A...")
    df_a = pd.read_excel(dataset_a_path)
    print("[System] Loading dataset B...")
    df_b = pd.read_excel(dataset_b_path)

    # Tolerate the two common spellings of the description column
    cve_id_col = 'CVE_ID'
    if 'CVE_Description' in df_a.columns:
        desc_col = 'CVE_Description'
    elif 'CVE Description' in df_a.columns:
        desc_col = 'CVE Description'
    else:
        desc_col = None

    if cve_id_col not in df_a.columns or desc_col is None:
        print(f"[Error] Dataset A is missing required CVE-ID or CVE_Description column. Found: {df_a.columns.tolist()}")
        return

    # Keep only the two needed columns and force the A side to be unique by CVE-ID
    df_a_subset = df_a[[cve_id_col, desc_col]].drop_duplicates(subset=[cve_id_col]).copy()
    df_a_subset = df_a_subset.rename(columns={cve_id_col: 'CVE-ID', desc_col: 'CVE_Description'})
    df_a_subset['CVE-ID'] = df_a_subset['CVE-ID'].astype(str).str.strip()

    len_a = len(df_a_subset)
    len_b = len(df_b)
    expected_rows = len_a * len_b

    print(f"[System] Unique CVEs in A: {len_a}")
    print(f"[System] CAPEC entries in B: {len_b}")
    print(f"[System] Expected combined rows: {expected_rows}")

    if expected_rows > 1040000:
        print("[Warning] Expected row count exceeds Excel's per-sheet limit (1,048,576).")
        print("          Switch output_path to a .csv extension to avoid crashes when writing.")

    print("[System] Performing cross join...")
    df_combined = pd.merge(df_a_subset, df_b, how='cross')

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[System] Writing output (large datasets may take a while)...")
    # Pick the writer based on the extension so we don't hit Excel's row limit when it's exceeded
    if output_path.endswith('.csv'):
        df_combined.to_csv(output_path, index=False, encoding='utf-8')
    else:
        df_combined.to_excel(output_path, index=False)

    print("[System] Done.")
    print(f"Final shape: {df_combined.shape}")
    print(f"File saved to: {output_path}")


if __name__ == "__main__":
    # Dataset A: contains CVE-ID and CVE_Description
    DATASET_A = "data/RankingEx/ThreatLinker_cve-capec-mapping.xlsx"

    # Dataset B: contains all CAPEC info (CAPEC-ID, Mechanism, Impact, ...)
    DATASET_B = "data/RankingEx/CAPEC_info/CAPEC_parsed.xlsx"

    # If the cross-product exceeds ~1M rows, change this to a .csv extension
    OUTPUT_FILE = "data/RankingEx/TL_CVE_CAPEC_cross_dataset.xlsx"

    build_cross_dataset(DATASET_A, DATASET_B, OUTPUT_FILE)