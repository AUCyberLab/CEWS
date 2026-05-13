import pandas as pd
import os


def filter_and_enrich_capec(
    main_file,
    info_files_config,
    output_file,
    main_id_col='CAPEC-ID',
    main_name_col='CAPEC_Name',
):
    """
    Drop deprecated entries from the main CAPEC file and enrich it by left-joining
    selected columns (Impact, Prerequisites, Description, ...) from the info files.
    """
    print(f"[System] Loading main file: {main_file}")
    if not os.path.exists(main_file):
        raise FileNotFoundError(f"[Error] Main file not found: {main_file}")

    df_main = pd.read_excel(main_file)

    # ==========================================
    # 1. Filter deprecated entries
    # ==========================================
    if main_name_col in df_main.columns:
        initial_count = len(df_main)
        # Fill NaN, cast to str and uppercase before matching so the filter is robust to formatting
        mask_deprecated = (
            df_main[main_name_col].fillna('').astype(str).str.upper().str.startswith('DEPRECATED:')
        )
        df_main = df_main[~mask_deprecated].copy()
        filtered_count = initial_count - len(df_main)
        print(f"[System] Filter complete: removed {filtered_count} deprecated entries. {len(df_main)} remain.")
    else:
        print(f"[Warning] Name column '{main_name_col}' not found in main file. Skipping deprecation filter.")

    # Standardize the merge key on the main side
    if main_id_col in df_main.columns:
        df_main[main_id_col] = df_main[main_id_col].astype(str).str.strip()

    # ==========================================
    # 2. Merge info files
    # ==========================================
    for file_key, config in info_files_config.items():
        source_path = config['path']
        source_id_col = config['id_col']
        # Mapping in the form {source_column: final_column_name}
        columns_to_extract = config['extract_cols']

        print(f"[System] Processing {file_key} (source: {source_path})...")

        if not os.path.exists(source_path):
            print(f"[Warning] File not found: {source_path}. Skipping merge for this source.")
            continue

        df_info = pd.read_excel(source_path)

        if source_id_col not in df_info.columns:
            print(f"[Error] Source {source_path} is missing the ID column '{source_id_col}'. Cannot merge.")
            continue

        df_info[source_id_col] = df_info[source_id_col].astype(str).str.strip()

        # Keep only columns that actually exist in the source
        valid_cols = {}
        for src_col, target_col in columns_to_extract.items():
            if src_col in df_info.columns:
                valid_cols[src_col] = target_col
            else:
                print(f"[Warning] Column '{src_col}' missing in {source_path}. Skipping it.")

        if not valid_cols:
            print(f"[Warning] No valid columns to extract from {source_path}. Skipping merge.")
            continue

        cols_to_keep = [source_id_col] + list(valid_cols.keys())
        df_info_subset = df_info[cols_to_keep].copy()

        # Drop duplicate IDs in the source so the merge can't blow up the row count
        df_info_subset = df_info_subset.drop_duplicates(subset=[source_id_col], keep='first')

        df_info_subset = df_info_subset.rename(columns=valid_cols)

        # Align the merge key name with the main file
        if source_id_col != main_id_col:
            df_info_subset = df_info_subset.rename(columns={source_id_col: main_id_col})

        df_main = pd.merge(df_main, df_info_subset, on=main_id_col, how='left')

    # ==========================================
    # 3. Save the merged result
    # ==========================================
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_main.to_excel(output_file, index=False)
    print(f"\n[System] Merge complete. File saved to: {output_file}")
    print(f"Final shape: {df_main.shape}")


if __name__ == "__main__":
    MAIN_FILE = "data/RankingEx/CAPEC_info/CAPEC_parsed_with_level_and_mechanism.xlsx"
    OUTPUT_FILE = "data/RankingEx/CAPEC_info/CAPEC_info_all.xlsx"

    MAIN_ID_COLUMN = "CAPEC-ID"
    MAIN_NAME_COLUMN = "Name"

    # Format: {source_column: final_column_name}
    INFO_CONFIG = {
        "InfoFile_1": {
            "path": "data/RankingEx/CAPEC_info/capec_desc_pre.xlsx",
            "id_col": "CAPEC-ID",
            "extract_cols": {
                "Description": "Description",
                "Prerequisites": "Prerequisites",
            },
        },
        "InfoFile_2": {
            "path": "data/RankingEx/CAPEC_info/CAPEC_Impacts_Only.xlsx",
            "id_col": "CAPEC-ID",
            "extract_cols": {
                "Impact": "Impact",
            },
        },
    }

    filter_and_enrich_capec(
        main_file=MAIN_FILE,
        info_files_config=INFO_CONFIG,
        output_file=OUTPUT_FILE,
        main_id_col=MAIN_ID_COLUMN,
        main_name_col=MAIN_NAME_COLUMN,
    )